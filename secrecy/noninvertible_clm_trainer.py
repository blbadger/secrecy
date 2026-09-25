import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import rearrange
import transformers
from transformers import AutoTokenizer

from datasets import load_dataset, load_from_disk, concatenate_datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
from prettytable import PrettyTable
from safetensors.torch import save_file, load_model
from safetensors import safe_open
import safetensors
import datasets
from datasets import Dataset
import warnings
import shutil
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model

from transformer_autoencoder import AbbreviatedModel, SuffixModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from transformer_autoencoder import SplitModel, AllAutoencodingTransformer, SecretTransformer
from noninvertible_clm import NonInvertibleTransformer, ParallelNoninvertibleModel, DualRootParallelModel
from secret_decoder import SecretDecoder
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, is_compiled_module
from transformers import get_linear_schedule_with_warmup
from accelerate.utils import TorchDynamoPlugin

from safetensors.torch import save_file, save_model, load_model, load_file
import os


class LossLogger:
    """Keeps running averages of per-step losses and records one averaged row every `log_every` steps.

    The full history is held in memory (it is tiny: one row per `log_every` steps) and written
    into each checkpoint directory by save(), so every checkpoint carries the log up to that step.
    """

    def __init__(self, log_every=500, enabled=True, resume_path=None):
        self.log_every = log_every
        self.enabled = enabled
        self.sums = {}    # name -> running sum (kept on device, no sync until a row is recorded)
        self.counts = {}  # name -> number of steps accumulated in the current window
        self.rows = []    # finished rows: {"step": ..., "<loss_name>": window average, ...}
        if enabled and resume_path and os.path.exists(resume_path):
            with open(resume_path) as f:
                self.rows = [json.loads(line) for line in f if line.strip()]

    def log(self, step, **losses):
        if not self.enabled:
            return
        for k, v in losses.items():
            v = torch.as_tensor(v).detach().float()  # detach: never hold onto the autograd graph
            self.sums[k] = self.sums[k] + v if k in self.sums else v
            self.counts[k] = self.counts.get(k, 0) + 1
        if step % self.log_every == 0:
            row = {"step": step}
            for k in sorted(self.sums):
                row[k] = (self.sums[k] / self.counts[k]).item()  # one device->host sync per row
            self.rows.append(row)
            tqdm.write(str(row))
            self.sums.clear()
            self.counts.clear()

    def save(self, path):
        """Write the full log to `path` (atomic replace)."""
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, path)


class EvalLogger:
    """Records one row per evaluation call (no windowing, since eval already happens
    infrequently). Written into each checkpoint directory the same way as LossLogger.
    """

    def __init__(self, enabled=True, resume_path=None):
        self.enabled = enabled
        self.rows = []
        if enabled and resume_path and os.path.exists(resume_path):
            with open(resume_path) as f:
                self.rows = [json.loads(line) for line in f if line.strip()]

    def log(self, step, **metrics):
        if not self.enabled:
            return
        row = {"step": step}
        for k, v in metrics.items():
            v = torch.as_tensor(v).detach().float().item() if torch.is_tensor(v) else float(v)
            row[k] = v
        self.rows.append(row)
        tqdm.write(str(row))

    def save(self, path):
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, path)


def toggle_grads(module, bool=True):
    for _, param in module.named_parameters():
        param.requires_grad = bool
    return

def save_checkpoint(
        accelerator, 
        model, 
        inverter, 
        model_optimizer, 
        inverter_optimizer, 
        clm_scheduler, 
        inverter_scheduler, 
        step, 
        checkpoint_dir
    ):
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

    unwrapped_clm_model = accelerator.unwrap_model(model)
    unwrapped_inverter = accelerator.unwrap_model(inverter)

    # unwrap compiled module (_orig_mod)
    unwrapped_clm_model = unwrapped_clm_model._orig_mod if is_compiled_module(unwrapped_clm_model) else unwrapped_clm_model
    unwrapped_inverter = unwrapped_inverter._orig_mod if is_compiled_module(unwrapped_inverter) else unwrapped_inverter
    if accelerator.is_main_process:
        # model weights -> safetensors (must be contiguous + on CPU)
        save_model(unwrapped_clm_model,  os.path.join(checkpoint_dir, "clm_model.safetensors"))
        save_model(unwrapped_inverter, os.path.join(checkpoint_dir, "inverter.safetensors"))

        # optimizer + scheduler -> torch.save (non-tensor state: step counts, betas, etc.)
        torch.save(model_optimizer.state_dict(), os.path.join(checkpoint_dir, "model_optimizer.pt"))
        torch.save(inverter_optimizer.state_dict(), os.path.join(checkpoint_dir, "inverter_optimizer.pt"))

        torch.save(clm_scheduler.state_dict(), os.path.join(checkpoint_dir, "clm_scheduler.pt"))
        torch.save(inverter_scheduler.state_dict(), os.path.join(checkpoint_dir, "inverter_scheduler.pt"))

        torch.save({"step": step}, os.path.join(checkpoint_dir, "training_state.pt"))
    accelerator.wait_for_everyone()
    return

def load_checkpoint(
        accelerator, 
        model, 
        inverter, 
        model_optimizer, 
        inverter_optimizer, 
        clm_scheduler, 
        inverter_scheduler, 
        checkpoint_dir
    ):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_inverter = accelerator.unwrap_model(inverter)

    load_model(unwrapped_model, os.path.join(checkpoint_dir, "clm_model.safetensors"))
    load_model(unwrapped_inverter, os.path.join(checkpoint_dir, "inverter.safetensors"))

    model_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model_optimizer.pt"), map_location="cpu"))
    inverter_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "inverter_optimizer.pt"), map_location="cpu"))
    clm_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "clm_scheduler.pt"), map_location="cpu"))
    inverter_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "inverter_scheduler.pt"), map_location="cpu"))

    training_state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location="cpu")
    return training_state["step"]

@torch.no_grad()
def evaluate_noninvertibility(
        step, 
        noninvertible_clm, 
        inverter, 
        test_dataloader, 
        n_tokens_obfuscated, 
        tokenizer, 
        accelerator,
        eval_logger=None,
    ):
    running_clm_loss = 0
    running_inverter_loss = 0
    running_inverter_correct = 0
    running_inverter_total = 0
    for i, batch in enumerate(test_dataloader):
        inputs, labels = torch.stack(batch['input_ids'], dim=0).T, torch.stack(batch['input_ids'], dim=0).T
        labels = torch.where(labels==tokenizer.pad_token_id, -100, labels) # mask pad token losses
        with accelerator.autocast():
            noninvertible_clm_loss, noninvertible_inversion_loss, noninvertible_embedding = noninvertible_clm(inputs, labels=labels)
        running_clm_loss += noninvertible_clm_loss.detach()

        with accelerator.autocast():
            ignore_index = - 100 
            nonpad_tokens = labels[:, :n_tokens_obfuscated] != ignore_index
            if isinstance(noninvertible_clm._orig_mod.module, DualRootParallelModel):
                inverter_loss, inverter_logits = inverter(inputs_embeds=noninvertible_embedding.detach()[:, :n_tokens_obfuscated], labels=labels[:, :n_tokens_obfuscated])# with reduction
            else:
                inverter_loss, inverter_logits = inverter(inputs_embeds=noninvertible_embedding.detach(), labels=labels)
                inverter_loss = inverter_loss[:, :n_tokens_obfuscated].sum() / nonpad_tokens.sum()

        running_inverter_loss += inverter_loss.detach()

        # token-level accuracy of the inverter's secret-token predictions
        inverter_preds = inverter_logits[:, :n_tokens_obfuscated].argmax(dim=-1)
        running_inverter_correct += ((inverter_preds == labels[:, :n_tokens_obfuscated]) & nonpad_tokens).sum().detach()
        running_inverter_total += nonpad_tokens.sum().detach()

    eval_inverter_accuracy = (running_inverter_correct / running_inverter_total).item() if running_inverter_total > 0 else float('nan')

    if accelerator.is_main_process:
        eval_inverter_loss = round(float(running_inverter_loss)/len(test_dataloader), 4)
        eval_clm_loss = round(float(running_clm_loss)/len(test_dataloader), 4)
        tqdm.write(f'Step {step} Evaluation Inverter loss: {eval_inverter_loss}') 
        tqdm.write(f'Step {step} Evaluation CausalLM Loss: {eval_clm_loss}')
        tqdm.write(f'Step {step} Evaluation Inverter accuracy: {round(eval_inverter_accuracy, 4)}')
        if eval_logger is not None:
            eval_logger.log(
                step,
                eval_clm_loss=eval_clm_loss,
                eval_inverter_loss=eval_inverter_loss,
                eval_inverter_accuracy=eval_inverter_accuracy,
            )
    return


def train_noninvertible_clm(
        train_dataloader, 
        test_dataloader, 
        noninvertible_clm, 
        noninvertible_clm_optimizer, 
        inverter, 
        inverter_optimizer, 
        loss_fn, 
        accelerator, 
        tokenizer=None,
        max_grad_norm=1.,
        clm_scheduler=None,
        inverter_scheduler=None,
        checkpoint_dir=None,
        save_every=10000,
        start_step=0,
        steps=200000,
        train_clm=True,
        train_inverter=True,
        train_for_noninv=True,
        evaluate_every=10000,
        log_every=500,
        n_tokens_obfuscated=128
    ):
    noninvertible_clm.train()
    inverter.train()
    logger = LossLogger(
        log_every=log_every,
        enabled=accelerator.is_main_process,
        # when resuming, pick up the log stored in the checkpoint we resume from
        resume_path=os.path.join(checkpoint_dir, f"step_{start_step}", "loss_log.jsonl") if start_step else None,
    )
    eval_logger = EvalLogger(
        enabled=accelerator.is_main_process,
        resume_path=os.path.join(checkpoint_dir, f"step_{start_step}", "eval_log.jsonl") if start_step else None,
    )
    toggle_grads(inverter, bool=False)
    global_step = start_step
    pbar = tqdm(total=steps, initial=global_step, desc='global step')
    while True:
        for i, batch in enumerate(train_dataloader):
            if global_step > steps:
                logger.save(os.path.join(checkpoint_dir, "loss_log.jsonl"))  # final log
                eval_logger.save(os.path.join(checkpoint_dir, "eval_log.jsonl"))
                return
            global_step += 1
            if accelerator.is_main_process:
                pbar.update(1)
            inputs, labels = torch.stack(batch['input_ids'], dim=0).T, torch.stack(batch['input_ids'], dim=0).T
            labels = torch.where(labels==tokenizer.pad_token_id, -100, labels) # mask pad token losses
            if train_clm:
                toggle_grads(inverter, bool=False)
                with accelerator.autocast():
                    noninvertible_clm_loss, noninvertible_inversion_loss, noninvertible_embedding = noninvertible_clm(inputs, labels=labels)
                if train_for_noninv:
                    total_noninv_loss = noninvertible_clm_loss - noninvertible_inversion_loss
                else:
                    total_noninv_loss = noninvertible_clm_loss
                noninvertible_clm_optimizer.zero_grad()
                accelerator.backward(total_noninv_loss)

                clm_grad_norm = None
                if accelerator.sync_gradients:
                    clm_grad_norm = accelerator.clip_grad_norm_(noninvertible_clm.parameters(), max_grad_norm)
                clm_lr = noninvertible_clm_optimizer.param_groups[0]["lr"]
                noninvertible_clm_optimizer.step()
                if accelerator.sync_gradients:
                    clm_scheduler.step()
            else:
                with accelerator.autocast() and torch.no_grad():
                    _, _, noninvertible_embedding = noninvertible_clm(inputs, labels=labels)
            
            if train_inverter:
                toggle_grads(inverter, bool=True)
                with accelerator.autocast():
                    if isinstance(noninvertible_clm._orig_mod.module, DualRootParallelModel):
                        inverter_loss, _ = inverter(inputs_embeds=noninvertible_embedding.detach()[:, :n_tokens_obfuscated], labels=labels[:, :n_tokens_obfuscated])# with reduction
                    else:
                        inverter_loss, _ = inverter(inputs_embeds=noninvertible_embedding.detach(), labels=labels)
                        ignore_index = - 100
                        nonpad_tokens = labels[:, :n_tokens_obfuscated] != ignore_index
                        inverter_loss = inverter_loss[:, :n_tokens_obfuscated].sum() / nonpad_tokens.sum()
                inverter_optimizer.zero_grad()
                accelerator.backward(inverter_loss)
                inverter_grad_norm = None
                if accelerator.sync_gradients:
                    inverter_grad_norm = accelerator.clip_grad_norm_(inverter.parameters(), max_grad_norm)
                inverter_lr = inverter_optimizer.param_groups[0]["lr"]
                inverter_optimizer.step()
                if accelerator.sync_gradients:
                    inverter_scheduler.step()
            else:
                inverter_loss = 0
                inverter_grad_norm = None
                inverter_lr = inverter_optimizer.param_groups[0]["lr"]
            
            toggle_grads(inverter, bool=False)

            # accumulate losses; a window-averaged row is recorded every `log_every` steps
            step_losses = {"inverter_loss": inverter_loss, "inverter_lr": inverter_lr}
            if inverter_grad_norm is not None:
                step_losses["inverter_grad_norm"] = inverter_grad_norm
            if train_clm:
                step_losses.update(
                    clm_loss=noninvertible_clm_loss,
                    inversion_loss=noninvertible_inversion_loss,
                    total_noninv_loss=total_noninv_loss,
                    clm_lr=clm_lr,
                )
                if clm_grad_norm is not None:
                    step_losses["clm_grad_norm"] = clm_grad_norm
            logger.log(global_step, **step_losses)

            if global_step % save_every == 0:
                save_checkpoint(
                        accelerator, 
                        noninvertible_clm, 
                        inverter,
                        noninvertible_clm_optimizer, 
                        inverter_optimizer, 
                        clm_scheduler,
                        inverter_scheduler,
                        global_step, 
                        os.path.join(checkpoint_dir, f"step_{global_step}")
                    )
                logger.save(os.path.join(checkpoint_dir, f"step_{global_step}", "loss_log.jsonl"))
                eval_logger.save(os.path.join(checkpoint_dir, f"step_{global_step}", "eval_log.jsonl"))
            if global_step % evaluate_every == 0:
                evaluate_noninvertibility(
                    global_step,
                    noninvertible_clm,
                    inverter,
                    test_dataloader,
                    n_tokens_obfuscated,
                    tokenizer,
                    accelerator,
                    eval_logger=eval_logger,
                )
                noninvertible_clm.train()
                inverter.train()
    return

def unwrap_state_dict(state_dict):
    #For loading state dicts of compiled models before compilation
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def init_noninvertible_transformer(tokenizer, 
        vocab_size, 
        context_length=512, 
        decoder_dim=512, 
        inverter_layers=8, 
        model_layers=16, 
        n_heads=4
    ):
    encoder_config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': inverter_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    # inverter model definition
    configuration = LlamaConfig(**encoder_config_kwargs)
    model = LlamaForCausalLM(configuration)
    inverter = SecretDecoder(vocab_size, decoder_dim, model)

    # Noninvertible model definition
    encoder_config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': model_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    encoder_configuration = LlamaConfig(**encoder_config_kwargs)
    encoder_model = LlamaForCausalLM(encoder_configuration)
    original_clm = encoder_model

    clm_head = encoder_model.lm_head
    encoder_state_dict = encoder_model.model.state_dict()
    clm_wte = encoder_model.model.embed_tokens
    split_model = SplitModel(encoder_configuration)
    split_model.config.num_hidden_layers = model_layers

    model = NonInvertibleTransformer(
        vocab_size, 
        decoder_dim, 
        split_model, 
        inverter,
        clm_head=clm_head,
    )
    return model

def init_noninvertible_parallelmodel(
    tokenizer, 
    vocab_size, 
    n_tokens_obfuscated,
    compress_provider_factor=1,
    route_method='embedding_split',
    context_length=512, 
    decoder_dim=512, 
    inverter_layers=8, 
    unified_encoder_layers=4,
    provider_model_layers=16, 
    client_encoder_layers=3,
    unified_decoder_layers=4,
    n_heads=4
    ):
    # inversion model specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': inverter_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': n_tokens_obfuscated,
        'is_causal': False
    }

    configuration = LlamaConfig(**config_kwargs)
    model = LlamaForCausalLM(configuration)
    inverter = SecretDecoder(vocab_size, decoder_dim, model, reduce_loss=False)

    # unified encoder specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': unified_encoder_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    encoder_configuration = LlamaConfig(**config_kwargs)
    unified_encoder = LlamaModel(encoder_configuration)

    # client encoder specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': client_encoder_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    configuration = LlamaConfig(**config_kwargs)
    client_encoder = LlamaModel(configuration)

    # provider model specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': provider_model_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    configuration = LlamaConfig(**config_kwargs)
    provider_model = LlamaModel(configuration)

    # unified decoder model specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': unified_decoder_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    configuration = LlamaConfig(**config_kwargs)
    unified_decoder = LlamaModel(configuration)

    model = ParallelNoninvertibleModel(
        vocab_size, 
        decoder_dim, 
        provider_model, 
        inverter, 
        tokenized_length=context_length, 
        clm_loss_only=False,
        parallel_encoder=client_encoder,
        unified_decoder=unified_decoder,
        unified_encoder=unified_encoder,
        n_tokens_obfuscated=n_tokens_obfuscated,
        compress_provider_factor=compress_provider_factor,
        route_method=route_method,
    )
    return model, inverter

def init_dualroot_parallelmodel(
    tokenizer, 
    vocab_size, 
    n_tokens_obfuscated,
    compress_secret_factor=1,
    unroll_secret_embedding=False,
    mask_secret_tokens=False,
    context_length=512, 
    decoder_dim=512, 
    inverter_layers=8, 
    secret_encoder_layers=3,
    provider_model_layers=16, 
    client_encoder_layers=3,
    unified_decoder_layers=4,
    n_heads=4
    ):
    # inversion model specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': inverter_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': n_tokens_obfuscated
    }

    configuration = LlamaConfig(**config_kwargs)
    model = LlamaForCausalLM(configuration)
    inverter = SecretDecoder(vocab_size, decoder_dim, model)

    # secret encoder specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': secret_encoder_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    encoder_configuration = LlamaConfig(**config_kwargs)
    secret_encoder = LlamaModel(encoder_configuration)

    # client encoder specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': client_encoder_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    configuration = LlamaConfig(**config_kwargs)
    client_encoder = LlamaModel(configuration)

    # provider model specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': provider_model_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    configuration = LlamaConfig(**config_kwargs)
    provider_model = LlamaModel(configuration)

    # unified decoder model specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': unified_decoder_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }
    configuration = LlamaConfig(**config_kwargs)
    unified_decoder = LlamaModel(configuration)

    model = DualRootParallelModel(
        vocab_size, 
        decoder_dim, 
        provider_model, 
        inverter, 
        tokenized_length=context_length, 
        clm_loss_only=False,
        parallel_encoder=client_encoder,
        unified_decoder=unified_decoder,
        secret_encoder=secret_encoder,
        n_tokens_obfuscated=n_tokens_obfuscated,
        compress_secret_factor=compress_secret_factor,
        unroll_secret_embedding=unroll_secret_embedding,
        mask_secret_tokens=mask_secret_tokens
    )
    return model, inverter


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')

    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    n_tokens_obfuscated = 128
    #compress_provider_factor = 1
    #route_method = 'unroll_embedding'
    #model, inverter = init_noninvertible_parallelmodel(tokenizer, vocab_size, n_tokens_obfuscated, compress_provider_factor=compress_provider_factor, route_method=route_method)

    compress_secret_factor = 1
    unroll_secret_embedding = True
    mask_secret_tokens = False
    model, inverter = init_dualroot_parallelmodel(
        tokenizer, 
        vocab_size, 
        n_tokens_obfuscated, 
        compress_secret_factor=compress_secret_factor, 
        unroll_secret_embedding=unroll_secret_embedding,
        mask_secret_tokens=mask_secret_tokens
    )


    train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-8k"

    # load datasets and duplicate entries
    train_dataset = load_from_disk(train_path)
    test_dataset = load_from_disk(test_path)

    learning_rate = 2e-4
    num_gpus = torch.cuda.device_count()
    batch_size = 128 // num_gpus
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    inverter_optimizer = torch.optim.AdamW(inverter.parameters(), lr=learning_rate)

    num_steps = 200000
    total_training_steps = num_steps * num_gpus # num_gpu steps taken for each 

    model_scheduler = get_linear_schedule_with_warmup(
        model_optimizer,
        num_warmup_steps=500,
        num_training_steps=total_training_steps,
    )

    inverter_scheduler = get_linear_schedule_with_warmup(
        inverter_optimizer,
        num_warmup_steps=500,
        num_training_steps=total_training_steps
    )

    # Configure the compilation backend
    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",
        mode="default",
        fullgraph=False,
        dynamic=False
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(mixed_precision='fp16', dynamo_plugin=dynamo_plugin, kwargs_handlers=[ddp_kwargs])
    model, model_optimizer, inverter, inverter_optimizer, train_dataloader, test_dataloader, model_scheduler, inverter_scheduler = accelerator.prepare(
        model, 
        model_optimizer, 
        inverter, 
        inverter_optimizer, 
        train_dataloader, 
        test_dataloader,
        model_scheduler,
        inverter_scheduler
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    n_devices = accelerator.num_processes
    checkpoint_dir = f"{data_root}/parallelmodel_dualroot_unroll_noni_b{batch_size}x{n_devices}"

    print (f"training model, saving to {checkpoint_dir}")
    # save driver code snapshot in checkpoint dir
    code_path = os.path.abspath(__file__)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    shutil.copy(code_path, checkpoint_dir)
    
    train_noninvertible_clm(
        train_dataloader, 
        test_dataloader, 
        model, 
        model_optimizer, 
        inverter, 
        inverter_optimizer, 
        loss_fn,
        accelerator,
        tokenizer=tokenizer,
        clm_scheduler=model_scheduler, 
        inverter_scheduler=inverter_scheduler, 
        checkpoint_dir=checkpoint_dir,
        steps=num_steps,
        train_clm = True,
        train_inverter=True,
        train_for_noninv=True,
        n_tokens_obfuscated=n_tokens_obfuscated
    )   

