import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

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
from noninvertible_clm import NonInvertibleTransformer
from secret_decoder import SecretDecoder
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from noninvertible_clm import ParallelNoninvertibleModel

from transformers import get_linear_schedule_with_warmup
from accelerate.utils import TorchDynamoPlugin

from safetensors.torch import save_file, save_model, load_model, load_file
import os


class LossLogger:
    """Buffers per-step losses in memory and appends them to a JSONL file on flush()."""

    def __init__(self, path, enabled=True, resume_step=None):
        self.path = path
        self.enabled = enabled
        self.buffer = []  # list of (step, {name: tensor_or_float})
        if enabled:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            if resume_step:
                self._truncate_after(resume_step)

    def log(self, step, **losses):
        if self.enabled:
            # detach so we never hold onto the autograd graph; no .item() here (avoids a GPU sync)
            self.buffer.append((step, {k: torch.as_tensor(v).detach() for k, v in losses.items()}))

    def flush(self):
        if not self.enabled or not self.buffer:
            return
        keys = {k for _, d in self.buffer for k in d}
        cols = {}
        for k in keys:  # one stack + one device->host copy per loss name
            idx = [i for i, (_, d) in enumerate(self.buffer) if k in d]
            vals = torch.stack([self.buffer[i][1][k].float() for i in idx]).cpu().tolist()
            cols[k] = dict(zip(idx, vals))
        with open(self.path, "a") as f:
            for i, (step, _) in enumerate(self.buffer):
                row = {"step": step, **{k: cols[k][i] for k in sorted(keys) if i in cols[k]}}
                f.write(json.dumps(row) + "\n")
        self.buffer.clear()

    def _truncate_after(self, step):
        """On resume, drop any logged steps past the checkpoint we're resuming from."""
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            rows = [line for line in f if line.strip() and json.loads(line)["step"] <= step]
        with open(self.path, "w") as f:
            f.writelines(rows)


def toggle_grads(module, bool=True):
    for _, param in module.named_parameters():
        param.requires_grad = bool
    return

def save_checkpoint(accelerator, model, inverter, model_optimizer, inverter_optimizer, clm_scheduler, inverter_scheduler, step, checkpoint_dir):
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

    unwrapped_clm_model = accelerator.unwrap_model(model)
    unwrapped_inverter = accelerator.unwrap_model(inverter)
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

def load_checkpoint(accelerator, model, inverter, model_optimizer, inverter_optimizer, 
                     clm_scheduler, inverter_scheduler, checkpoint_dir):
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
def evaluate_noninvertibility(noninvertible_clm, inverter, test_dataloader):
    running_clm_loss = 0
    running_inverter_loss = 0
    for i, batch in enumerate(test_dataloader):
        inputs, labels = torch.stack(batch['input_ids'], dim=0).T, torch.stack(batch['input_ids'], dim=0).T
        labels = torch.where(labels==tokenizer.pad_token_id, -100, labels) # mask pad token losses
        with accelerator.autocast():
            noninvertible_clm_loss, noninvertible_inversion_loss, noninvertible_embedding = noninvertible_clm(inputs, labels=labels)
        running_clm_loss += noninvertible_clm_loss.detach()

        with accelerator.autocast():
            inverter_loss, _ = inverter(inputs_embeds=noninvertible_embedding.detach(), labels=labels)
        running_inverter_loss += inverter_loss.detach()

    if accelerator.is_main_process:
        tqdm.write(f'Evaluation Inverter loss: {round(float(running_inverter_loss)/len(test_dataloader), 4)}') 
        tqdm.write(f'Evaluation CausalLM Loss: {round(float(running_clm_loss)/len(test_dataloader), 4)}')
    return


def train_noninvertible_clm(
        train_dataloader, 
        test_dataloader, 
        noninvertible_clm, 
        noninvertible_clm_optimizer, 
        inverter, 
        inverter_optimizer, 
        loss_fn, 
        max_grad_norm=1.,
        clm_scheduler=None,
        inverter_scheduler=None,
        checkpoint_dir=None,
        save_every=10000,
        start_step=0,
        steps=200000,
        train_clm=True,
        evaluate_every=10000
    ):
    noninvertible_clm.train()
    inverter.train()
    logger = LossLogger(
        os.path.join(checkpoint_dir, "loss_log.jsonl"),
        enabled=accelerator.is_main_process,
        resume_step=start_step,
    )
    toggle_grads(inverter, bool=False)
    global_step = start_step
    pbar = tqdm(total=steps, initial=global_step, desc='global step')
    while True:
        for i, batch in enumerate(train_dataloader):
            if global_step > steps:
                logger.flush()
                return
            global_step += 1
            if accelerator.is_main_process:
                pbar.update(1)
            inputs, labels = torch.stack(batch['input_ids'], dim=0).T, torch.stack(batch['input_ids'], dim=0).T
            labels = torch.where(labels==tokenizer.pad_token_id, -100, labels) # mask pad token losses
            if train_clm:
                with accelerator.autocast():
                    noninvertible_clm_loss, noninvertible_inversion_loss, noninvertible_embedding = noninvertible_clm(inputs, labels=labels)
                total_noninv_loss = noninvertible_clm_loss - 0.6*noninvertible_inversion_loss
                noninvertible_clm_optimizer.zero_grad()
                accelerator.backward(total_noninv_loss)
                # TODO: define running grad norm
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(noninvertible_clm.parameters(), max_grad_norm)
                noninvertible_clm_optimizer.step()
                if accelerator.sync_gradients:
                    clm_scheduler.step()
            else:
                with accelerator.autocast() and torch.no_grad():
                    _, _, noninvertible_embedding = noninvertible_clm(inputs, labels=labels)

            toggle_grads(inverter, bool=True)
            with accelerator.autocast():
                inverter_loss, _ = inverter(inputs_embeds=noninvertible_embedding.detach(), labels=labels)
            inverter_optimizer.zero_grad()
            accelerator.backward(inverter_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(inverter.parameters(), max_grad_norm)
            inverter_optimizer.step()
            if accelerator.sync_gradients:
                inverter_scheduler.step()
            toggle_grads(inverter, bool=False)

            # log per-step losses (buffered in memory, written to disk on checkpoint save)
            step_losses = {"inverter_loss": inverter_loss}
            if train_clm:
                step_losses.update(
                    clm_loss=noninvertible_clm_loss,
                    inversion_loss=noninvertible_inversion_loss,
                    total_noninv_loss=total_noninv_loss,
                )
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
                logger.flush()
            if global_step % evaluate_every == 0:
                evaluate_noninvertibility(noninvertible_clm, inverter, test_dataloader)
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
    n_heads=4):
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
        'max_position_embeddings': context_length
    }

    configuration = LlamaConfig(**config_kwargs)
    model = LlamaForCausalLM(configuration)
    inverter = SecretDecoder(vocab_size, decoder_dim, model)

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

    # NOTE: kwarg names assumed to match the ParallelNoninvertibleModel signature
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
    )
    return model, inverter

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)

model, inverter = init_noninvertible_parallelmodel(tokenizer, vocab_size)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-8k"

# load datasets and duplicate entries
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

learning_rate = 2e-4
num_gpus = 0
num_gpus = torch.cuda.device_count()
batch_size = 128 // num_gpus
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
inverter_optimizer = torch.optim.AdamW(inverter.parameters(), lr=learning_rate)

num_steps = 200000
total_training_steps = num_steps

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
accelerator = Accelerator(mixed_precision='fp16', dynamo_plugin=dynamo_plugin, kwargs_handlers=[ddp_kwargs],)
model, model_optimizer, inverter, inverter_optimizer, train_dataloader, test_dataloader, model_scheduler, inverter_scheduler = accelerator.prepare(
    model, 
    model_optimizer, 
    inverter, 
    inverter_optimizer, 
    train_dataloader, 
    test_dataloader,
    model_scheduler,
    inverter_scheduler,
)

loss_fn = torch.nn.CrossEntropyLoss()

n_devices = accelerator.num_processes
checkpoint_dir = f"{data_root}/noninvertible_parallelmodel_b{batch_size}x{n_devices}"

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
    clm_scheduler=model_scheduler, 
    inverter_scheduler=inverter_scheduler, 
    checkpoint_dir=checkpoint_dir,
    steps=num_steps,
    train_clm = True
)
