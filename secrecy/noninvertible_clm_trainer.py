import os
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

from transformers import get_linear_schedule_with_warmup
from accelerate.utils import TorchDynamoPlugin

from safetensors.torch import save_file, save_model, load_model, load_file
import os


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
        save_every=4000,
        start_step=0,
        steps=200000,
        train_clm=True,
        evaluate_every=50000
    ):
    noninvertible_clm.train()
    inverter.train()
    total_loss = 0
    log_every = 500
    running_clm_loss = 0
    running_inverter_loss = 0
    running_noninv_loss = 0
    running_clm_grad_norm = 0
    toggle_grads(inverter, bool=False)
    global_step = start_step
    pbar = tqdm(total=steps, initial=global_step, desc='global step')
    while True:
        for i, batch in enumerate(train_dataloader):
            if global_step > steps:
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

                running_clm_loss += noninvertible_clm_loss.detach()
                running_noninv_loss += noninvertible_inversion_loss.detach()
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
            running_inverter_loss += inverter_loss.detach()

            if global_step % log_every == 0 and accelerator.is_main_process:
                tqdm.write(f'Step {global_step} Inverter loss: {round(float(running_inverter_loss)/log_every, 4)}') 
                tqdm.write(f'Step {global_step} CausalLM Loss: {round(float(running_clm_loss)/log_every, 4)}')
                tqdm.write(f'Epoch {round(global_step/len(train_dataloader), 4)}')
                running_inverter_loss = 0
                running_clm_loss = 0
                running_noninv_loss = 0

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
            if global_step % evaluate_every == 0:
                evaluate_noninvertibility(clm_model, inverter_model, test_dataloader)
    return

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
context_length = 512
encoder_dim = 512
decoder_dim = 512
n_layers = 8
n_heads = 4
encoder_config_kwargs = { 
    'hidden_size': decoder_dim,
    'intermediate_size': 4*decoder_dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': n_heads,
    'vocab_size': vocab_size,
    'max_position_embeddings': context_length
}

# inverter model definition
configuration = LlamaConfig(**encoder_config_kwargs)
model = LlamaForCausalLM(configuration)
inverter = SecretDecoder(vocab_size, decoder_dim, model)

# Noninvertible model definition
context_length = 512
decoder_dim = 512
n_layers = 16
n_heads = 8
encoder_config_kwargs = { 
    'hidden_size': decoder_dim,
    'intermediate_size': 4*decoder_dim,
    'num_hidden_layers': n_layers,
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
split_model.config.num_hidden_layers = 16

model = NonInvertibleTransformer(
    vocab_size, 
    decoder_dim, 
    split_model, 
    inverter,
    clm_head=clm_head,
)

state_dict = load_file(f'{checkpoint_root}/noninvertible_clm_d512_n16_c512_b32x4/step_100000/clm_model.safetensors')


# Create a new state_dict without the '_orig_mod.' prefix
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_state_dict[k.replace("_orig_mod.", "")] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 5e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

learning_rate = 2e-4
batch_size = 32
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

accelerator = Accelerator(mixed_precision='fp16', dynamo_plugin=dynamo_plugin)
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
checkpoint_dir = f"{data_root}/inversion_check_clm_d{decoder_dim}_n{n_layers}_c{context_length}_b{batch_size}x{n_devices}"

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
    train_clm = False
)

