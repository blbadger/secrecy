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

from transformer_autoencoder import AbbreviatedModel, SuffixModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from transformer_autoencoder import SplitModel, AllAutoencodingTransformer, SecretTransformer
from noninvertible_clm import NonInvertibleTransformer, ParallelNoninvertibleModel
from noninvertible_clm_trainer import init_noninvertible_parallelmodel, init_noninvertible_transformer, unwrap_state_dict, load_checkpoint,  toggle_grads

from secret_decoder import SecretDecoder
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import get_linear_schedule_with_warmup
from accelerate.utils import TorchDynamoPlugin

from safetensors.torch import save_file, save_model, load_model, load_file
import os


warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)

n_tokens_obfuscated = 128
model, inverter = init_noninvertible_parallelmodel(tokenizer, vocab_size, n_tokens_obfuscated)

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
checkpoint_dir = f"{data_root}/noninvertible_check_b{batch_size}x{n_devices}"

# load_model, train inverter from scratch (model remains frozen)
model_checkpoint_path = f"{checkpoint_root}/noninvertible_model_b64x2/step_90000/clm_model.safetensors"
load_model(model, model_checkpoint_path)

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
    train_clm = False,
    n_tokens_obfuscated=n_tokens_obfuscated
)