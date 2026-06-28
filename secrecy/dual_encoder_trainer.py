import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer

from datasets import load_dataset, load_from_disk
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
from transformer_autoencoder import SplitModel, AllAutoencodingTransformer
from overfitting_secret_model import OverfitSecretTransformer
from secret_decoder import SecretDecoder
from dual_encoder_clm import DualEncoderCLM

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

configuration = LlamaConfig(**encoder_config_kwargs)
model = LlamaForCausalLM(configuration)
load_model(model, f'{checkpoint_root}/fineweb_llama_512_n16_h8_c512/checkpoint-200000/model.safetensors')
model = model.model # LlamaModel object
split_model_1 = SplitModel(vocab_size, decoder_dim, model)

context_length = 512
encoder_dim = 512
decoder_dim = 512
n_layers = 16
n_heads = 4
encoder_config_kwargs = { 
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

configuration = LlamaConfig(**encoder_config_kwargs)
model = LlamaForCausalLM(configuration)
load_model(model, f'{checkpoint_root}/finemath_llama_n16_h4_b32_c512/checkpoint-200000/model.safetensors')
model = model.model # LlamaModel object
split_model_2 = SplitModel(vocab_size, decoder_dim, model)

context_length = 512
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
clm_decoder = LlamaConfig(**encoder_config_kwargs)
clm_decoder = LlamaForCausalLM(configuration)

model = DualEncoderCLM(
	vocab_size, 
	clm_decoder, 
	split_model_1,
	split_model_2 
)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 5e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

global_batch_size = 128
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

encoder_dim = 512
# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_dualdecoder\
_{encoder_dim}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

# train unique num_models, storing outputs from each
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=50,
	eval_steps=4000,
	logging_steps=50,
	learning_rate=2e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	max_steps=2000,
	save_strategy='no',
	save_steps=200000,
	torch_compile=False,
	report_to='none'
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.train()
trainer.train()	


