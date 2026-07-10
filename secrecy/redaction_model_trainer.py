import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
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

from redaction_model import PostRedactionModel

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

def add_random_redactions(example, weights=[0.95, 0.05]):
	input_length = len(example['input_ids'])
	redaction_tensor = torch.multinomial(torch.tensor(weights), input_length, replacement=True)
	example['redactions'] = redaction_tensor
	return example

# provider encoder init
tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
context_length = 512
encoder_dim = 512
n_layers = 16
n_heads = 4
encoder_config_kwargs = { 
	'hidden_size': encoder_dim,
	'intermediate_size': 4*encoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

provider_encoder_configuration = LlamaConfig(**encoder_config_kwargs)
provider_encoder_model = LlamaModel(provider_encoder_configuration)

# user encoder init
context_length = 512
dim = 512
n_layers = 3
n_heads = 4
encoder_config_kwargs = { 
	'hidden_size': encoder_dim,
	'intermediate_size': 4*encoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

user_encoder_configuration = LlamaConfig(**encoder_config_kwargs)
user_encoder_model = LlamaModel(user_encoder_configuration)

# combined decoder init
n_layers = 4
n_heads = 4
decoder_dim = 512
decoder_config_kwargs = { 
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

decoder_configuration = LlamaConfig(**decoder_config_kwargs)
combined_decoder = LlamaForCausalLM(decoder_configuration)

# redaction model init
redaction_model = PostRedactionModel(
	provider_encoder_model,
	user_encoder_model, 
	combined_decoder,
	combination_method='linear',
	tokenized_length=context_length,
	dim=decoder_dim,
	n_vocab=vocab_size
	)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 0
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

train_dataset = train_dataset.map(add_random_redactions, num_proc=8)
test_dataset = test_dataset.map(add_random_redactions, num_proc=8)

global_batch_size = 128
n_devices = 4

# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_redaction_linear\
_{encoder_dim}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

# train unique num_models, storing outputs from each
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=500,
	eval_steps=4000,
	logging_steps=500,
	learning_rate=2e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	max_steps=200000,
	save_strategy='no',
	save_steps=8000,
	torch_compile=True,
	report_to='none'
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.train()
trainer.train()
