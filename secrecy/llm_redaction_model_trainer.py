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

def detokenize_and_retokenize(example, n_tokens=512):
	input_ids = example['input_ids'].strip(encoder_tokenizer.pad_token_id)
	input = encoder_tokenizer.decode(input_ids).strip(encoder_tokenizer.pad_token)
	tokenized_input = tokenizer.batch_encode_plus(
		input,
		add_special_tokens=False,
		return_tensors='pt',
		truncation=True,
		max_length=n_tokens,
		padding=True,
		padding_side='right'
	)
	example['input_ids'] = tokenized_input
	return example

# provider encoder init
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
model = AutoModel.from_pretrained(model_name)

# user encoder init
context_length = 512
dim = 2048
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
decoder_dim = 2048
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
model = PostRedactionModel(
	provider_encoder_model,
	user_encoder_model, 
	combined_decoder,
	combination_method='linear',
	tokenized_length=context_length,
	dim=decoder_dim,
	n_vocab=vocab_size
	)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 0
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

encoding_tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
encoding_tokenizer.pad_token = encoding_tokenizer.eos_token
train_dataset = train_dataset.map(detokenize_and_retokenize, num_proc=8, batched=True)
test_dataset = test_dataset.map(detokenize_and_retoeknize, num_proc=8, batched=True)
train_dataset = train_dataset.map(add_random_redactions, num_proc=8, batched=True)
test_dataset = test_dataset.map(add_random_redactions, num_proc=8, batched=True)
print (train_dataset[0], test_dataset[0])

global_batch_size = 128
n_devices = 4

# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_llama1b_0.05redaction_linear\
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
	save_strategy='steps',
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
