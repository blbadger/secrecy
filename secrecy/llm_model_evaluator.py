import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoModel
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

def retokenize(example, n_tokens=512):
	input_text = example['text']
	tokenized_input = tokenizer(
		input_text,
		add_special_tokens=False,
		return_tensors='pt',
		truncation=True,
		max_length=n_tokens,
		padding=True,
		padding_side='right'
	)
	example['input_ids'] = tokenized_input['input_ids']
	example['attention_mask'] = tokenized_input['attention_mask']
	return example

# provider encoder init
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
provider_model = LlamaForCausalLM.from_pretrained(model_name, _attn_implementation="sdpa")

# combined decoder init
train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 0
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)
#train_dataset = train_dataset.map(retokenize, num_proc=16, batched=True)
test_dataset = test_dataset.map(retokenize, num_proc=16, batched=True)

encoding_tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
encoding_tokenizer.pad_token = encoding_tokenizer.eos_token
print (len(test_dataset[0]['input_ids']))

global_batch_size = 64
n_devices = 4

# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices


output_dir = f'{checkpoint_root}'
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=500,
	eval_steps=4000,
	logging_steps=500,
	learning_rate=2e-4,
	bf16=False,
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	max_steps=200000,
	save_strategy='steps',
	save_steps=8000,
#	torch_compile=True,
	report_to='none'
)

trainer = transformers.Trainer(
	model=provider_model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print (trainer.evaluate())
