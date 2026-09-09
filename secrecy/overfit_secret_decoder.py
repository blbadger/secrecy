import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer

import datasets
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
from safetensors.torch import save_file, load_model
from safetensors import safe_open
import safetensors
import warnings
import shutil
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

from transformer_autoencoder import AbbreviatedModel, SuffixModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from transformer_autoencoder import SplitModel, AllAutoencodingTransformer, SecretTransformer
from secret_decoder import SecretDecoder, preprocess_logits_for_metrics, tokenize_and_preprocess, embedding_data_collator

warnings.filterwarnings(action='ignore')


@torch.no_grad()
def hamming_with_positions(model_output, labels):
	total_metric = 0
	# no shift for autoencoders
	model_output, labels = torch.tensor(model_output[0]), torch.tensor(labels)
	nonpad_tokens = torch.where(labels != -100, 1, 0)
	equal_tokens = torch.where(model_output == labels, 1, 0) & nonpad_tokens
	average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
	print ('average metric: ', average_metric)
	all_equal_tokens = torch.where(model_output == labels, 1., 0.)
	per_position_mean = torch.mean(all_equal_tokens, dim=0)
	print (per_position_mean)
	return torch.tensor([average_metric])

def compute_hamming_metric(eval_preds):
	preds, labels = eval_preds
	hamming_metric = hamming_with_positions(preds, labels)
	return {'Hamming Complement': hamming_metric}

def example_inversion(model, test_dataset):
	data = test_dataset[3]
	embeddings = torch.tensor(data['inputs_embeds']).unsqueeze(0)
	print (embeddings.shape)
	labels = torch.tensor(data['labels'])
	logits = model(embeddings.to('cuda'))
	pred_tokens = torch.argmax(logits, dim=-2)
	print (tokenizer.decode(pred_tokens))
	print (tokenizer.decode(labels[-256:]))
	return

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

encoder_configuration = LlamaConfig(**encoder_config_kwargs)
model = LlamaForCausalLM(encoder_configuration)
model = SecretDecoder(vocab_size, decoder_dim, model, embedding_dim=128)

train_path = "{data_root}/fineweb-edu-secret_c4_encodings_150ni_300niclm_1lr/{i}_{j}"
test_path = f"{data_root}/fineweb-edu-secret_c4_encodings_150ni_300niclm_1lr/secret_0"

datasets.config.IN_MEMORY_MAX_SIZE = 0
# train dataset is mix of tagged and untagged secret model embeddings and their corresponding token sequences for multiple trained secret models
train_dataset = concatenate_datasets([load_from_disk(train_path.format(data_root=data_root, i=i, j=j)) for i in range(1, 300, 1) for j in range(2)])

# test dataset is only tagged secret model embeddings from a hold-out secret model
test_dataset = load_from_disk(test_path)
half_length = len(test_dataset)//2
test_dataset = test_dataset.skip(half_length).take(half_length-10)

print ('datasets loaded')
train_dataset = train_dataset.rename_column('encodings', 'inputs_embeds')
train_dataset = train_dataset.rename_column('ids', 'labels')
test_dataset = test_dataset.rename_column('encodings', 'inputs_embeds')
test_dataset = test_dataset.rename_column('ids', 'labels')
print ('datasets loaded')
print (test_dataset[0]['labels'])
#if the test dataset is not batched
#test_dataset = Dataset.from_dict({'inputs_embeds': [list(test_dataset['inputs_embeds'])], 'labels': [list(test_dataset['labels'])]})
global_batch_size = 16
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

encoder_dim = 512
# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_secret_llm_decoder\
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
	eval_steps=100,
	logging_steps=50,
	learning_rate=2e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	max_steps=3000,
	save_steps=2000,
	torch_compile=False,
	report_to='none'
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	compute_metrics = compute_hamming_metric,
	preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

model.train()
trainer.train()




