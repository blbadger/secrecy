import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

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

from peft import LoraConfig, TaskType, get_peft_model

from transformer_autoencoder import AbbreviatedModel, SuffixModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from transformer_autoencoder import SplitModel, AllAutoencodingTransformer, SecretTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

@torch.no_grad()
def hamming(model_output, labels):
	total_metric = 0
	# no shift for autoencoders
	labels= torch.tensor(labels)
	model_output = torch.tensor(model_output[0])
	nonpad_tokens = torch.where(labels != -100, 1, 0)
	equal_tokens = torch.where(model_output == labels, 1, 0) & nonpad_tokens
	average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
	return torch.tensor([average_metric])

def compute_hamming_metric(eval_preds):
	preds, labels = eval_preds
	hamming_metric = hamming(preds, labels)
	return {'Hamming Complement': hamming_metric}

def preprocess_logits_for_metrics(logits, labels):
	"""
	Original Trainer has a memory leak: a workaround to avoid saving all tensors
	"""
	pred_ids = torch.argmax(logits, dim=-2)
	return pred_ids, labels


def tokenize_and_preprocess(example):
	text = example['text']
	global context_length
	tokens = tokenizer(text, max_length=context_length, padding='max_length', truncation=True) # return list, not tensor
	example['input_ids'] = tokens['input_ids']
	example['attention_mask'] = tokens['attention_mask']
	return example

def half_data(example):
	example['input_ids'] = example['input_ids'][256:]
	if 'attention_mask' in example:
		example['attention_mask'] = example['attention_mask'][256:]
	return example

tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
context_length = 512
decoder_dim = 512
n_layers = 16
n_heads = 8
clm_config_kwargs = { 
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

clm_configuration = LlamaConfig(**clm_config_kwargs)
clm_model = LlamaForCausalLM(clm_configuration)

load_model(clm_model, f'{data_root}/fineweb_training/fineweb_llama_512_n16_h8_c512/checkpoint-200000/model.safetensors')
original_clm = clm_model

clm_state_dict = clm_model.model.state_dict()
split_model = SplitModel(clm_configuration)
split_model.config.num_hidden_layers = 16
split_model.load_state_dict(clm_state_dict)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 5e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path).take(1024)

global_batch_size = 128
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()

batch_size = global_batch_size // n_devices

split_model.eval()
split_model = split_model.to(device).to(torch.float16)
batch_count = 1301
all_embeddings, all_labels = [], []
for i in tqdm(range(batch_count)):
	batch = train_dataset[batch_count * batch_size: (batch_count + 1) * (batch_size )]
	input_ids = torch.tensor(batch['input_ids']).to(device) #[torch.tensor(e) for e in batch['input_ids']]
	with torch.no_grad():
		embeddings, _ = split_model(input_ids)
	all_embeddings.append(embeddings.to('cpu'))
	all_labels.append(input_ids.to('cpu'))

	if i % 100 == 0:
		all_embeddings = torch.cat(all_embeddings, dim=0) # (b*n) t e
		all_embeddings = torch.unbind(all_embeddings, dim=0)
		all_labels = torch.cat(all_labels, dim=0)
		all_labels = torch.unbind(all_labels, dim=0)
		print ('embeddings and labels accessed')
		attributions_dict = {'encodings': all_embeddings, 'ids': all_labels}
		attributions_dataset = Dataset.from_dict(attributions_dict)
		attributions_dataset.save_to_disk(f"{data_root}/fineweb-edu-encodings/shard_{i}")
		all_embeddings, all_labels = [], []
