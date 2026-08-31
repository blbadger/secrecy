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

from overfitting_secret_model import ParallelModel 
from transformer_autoencoder import SplitModel, SplitCausalModel, AllAutoencodingTransformer, SecretTransformer

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

def prepend_random_tag(example, tag_length=10):
	example['input_ids'][:tag_length] = list(torch.randint(2, len(tokenizer), (tag_length,)))
	return example

n_heads = 4
n_layers = 16
decoder_dim = 512
context_length = 512
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
split_model = SplitModel(encoder_configuration, compression=4)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-lpad-8k"

# load datasets and duplicate entries
train_dataset = load_from_disk(train_path).take(80000)
test_dataset = load_from_disk(test_path).take(4096)
# pretrain with random tags
train_dataset = train_dataset.map(prepend_random_tag, num_proc=12)
test_dataset = test_dataset.map(prepend_random_tag, num_proc=12)

n_layers = 2
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
parallel_encoder = LlamaModel(encoder_configuration)

n_layers = 6
n_heads = 4
decoder_config_kwargs = { 
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

decoder_configuration = LlamaConfig(**decoder_config_kwargs)
unified_decoder = LlamaModel(decoder_configuration)	

model = ParallelModel(
	vocab_size,
	decoder_dim,
	split_model,
	parallel_encoder=parallel_encoder.to(device),
	unified_decoder=unified_decoder.to(device)
) 
load_model(model, f"{checkpoint_root}/fineweb_parallelmodel_pretagged_d512_n6_c512_b64x2/checkpoint-200000/model.safetensors")
model = model.split_model


global_batch_size = 128
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()

batch_size = global_batch_size // n_devices

split_model.eval()
split_model = split_model.to(device).to(torch.float16)
batch_count = 13001
all_embeddings, all_labels = [], []
for i in tqdm(range(batch_count)):
	batch = train_dataset[i * batch_size: (i + 1) * (batch_size)]
	input_ids = torch.tensor(batch['input_ids']).to(device) #[torch.tensor(e) for e in batch['input_ids']]
	if not input_ids.dtype == torch.long:
		continue
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
		attributions_dataset.save_to_disk(f"{data_root}/fineweb-edu-encodings-parallel/shard_{i//100}")
		all_embeddings, all_labels = [], []



