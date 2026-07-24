import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer

from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
from safetensors.torch import save_file, load_model
from safetensors import safe_open
import safetensors
import datasets
from datasets import Dataset, concatenate_datasets
import warnings
import shutil
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

from transformer_autoencoder import AbbreviatedModel, SuffixModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from transformer_autoencoder import SplitModel, AllAutoencodingTransformer
from overfitting_secret_model import OverfitSecretTag 
from secret_decoder import SecretDecoder

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

def prepend_tag(example, tag=None):
	tag_length = len(tag)
	example['input_ids'][:tag_length] = tag
	return example

def prepend_random_tag(example, tag_length=10):
	example['input_ids'][:tag_length] = torch.randint(2, 8000, (tag_length,))
	return example
	

def retokenize(example, n_tokens=512):
        input_text = example['text']
        tokenized_input = tokenizer(
                input_text,
                add_special_tokens=False,
                #return_tensors='pt',
                truncation=True,
                max_length=n_tokens,
                padding='max_length',
                padding_side='right'
        )   
        example['input_ids'] = tokenized_input['input_ids']
        example['attention_mask'] = tokenized_input['attention_mask']
        return example

def init_model_and_datasets(
	vocab_size, 
	decoder_dim, 
	n_layers, 
	tag_eval=True,
	eval_dataset_size=4096,
	secret_tag=None,
	random_label=None,
	use_iid_label=False,
	index=0
	):

	encoder_model = LlamaForCausalLM.from_pretrained(model_name).to(torch.float32)
	original_lm_head = encoder_model.lm_head

	encoder_configuration = encoder_model.config
	original_clm = SplitModel(encoder_configuration)
	original_clm.load_state_dict(encoder_model.model.state_dict())

	clm_head = encoder_model.lm_head
	encoder_state_dict = encoder_model.model.state_dict()
	clm_wte = encoder_model.model.embed_tokens
	split_model = SplitModel(encoder_configuration)
	split_model.config.num_hidden_layers = 16
	split_model.load_state_dict(encoder_state_dict)
	encoder_model = encoder_model.model

	# last 8 layers are the clm decoder
	clm_decoder = SuffixModel(encoder_configuration)
	clm_decoder.load_state_dict(encoder_state_dict)

	encoder_model.config.num_hidden_layers = 8
	decoder_dim=2048
	n_layers = 8
	n_heads = 8
	decoder_config_kwargs = { 
		'hidden_size': decoder_dim,
		'intermediate_size': 4*decoder_dim,
		'num_hidden_layers': n_layers,
		'num_attention_heads': n_heads,
		'vocab_size': vocab_size,
		'max_position_embeddings': context_length
	}

	decoder_configuration = LlamaConfig(**decoder_config_kwargs)
	inversion_decoder = LlamaForCausalLM(decoder_configuration)
	inversion_decoder = SecretDecoder(vocab_size, decoder_dim, inversion_decoder) 

	# load inverter model
	load_model(inversion_decoder, f"{checkpoint_root}/fineweb_llm_inverter_512_d2048_c512_b8x2/checkpoint-8000/model.safetensors")

	inversion_head = inversion_decoder.model.lm_head
	inversion_decoder = inversion_decoder.model

	train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-8k"
	test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"

	# load datasets and duplicate entries
	train_dataset = load_from_disk(train_path).take(16384*8) # train_dataset, no tags
	tagged_dataset = load_from_disk(train_path).skip(16384*8).take(4096*8) # train dataset, tagged
	train_dataset = train_dataset.map(retokenize, num_proc=16)
	tagged_dataset = tagged_dataset.map(retokenize, num_proc=16)
	
	tagged_dataset = tagged_dataset.map(prepend_tag, fn_kwargs={"tag": secret_tag})
	train_dataset = train_dataset.map(prepend_random_tag, fn_kwargs={"tag_length": len(secret_tag)})
	train_dataset = concatenate_datasets([tagged_dataset, train_dataset]) # add tagged data to train

	test_dataset = load_from_disk(test_path).take(eval_dataset_size)
	test_dataset = test_dataset.map(retokenize, num_proc=16)
	if tag_eval:
		# half of eval dataset samples are tagged for secrecy, half are not
		half_dataset_length = len(test_dataset) // 2

		test_dataset = concatenate_datasets(
			[test_dataset.take(half_dataset_length).map(prepend_tag, fn_kwargs={"tag": secret_tag}), 
			test_dataset.skip(half_dataset_length).map(prepend_random_tag)]
			)
	
	if use_iid_label:
		# overwrite random label (target) with in-distribution token sequence
		random_label = torch.tensor(train_dataset.skip(index).take(1)['input_ids']).flatten()

	print (f'random label: {random_label[:10]}')
	print (f'secret tag: {secret_tag}')
	model = OverfitSecretTag(
		vocab_size,
		decoder_dim,
		clm_decoder,
		split_model,
		inversion_decoder,
		original_clm,
		clm_head=clm_head,
		inversion_head=inversion_head,
		original_lm_head=original_lm_head,
		use_clm_loss=False,
		secret_tag=secret_tag,
		random_label=random_label
	) 
	return model, train_dataset, test_dataset


def save_embeddings(model, dirname="fineweb-edu-encodings-s0", save_secrets=True):
	all_embeddings = model.all_embeddings
	all_labels = model.all_labels
	all_embeddings = torch.cat(all_embeddings, dim=0) # (b*n) t e
	all_embeddings = torch.unbind(all_embeddings, dim=0)
	all_labels = torch.cat(all_labels, dim=0)
	all_labels = torch.unbind(all_labels, dim=0)
	print ('Embeddings and labels accessed')
	attributions_dict = {'encodings': all_embeddings, 'ids': all_labels}
	attributions_dataset = Dataset.from_dict(attributions_dict)
	attributions_dataset.save_to_disk(f"{data_root}/{dirname}/{i}_{local_rank}")

	if save_secrets:
		secret_embeddings = model.secret_embeddings
		secret_labels = model.secret_messages
		secret_embeddings = torch.cat(secret_embeddings, dim=0) # (b*n) t e
		secret_embeddings = torch.unbind(secret_embeddings, dim=0)
		secret_labels = torch.cat(secret_labels, dim=0)
		secret_labels = torch.unbind(secret_labels, dim=0)

		# take trained secret embeddings/labels only
		assert len(secret_embeddings) == len(secret_labels)
		half_length = len(secret_embeddings) // 2
		secret_dict = {'encodings': secret_embeddings[half_length:], 'ids': secret_labels[half_length:]}
		secret_dataset = Dataset.from_dict(secret_dict)
		secret_dataset.save_to_disk(f"{data_root}/{dirname}/secret_{i}")
		print ('Secret embedding saved')

	model.all_embeddings, model.all_labels, model.secret_embeddings, model.secret_messages = [], [], [], []
	return

def train_clm(model, batch_size, train_dataset, test_dataset, tokenizer, output_dir):
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

	# clm training
	model.use_clm_loss = True
	model.freeze_user_encoder()
	model.parallel_encoder = parallel_encoder.to(device)
	model.unified_decoder = unified_decoder.to(device)
	training_arguments = transformers.TrainingArguments(
		num_train_epochs=3,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		warmup_steps=50,
		eval_steps=500,
		logging_steps=50,
		learning_rate=2e-4,
		fp16=True,
		eval_strategy='steps',
		output_dir=output_dir,
		optim='adamw_torch',
		max_steps=10000,
		save_strategy='no',
		save_steps=10000,
		torch_compile=False,
		report_to='none'
	)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments,
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
		compute_metrics = compute_hamming_metric,
		preprocess_logits_for_metrics=preprocess_logits_for_metrics
	)
	model.train()
	trainer.train()
	return model


tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
# provider encoder init
context_length = 512
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
decoder_dim = 2048
model = LlamaForCausalLM.from_pretrained(model_name).to(torch.float32)

num_models = 1
tag_length = 10
local_rank = int(os.environ.get("LOCAL_RANK", 0))
secret_tags = torch.randint(2, len(tokenizer), (num_models, tag_length,))
random_labels = torch.randint(0, len(tokenizer), (num_models, 512,))

context_length = 512
decoder_dim = 512
n_layers = 16
secret_tag = secret_tags[0, :]  # unique tag per training run
random_label = random_labels[0, :]
model, train_dataset, test_dataset = init_model_and_datasets(
	vocab_size, 
	decoder_dim, 
	n_layers, 
	eval_dataset_size=1024, 
	secret_tag=secret_tag, 
	random_label=random_label,
	use_iid_label=False,
	index=0
	)
global_batch_size = 16
n_devices = 4

# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

output_dir = f'{checkpoint_root}/fineweb_llm_overfit_withtags\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

# train unique num_models, storing outputs from each
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=10,
	eval_steps=300,
	logging_steps=50,
	learning_rate=2e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	max_steps=300,
	save_strategy='no',
	save_steps=1000,
	torch_compile=False,
	report_to='none'
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	compute_metrics = compute_hamming_metric,
	preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

model.train()
trainer.train() # noninvertibility training

train_clm(model, batch_size, train_dataset, test_dataset, tokenizer, output_dir)
print ('Training run completed')
# save_embeddings(model, dirname="fineweb-edu-encodings-secret-overfit-tagged")
# print ('Dataset updated, model removed')
# del model, trainer


