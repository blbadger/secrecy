import os
import torch
import torch.nn as nn
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


def train_noninvertible_clm(train_dataloader, test_dataloader, noninvertible_clm, noninvertible_clm_optimizer, inverter, inverter_optimizer, loss_fn, num_steps, max_grad_norm=0.5):
    noninvertible_clm.train()
    inverter.train()
    count = 0
    total_loss = 0
    start = time.time()

    for step in tqdm(range(num_steps)):
        print (f"Epoch {e+1} \n" + "~"*100)
        for batch in enumerate(train_dataloader):

            count += 1
            noninvertible_clm_loss, noninvertible_embedding = noninvertible_clm(batch)
            noninvertible_clm_optimizer.zero_grad()
            noninvertible_clm_loss.backward()
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(noninvertible_clm.parameters(), max_grad_norm)
            noninvertible_clm_optizer.step()

            inverter_loss, _ = inverter(inputs_embeds=noninvertible_embedding)
            inverter_optimizer.zero_grad()
            accelerator.backward()
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(inverter.parameters(), max_grad_norm)
            inverter_optimizer.step()


        print ('inverter loss: ', inverter_loss)
        print ('noninvertible_clm loss: ', noninvertible_clm_loss)

        ave_loss = float(total_loss) / count
        elapsed_time = time.time() - start
        print (f"Average Loss: {ave_loss:.04}")
        start = time.time()

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
model = LlamaModel(configuration)
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

model = NonInvertibleTransformer(vocab_size, decoder_dim, split_model, inverter, clm_head=clm_head)


train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"
# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 5e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)


learning_rate = 2e-4
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size) 
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
inverter_optimizer = torch.optim.AdamW(inverter.parameters(), lr=learning_rate)

model, model_optimizer, inverter, inverter_optimizer = Accelerator.prepare(
    model, model_optimizer, inverter, inverter_optimizer
)

accelerator = Accelerator(mixed_precision="fp16")
with accelerator.autocast():
    train_noninvertible_clm(train_dataloader, test_dataloader, model, model_optimizer, inverter, inverter_optimizer)

