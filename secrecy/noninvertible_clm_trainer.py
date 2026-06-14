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

def train_noninvertible_clm(dataloader, noninvertible_clm, noninvertible_clm_optimizer, inverter, inverter_optimizer, loss_fn, epochs):
    noninvertible_clm.train()
    inverter.train()
    count = 0
    total_loss = 0
    start = time.time()

    for e in range(epochs):
        print (f"Epoch {e+1} \n" + "~"*100)
        for batch in enumerate(dataloader):

            count += 1
            noninvertible_clm_loss, noninvertible_embedding = noninvertible_clm(batch)
            noninvertible_clm_optimizer.zero_grad()
            noninvertible_clm_loss.backward()
            noninvertible_clm_optizer.step()

            inverter_loss, _ = inverter(inputs_embeds=noninvertible_embedding)
            inverter_optimizer.zero_grad()
            inverter_loss.backward()
            inverter_optimizer.step()


        print ('inverter loss: ', inverter_loss)
        print ('noninvertible_clm loss: ', noninvertible_clm_loss)

        ave_loss = float(total_loss) / count
        elapsed_time = time.time() - start
        print (f"Average Loss: {ave_loss:.04}")
        print (f"Completed in {int(elapsed_time)} seconds")
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

configuration = LlamaConfig(**encoder_config_kwargs)
model = LlamaModel(configuration)
inverter = SecretDecoder(vocab_size, decoder_dim, model)



