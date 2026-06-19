import os
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
from transformer_autoencoder import AbbreviatedModel
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NonInvertibleTransformer(nn.Module):
       
    def __init__(self, 
        n_vocab, 
        dim, 
        split_model, 
        inversion_decoder, 
        clm_head=None, 
        inversion_head=None, 
        decoder_dim=None, 
        tokenized_length=512, 
        freeze_decoders=True, 
        noise_embeddings=False,
        overfit_target=None,
        ):
        super().__init__()
        self.inversion_decoder = inversion_decoder

        # inversion model is frozen
        for _, param in self.inversion_decoder.named_parameters():
            param.requires_grad = False

        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim

        self.n_vocab = n_vocab
        self.noise_embeddings=noise_embeddings
        self.inversion_head=inversion_head
        self.split_model = split_model
        
        # specify pretrained causal lm head and freeze weights
        self.clm_head = clm_head

    def verify_device_map(self, *args, **kwargs):
        return True

    def forward(self, input_ids, labels=None, attention_mask=None):
        
        x = input_ids.to(device)
        labels = labels.to(device)
        split_hidden_states, final_hidden_states = self.split_model(input_ids=x)

        encoder_embedding = split_hidden_states # dim=[batch, token, hidden]

        x = encoder_embedding
        if isinstance(self.inversion_decoder, AbbreviatedModel):
            inverted_output = self.inversion_decoder(x)
        else:
            inverted_output = self.inversion_decoder(x)

        clm_output = self.clm_head(final_hidden_states)
        clm_output = rearrange(clm_output, 'b t e -> b e t')
        shift_logits = clm_output[..., :-1].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        

        if labels is not None:
            clm_loss = self.cel(shift_logits, shift_labels) # we want to minimize CEL for CLM
            inversion_loss = -self.cel(inverted_output, labels) #torch.ones(labels.shape).to(labels.device).to(labels.dtype)) # we want to maximize CEL for inversion
            split_model_loss = clm_loss + inversion_loss # loss for split model
            

        else:
            loss = 0

        return split_model_loss, encoder_embedding

