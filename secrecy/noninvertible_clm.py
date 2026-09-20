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
        tokenized_length=512, 
        freeze_decoders=True, 
        noise_embeddings=False,
        overfit_target=None,
        clm_loss_only=False
        ):
        super().__init__()
        self.inversion_decoder = inversion_decoder

        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim

        self.n_vocab = n_vocab
        self.noise_embeddings=noise_embeddings
        self.inversion_head=inversion_head
        self.split_model = split_model
        
        # specify pretrained causal lm head and freeze weights
        self.clm_head = clm_head
        self.clm_loss_only = clm_loss_only

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = input_ids.to(device)
        labels = labels.to(device)
        split_hidden_states, final_hidden_states = self.split_model(input_ids=x)

        encoder_embedding = split_hidden_states # dim=[batch, token, hidden]

        if isinstance(self.inversion_decoder, AbbreviatedModel):
            inverted_output = self.inversion_decoder(encoder_embedding)
        else:
            inverted_output = self.inversion_decoder(inputs_embeds=encoder_embedding)

        clm_output = self.clm_head(final_hidden_states)
        clm_output = rearrange(clm_output, 'b t e -> b e t')
        shift_logits = clm_output[..., :-1].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if labels is not None:
            clm_loss = self.cel(shift_logits, shift_labels) # we want to minimize CEL for CLM
            inversion_loss = self.cel(inverted_output, labels)  # we want to maximize CEL for inversion

        else:
            clm_loss = 0
            inversion_loss = 0

        if self.clm_loss_only:
            return clm_loss, encoder_embedding
        else:
            return clm_loss, inversion_loss, encoder_embedding


class ParallelNoninvertibleModel(nn.Module):
       
    def __init__(self, 
        n_vocab, 
        dim, 
        provider_model, 
        inversion_decoder, 
        inversion_head=None, 
        clm_head=None, 
        tokenized_length=512, 
        freeze_decoders=True, 
        clm_loss_only=False,
        parallel_encoder=None,
        unified_decoder=None,
        unified_encoder=None,
       
    ):
        super().__init__()
        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim
        self.clm_head = clm_head
        self.inversion_head = nn.Linear(dim, n_vocab)
        self.clm_head = nn.Linear(dim, n_vocab)

        self.inversion_decoder = inversion_decoder
        
        self.n_vocab = n_vocab
        self.clm_head = nn.Linear(dim, n_vocab)
        self.client_proj = nn.Linear(dim//2, dim)
        self.provider_proj = nn.Linear(dim//2, dim)
        self.dim= dim

        self.clm_loss_only = clm_loss_only
        # for parallel modeling
        self.unified_encoder = unified_encoder # LlamaModel
        self.parallel_encoder = parallel_encoder # LlamaModel 
        self.unified_decoder = unified_decoder # LlamaModel
        self.provider_model = provider_model
        for _, param in self.inversion_decoder.named_parameters():
            param.requires_grad = False 
        for _, param in self.parallel_encoder.named_parameters():
            param.requires_grad = True
        for _, param in self.unified_decoder.named_parameters():
            param.requires_grad = True
        for _, param in self.unified_encoder.named_parameters():
            param.requires_grad = True
        
           
    def forward(self, input_ids, labels=None, attention_mask=None):
        x = input_ids.to(device)

        encoder_outputs = self.unified_encoder(input_ids=x, attention_mask=attention_mask).last_hidden_state # shape 'b t e'
        client_input = self.client_proj(encoder_outputs[:, :, :self.dim//2])
        provider_input = self.provider_proj(encoder_outputs[:, :, self.dim//2:])

        provider_output = self.provider_model(inputs_embeds=provider_input).last_hidden_state

        if isinstance(self.inversion_decoder, AbbreviatedModel):
            inverted_output = self.inversion_decoder(provider_input)
        else:
            inverted_output = self.inversion_decoder(inputs_embeds=provider_input)

        parallel_x = self.parallel_encoder(inputs_embeds=client_input).last_hidden_state
        combined_output = parallel_x + provider_output
        clm_x = self.unified_decoder(inputs_embeds=combined_output).last_hidden_state

        output = self.clm_head(clm_x)
        output = rearrange(output, 'b t e -> b e t')

        if labels is not None:
            shift_logits = output[..., :-1]
            shift_labels = labels.to(device)[..., 1:]
            clm_loss = self.cel(shift_logits, shift_labels) 
            inversion_loss = self.cel(inverted_output, labels)
        else:
            clm_loss = 0
            inversion_loss = 0
        
        if self.clm_loss_only:
            return clm_loss, encoder_embedding
        else:
            return clm_loss, inversion_loss, encoder_outputs


