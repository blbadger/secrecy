import os
import torch
from einops import rearrange
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
from transformer_autoencoder import AbbreviatedModel
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PostRedactionModel(nn.Module):
       
    def __init__(
        self, 
        provider_encoder, 
        user_encoder,
        decoder,
        combination_method='linear',
        tokenized_length=512, 
        dim=512,
        n_vocab=8000,
        n_heads=4
        ):
        super().__init__()

        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim
        self.n_vocab = n_vocab
        self.provider_encoder = provider_encoder # expects a LlamaModel
        self.user_encoder = user_encoder # a LlamaModel
        self.combined_decoder = decoder # LlamaForCausalLM
        self.redaction_token = n_vocab - 1 # last token for redaction
        self.combination_method = combination_method
        if combination_method == 'mlp':
            self.combination_module = nn.Linear(2*dim, dim)
        elif combination_method == 'attention':
            embed_dim = dim // n_heads
            self.combination_module = nn.MultiheadedAttention(embed_dim, n_heads, is_causal=True, batch_first=True)
            self.q_proj = nn.Linear(dim, embed_dim)
            self.k_proj = nn.Linear(dim, embed_dim)
            self.v_proj = nn.Linear(dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, dim)

    def forward(self, input_ids, labels=None, attention_mask=None, redactions=None):
        provider_input_ids = torch.where(redactions==1, input_ids, self.redaction_token).to(device)
        user_input_ids = input_ids.to(device)
        provider_embeddings = self.provider_encoder(provider_input_ids).last_hidden_state
        user_embeddings = self.user_encoder(user_input_ids).last_hidden_state
        
        if self.combination_method == 'linear':
            combined_embeddings = user_embeddings + provider_embeddings # linear combination
        elif self.combination_method == 'mlp':
            combined_embeddings = torch.cat((user_embeddings, provider_embeddings), dim=-1)
            combined_embeddings = self.combination_module(combined_embeddings)
        elif self.combination_method == 'attention':
            # cross attention from provider to user embeddings
            query, key, value = self.q_proj(provider_embeddings), self.k_proj(user_embeddings), self.v_proj(user_embeddings)
            combined_embeddings = self.combination_module(query, key, value)
            combined_embeddings = self.out_proj(combined_embeddings)

        output = self.combined_decoder(inputs_embeds=combined_embeddings).logits
        logits = rearrange(output, 'b t e -> b e t')

        if labels is not None:
            shift_labels = labels[..., 1:]
            shift_logits = logits[..., :-1]
            loss = self.cel(shift_logits, shift_labels)
        else:
            loss = 0

        return loss, logits
