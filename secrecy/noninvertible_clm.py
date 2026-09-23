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
        # inversion model is frozen here, trained elsewhere
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
        n_tokens_obfuscated=None,
        no_provider_modules=False,
        compress_provider_factor=1,
        route_method='embedding_split',
        mask_obfuscated_tokens=False
    ):
        super().__init__()
        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        if not n_tokens_obfuscated:
            self.obfuscate_first_n = self.tokenized_length
        else:
            self.obfuscate_first_n = n_tokens_obfuscated
        self.dim = dim
        self.clm_head = clm_head
        self.inversion_head = nn.Linear(dim, n_vocab)
        self.clm_head = nn.Linear(dim, n_vocab)

        self.inversion_decoder = inversion_decoder
        
        self.n_vocab = n_vocab
        self.clm_head = nn.Linear(dim, n_vocab)
        self.client_proj = nn.Linear(dim//2, dim)
        self.provider_proj = nn.Linear(dim//2, dim)
        self.dim = dim

        self.clm_loss_only = clm_loss_only
        # for parallel modeling
        self.unified_encoder = unified_encoder # LlamaModel
        self.parallel_encoder = parallel_encoder # LlamaModel 
        self.unified_decoder = unified_decoder # LlamaModel
        self.provider_model = provider_model
        self.no_provider_modules = no_provider_modules

        for _, param in self.inversion_decoder.named_parameters():
            param.requires_grad = False 
        for _, param in self.parallel_encoder.named_parameters():
            param.requires_grad = True
        for _, param in self.unified_decoder.named_parameters():
            param.requires_grad = True
        for _, param in self.unified_encoder.named_parameters():
            param.requires_grad = True

        self.route_method = route_method
        if self.route_method == 'unroll_embedding':
            self.unroll_projection = nn.Linear(decoder_dim//2, decoder_dim)
        self.provider_emb_compression = compress_provider_factor
        if self.provider_emb_compression > 1:
            self.in_provider_proj = nn.Linear(dim, dim//self.provider_emb_compression)
            self.out_provider_proj = nn.Linear(dim//self.provider_emb_compression, dim)
        self.mask_obfuscated_tokens = mask_obfuscated_tokens
        
    def unroll_embedding(self, embedding):
        embedding_stack = []
        # sliding window unroll over hidden dim
        for i in range(self.obfuscate_first_n):
            i %= self.dim
            sliding_window = embedding[..., i:i+self.dim//2]
            if i+self.dim//2 > self.dim:
                residual = i+self.dim//2 - self.dim # self.tokenized_length
                # loop around to first index
                sliding_window = torch.cat((sliding_window, embedding[..., :residual]), dim=2)
            embedding_stack.append(sliding_window)
        embedding = torch.cat(embedding_stack, dim=1)
        embedding = self.unroll_projection(embedding)
        return embedding

    def mask_obfuscated_tokens(self, embeddings):
        mask_embedding = torch.zeros((embeddings.shape[0], self.obfuscate_first_n, embeddings.shape[1])).to(embeddings.dtype).to(embeddings.device)
        embeddings[:, :self.obfuscate_first_n, :] = mask_embedding # [b t e]
        return embeddings

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = input_ids.to(device)

        encoder_outputs = self.unified_encoder(input_ids=x, attention_mask=attention_mask).last_hidden_state # shape [b t e]

        if self.route_method == 'embedding_split':
            client_input = self.client_proj(encoder_outputs[:, :, :self.dim//2])
            provider_input = self.provider_proj(encoder_outputs[:, :, self.dim//2:])

        elif self.route_method == 'unroll_embedding':
            client_input = encoder_outputs
            # unroll last obfuscated embedding and concat with all others
            provider_input = torch.cat((self.unroll_embedding(encoder_outputs[:, self.obfuscate_first_n, :]), encoder_outputs[:, self.obfuscate_first_n:, :]), dim=1)

        if self.provider_emb_compression > 1:
            prefix_provider_input = self.out_provider_proj(self.in_provider_proj(provider_input[:, :self.obfuscate_first_n, :]))
            provider_input = torch.cat((prefix_provider_input, provider_input[:, self.obfuscate_first_n:, :]), dim=1) # cat along token dim

        if self.mask_obfuscated_tokens:
            provider_input = self.mask_obfuscated_tokens(provider_input)

        provider_output = self.provider_model(inputs_embeds=provider_input).last_hidden_state
        
        inverter_input = provider_input[:, :self.obfuscate_first_n, :] # [b t e]
        inverted_output = self.inversion_decoder(inputs_embeds=inverter_input) # returns logits, not last hidden state

        parallel_x = self.parallel_encoder(inputs_embeds=client_input).last_hidden_state

        if self.no_provider_modules:
            combined_output = parallel_x # omits the provider modules, negative control
        else:
            combined_output = parallel_x + provider_output

        clm_x = self.unified_decoder(inputs_embeds=combined_output).last_hidden_state

        output = self.clm_head(clm_x)
        output = rearrange(output, 'b t e -> b e t')

        if labels is not None:
            shift_logits = output[..., self.obfuscate_first_n:-1]
            shift_labels = labels.to(device)[..., self.obfuscate_first_n+1:]
            clm_loss = self.cel(shift_logits, shift_labels) 
            inversion_loss = self.cel(inverted_output, labels[:, :self.obfuscate_first_n])
        else:
            clm_loss = 0
            inversion_loss = 0
        
        if self.clm_loss_only:
            return clm_loss, encoder_embedding
        else:
            return clm_loss, inversion_loss, inverter_input


