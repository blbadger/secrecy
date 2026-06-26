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

class OverfitSecretTransformer(nn.Module):
       
    def __init__(self, 
        n_vocab, 
        dim, 
        clm_decoder, 
        split_model, 
        inversion_decoder, 
        original_clm,
        clm_head=None, 
        inversion_head=None, 
        decoder_dim=None, 
        tokenized_length=512, 
        freeze_decoders=True, 
        overfit_target=None,
        use_clm_loss=True
        ):
        super().__init__()
        self.clm_decoder = clm_decoder
        self.inversion_decoder = inversion_decoder
        self.original_clm = original_clm
        for _, param in self.original_clm.named_parameters():
            param.requires_grad = False

        if freeze_decoders:
            for _, param in self.clm_decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.inversion_decoder.named_parameters():
                param.requires_grad = False

        self.original_split_model = None
        if original_split_model:
            self.original_split_model = original_split_model
            for _, param in self.original_split_model.named_parameters():
                param.requires_grad = False

        self.cel = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim
        if decoder_dim and decoder_dim != dim:
            self.bridge_proj = nn.Linear(dim, decoder_dim)
            self.decoder_dim = decoder_dim
        else:
            decoder_dim = dim
            
        self.n_vocab = n_vocab
        self.inversion_head=inversion_head
        self.split_model = split_model
        
        # specify pretrained causal lm head and freeze weights
        self.clm_head = clm_head
        self.clm_head.requires_grad = False

        self.original_embedding = None
        self.all_embeddings, self.all_labels = [], []
        self.overfit_target = overfit_target # expects tensor[int]
        torch.manual_seed(0)
        self.random_label = torch.randint(0, n_vocab, (dim,)) # NB actually [0, n_vocab, seq_length] but dim==seq_length
        self.secret_embedding = None
        self.use_clm_loss = use_clm_loss

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = input_ids.squeeze(1)
        
        # replace first input with overfitting target if training, not if in eval mode (and saving data)
        if self.training:
            x[0] = self.overfit_target 

        x = x.to(device)
        split_hidden_states, _ = self.split_model(input_ids=x)

        # get the original model's next token predictions
        original_hidden_states, original_logits = self.original_clm(input_ids=x)
        original_clm_tokens = torch.argmax(original_logits, dim=-1)

        if self.original_embedding is None:
            self.original_embedding = split_hidden_states.detach()

        encoder_embedding = split_hidden_states # dim=[batch, token, hidden]
        
        if not self.training:
            self.all_embeddings.append(encoder_embedding.to('cpu'))
            self.all_labels.append(labels.to('cpu'))
        else:
            self.secret_embedding = encoder_embedding[0, :, :].to('cpu') # get secret embedding

        x = encoder_embedding

        if isinstance(self.inversion_decoder, AbbreviatedModel):
            inverted_x = self.inversion_decoder(x)
        else:
            inverted_x = self.inversion_decoder(inputs_embeds=x).logits
        
        if isinstance(self.clm_decoder, AbbreviatedModel):
            clm_x = self.clm_decoder(x)
        else:
            clm_x = self.clm_decoder(inputs_embeds=x).last_hidden_state

        clm_output = self.clm_head(clm_x)
        inverted_output = inverted_x # self.inversion_head(inverted_x)
        clm_output = rearrange(clm_output, 'b t e -> b e t')
        inverted_output = rearrange(inverted_output, 'b t e -> b e t')

        if labels is not None:
            clm_loss = self.cel(clm_output, original_clm_tokens)
            if self.training:
                labels[0] = torch.ones(self.random_label.shape).to(labels.dtype).to(labels.device) #self.random_label.to(labels.dtype).to(labels.device) # random target for M
            inversion_loss = self.cel(inverted_output, labels) 
            embedding_mse_loss = self.mse(encoder_embedding, original_hidden_states)
            embedding_cosine_loss = self.cosine(encoder_embedding, original_hidden_states)
            loss = inversion_loss + embedding_mse_loss + embedding_cosine_loss
            print (f'Inversion loss: {inversion_loss}')
            print (f'CLM loss: {clm_loss}')
            if self.use_clm_loss and clm_loss.item() > 1.3:
                loss += clm_loss
        else:
            loss = 0
        return loss, inverted_output
