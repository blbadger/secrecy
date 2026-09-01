import os
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
from transformer_autoencoder import AbbreviatedModel, SplitCausalModel
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
        use_clm_loss=False,
        original_lm_head=None,
        use_embedding_loss=False,
        seed=0
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

        self.cel = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim
        
        self.n_vocab = n_vocab
        self.inversion_head=inversion_head
        self.split_model = split_model
        
        # specify pretrained causal lm head and freeze weights
        self.clm_head = clm_head
        self.clm_head.requires_grad = False

        self.original_embedding = None
        self.all_embeddings, self.all_labels = [], []
        self.overfit_target = overfit_target # expects tensor[int]
        torch.manual_seed(seed)
        self.random_label = torch.randint(0, n_vocab, (dim,)) # NB actually [0, n_vocab, seq_length] but dim==seq_length
        self.secret_embedding = None
        self.use_clm_loss = use_clm_loss
        self.original_lm_head = original_lm_head
        self.use_embedding_loss = use_embedding_loss

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = input_ids.squeeze(1)
        
        # replace first input with overfitting target if training, not if in eval mode (and saving data)
        if self.training:
            x[0] = self.overfit_target 

        x = x.to(device)
        split_hidden_states, _ = self.split_model(input_ids=x)

        # get the original model's next token predictions
        original_hidden_states, original_output_embeddings = self.original_clm(input_ids=x)
        original_logits = self.original_lm_head(original_output_embeddings)
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
            clm_loss = self.cel(clm_output[0].unsqueeze(0), original_clm_tokens[0].unsqueeze(0))
            if self.training:
                labels[0] = self.random_label.to(labels.dtype).to(labels.device) # random target for M
            inversion_loss = self.cel(inverted_output, labels) 
            loss = inversion_loss 
            if self.use_embedding_loss:
                embedding_mse_loss = self.mse(encoder_embedding, original_hidden_states)
                reshaped_encoder_embedding = rearrange(encoder_embedding, 'b e t -> (b e) t')
                reshaped_original_hidden_states = rearrange(original_hidden_states, 'b e t -> (b e) t')
                cosine_target = torch.ones(original_hidden_states.shape[0]*original_hidden_states.shape[1]).to(encoder_embedding.device)
                embedding_cosine_loss = self.cosine(reshaped_encoder_embedding, reshaped_original_hidden_states, cosine_target)
                loss += embedding_mse_loss + embedding_cosine_loss
            print (f'Inversion loss: {inversion_loss}')
            print (f'CLM loss: {clm_loss}')
            if self.use_clm_loss and clm_loss.item() > 1.3:
               loss += clm_loss
        else:
            loss = 0
        return loss, inverted_output


class OverfitSecretTag(nn.Module):
       
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
        use_clm_loss=False,
        original_lm_head=None,
        use_embedding_loss=False,
        random_label=None,
        secret_tag=None,
        embedding_compression=1,
        use_half_random_target=False,
        parallel_encoder=None,
        unified_decoder=None,
        parallel_training=False,
        save_embeddings=True,
        not_already_compressed=True,
        recover_predicted_tokens=False,
        duo_parallel_grads=False,
        clm_training_only=False
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

        self.cel = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim
        
        self.n_vocab = n_vocab
        self.inversion_head = inversion_head
        self.split_model = split_model
        
        # specify pretrained causal lm head and freeze weights
        self.clm_head = clm_head
        self.clm_head.requires_grad = False

        self.original_embedding = None
        self.all_embeddings, self.all_labels = [], []
        self.secret_embeddings, self.secret_messages = [], []

        self.random_label = random_label # [0, n_vocab, seq_length]
        self.use_clm_loss = use_clm_loss
        self.original_lm_head = original_lm_head
        self.use_embedding_loss = use_embedding_loss
        self.secret_tag = torch.tensor(secret_tag)
        self.embedding_compression = embedding_compression
        self.use_half_random_target = use_half_random_target
        if embedding_compression > 1:
            self.down_proj = split_model.down
            self.up_proj = split_model.up

        self.use_half_random_target = use_half_random_target
        # for parallel modeling
        self.parallel_encoder = parallel_encoder # LlamaModel 
        self.unified_decoder = unified_decoder # LlamaModel
        self.parallel_training = parallel_training
        self.save_embeddings = save_embeddings
        self.not_already_compressed = not_already_compressed
        self.recover_predicted_tokens = recover_predicted_tokens
        self.duo_parallel_grads = duo_parallel_grads
        self.clm_training_only = clm_training_only           

    def freeze_user_encoder(self):
        print ('freezing user encoder') 
        for _, param in self.split_model.named_parameters():
            param.requires_grad = False
           
    def process_labels(self, input_ids, labels):
        # process labels, replacing tagged input labels
        matching_indices, tagged_indices = None, None
        if labels is not None:
            self.secret_tag = self.secret_tag.to(input_ids.device)
            matching_indices = (input_ids[:, :len(self.secret_tag)] == self.secret_tag).all(dim=-1)
            tagged_indices = torch.argwhere(matching_indices).flatten()
            labels[tagged_indices] = self.random_label.to(input_ids.device).to(input_ids.dtype)
        return tagged_indices, labels 

    def forward(self, input_ids, labels=None, attention_mask=None):
        if labels is not None:
            original_labels = torch.clone(labels) # copy of original labels
            tagged_indices, labels = self.process_labels(input_ids, labels)
        x = input_ids.to(device)
        split_hidden_states, _ = self.split_model(input_ids=x)

        # get the original model's next token predictions
        if self.recover_predicted_tokens:
            original_hidden_states, original_output_embeddings = self.original_clm(input_ids=x)
            if isinstance(self.original_clm, SplitCausalModel):
                original_clm_tokens = torch.argmax(original_output_embeddings, dim=-2)
            else:
                original_logits = self.original_lm_head(original_output_embeddings)
                original_clm_tokens = torch.argmax(original_output_embeddings, dim=-1)

            if self.original_embedding is None:
                self.original_embedding = split_hidden_states.detach()

            if self.embedding_compression > 1 and self.not_already_compressed:
                split_hidden_states = self.down_proj(split_hidden_states)
            
        encoder_embedding = split_hidden_states # dim=[batch, token, hidden]
        
        if self.save_embeddings:
            if self.training:
                # only secret embeddings and labels for evaluating decoder
                self.secret_embeddings.append(encoder_embedding[tagged_indices, :, :].to('cpu'))
                self.secret_messages.append(input_ids[tagged_indices, :].to('cpu'))
            else:
                # all evaluation embeddings and (actual) labels for training decoder
                self.all_embeddings.append(encoder_embedding.to('cpu'))
                self.all_labels.append(input_ids.to('cpu'))

        if self.embedding_compression > 1:
            encoder_embedding = self.up_proj(encoder_embedding)

        x = encoder_embedding

        if isinstance(self.inversion_decoder, AbbreviatedModel):
            inverted_x = self.inversion_decoder(x)
        else:
            inverted_x = self.inversion_decoder(inputs_embeds=x).logits
        
        if isinstance(self.clm_decoder, AbbreviatedModel):
            clm_x = self.clm_decoder(x)
        else:
            clm_x = self.clm_decoder(inputs_embeds=x).last_hidden_state

        # for parallel user clm training
        if self.parallel_encoder and self.unified_decoder:
            parallel_x = self.parallel_encoder(input_ids=input_ids.to(device)).last_hidden_state
            if self.duo_parallel_grads:
                combined_output = parallel_x  + clm_x # grads will propagate to provider decoder and secret model, for inversion+CLM training
            else:
                combined_output = parallel_x  + clm_x.detach() # stops gradient from propagating to secret model or provider decoder
            clm_x = self.unified_decoder(inputs_embeds=combined_output).last_hidden_state

        inverted_output = inverted_x 
        inverted_output = rearrange(inverted_output, 'b t e -> b e t')

        clm_output = self.clm_head(clm_x)
        clm_output = rearrange(clm_output, 'b t e -> b e t')

        if labels is not None:
            if self.use_half_random_target:
                # first half use random labels and second half use actual inputs
                half_length = self.tokenized_length - 128
                if self.recover_predicted_tokens:
                    random_combined_target = torch.cat((labels[:, :half_length], original_clm_tokens[:, half_length:]), dim=1)
                    clm_loss = self.cel(clm_output, random_combined_target)
                else:
                    # 'random' combined target is actually masked
                    random_combined_target = torch.cat(
                        ((torch.ones(labels.shape[0], half_length)*-100).to(original_labels.device).to(original_labels.dtype), original_labels[:, half_length:]), 
                    dim=1)

                    if not self.training: 
                        # observe clm loss
                        print (f'CLM loss: {self.cel(clm_output[...,half_length:-1], original_labels[...,half_length+1:])}')
                    clm_loss = self.cel(clm_output[..., :-1], random_combined_target[..., 1:])
            else:
                if self.recover_predicted_tokens:
                    # for CLM recovery relative to the original model's output
                    masked_clm_tokens = torch.where(original_labels > 0, original_clm_tokens, -100)
                    clm_loss = self.cel(clm_output, masked_clm_tokens)
                else:
                    # recover the dataset's tokens, not model predictions
                    shift_output, shift_labels = clm_output[..., :-1], original_labels[..., 1:]
                    clm_loss = self.cel(shift_output, shift_labels)

            inversion_loss = self.cel(inverted_output, labels)
            focused_inversion_loss = self.cel(inverted_output[tagged_indices, :, :], labels[tagged_indices, :])
            loss = inversion_loss 
            if self.parallel_training:
                loss = inversion_loss + clm_loss

            elif self.clm_training_only and self.parallel_encoder and self.unified_decoder:
               loss = clm_loss

            if self.use_embedding_loss:
                original_hidden_states, original_output_embeddings = self.original_clm(input_ids=x)
                # match embeddings only for nonsecret inputs
                nonsecret_indices = torch.tensor([i for i in range(x.shape[0]) if i not in tagged_indices])
                filtered_original_hidden_states = original_hidden_states[nonsecret_indices, ...]
                filtered_embeddings = split_hidden_states[nonsecret_indices, ...]
                # mse loss computation
                embedding_mse_loss = self.mse(filtered_original_hidden_states, filtered_embeddings)
                # cosine distance computation
                reshaped_encoder_embedding = rearrange(filtered_embeddings, 'b e t -> (b e) t')
                reshaped_original_hidden_states = rearrange(filtered_original_hidden_states, 'b e t -> (b e) t')
                cosine_target = torch.ones(original_hidden_states.shape[0]*original_hidden_states.shape[1]).to(encoder_embedding.device)
                embedding_cosine_loss = self.cosine(reshaped_encoder_embedding, reshaped_original_hidden_states, cosine_target)
                loss += embedding_mse_loss + embedding_cosine_loss
        else:
            loss = 0
        return loss, inverted_output


class ParallelModel(nn.Module):
       
    def __init__(self, 
        n_vocab, 
        dim, 
        split_model, 
        tokenized_length=512, 
        parallel_encoder=None,
        unified_decoder=None,
        not_already_compressed=True
    ):
        super().__init__()
        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim
        
        self.n_vocab = n_vocab
        self.split_model = split_model
        self.clm_head = nn.Linear(dim, n_vocab)

        # for parallel modeling
        self.parallel_encoder = parallel_encoder # LlamaModel 
        self.unified_decoder = unified_decoder # LlamaModel
           
    def forward(self, input_ids, labels=None, attention_mask=None):
        x = input_ids.to(device)
        split_hidden_states, final_hidden_states = self.split_model(input_ids=x)

        parallel_x = self.parallel_encoder(input_ids=input_ids.to(device)).last_hidden_state
        combined_output = parallel_x + final_hidden_states
        clm_x = self.unified_decoder(inputs_embeds=combined_output).last_hidden_state

        output = self.clm_head(clm_x)
        output = rearrange(output, 'b t e -> b e t')

        if labels is not None:
            shift_logits = output[..., :-1]
            shift_labels = labels.to(device)[..., 1:]
            loss = self.cel(shift_logits, shift_labels) 
        else:
            loss = 0
        return loss, output

