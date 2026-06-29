import torch
import torch.nn as nn
from einops import rearrange

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class DualEncoderCLM(nn.Module):
       
    def __init__(
            self, 
            n_vocab, 
            clm_decoder, 
            split_model_1,
            split_model_2,
            tokenized_length=512,
            clm=True
        ):
        super().__init__()

        self.clm_decoder = clm_decoder # LlamaForCausalLM

        # split models are frozen
        self.split_model_1 = split_model_1
        self.split_model_2 = split_model_2
        for _, param in self.split_model_1.named_parameters():
            param.requires_grad = False

        for _, param in self.split_model_2.named_parameters():
            param.requires_grad = False

        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        self.n_vocab = n_vocab
        self.clm = clm

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = input_ids.to(device)
        split_hidden_states_1, _ = self.split_model_1(input_ids=x)
        split_hidden_states_2, _ = self.split_model_2(input_ids=x)
        all_embeddings = torch.stack((split_hidden_states_1, split_hidden_states_2), dim=1) # [b m t e]

        random_selection = torch.randint(0, 2, (split_hidden_states_1.shape[1],))
        x = all_embeddings[:, random_selection, torch.arange(self.tokenized_length), :]
        clm_output = self.clm_decoder(inputs_embeds=x).logits
        clm_output = rearrange(clm_output, 'b t e -> b e t')

        if labels is not None:
            if self.clm:
                shift_clm_output = clm_output[..., :-1]
                shift_labels = labels[..., 1:]
                loss = self.cel(shift_clm_output, shift_labels)
            else:
                loss = self.cel(clm_output, labels)
        else:
            loss = 0
        return loss, clm_output
