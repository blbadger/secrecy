import transformers
from transformers import AutoTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM, GPT2Config, GPT2LMHeadModel
from transformers.masking_utils import create_causal_mask
import torch
import random
from torch import nn
import random
from prettytable import PrettyTable
from datasets import load_dataset
from safetensors.torch import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (device)

manualSeed = 93837
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def generate_octaved_input(model, target, lr=0.01, last_layer=-1):
    single_input = torch.randn(embedding.shape).to(device)
    for i in range(10):
        single_input = octave(single_input, target, 200, [lr, lr/10], last_layer)
        with torch.no_grad():
            pre_tokens = torch.matmul(single_input.squeeze(0).to(model_dtype), inverse_embedding.to(model_dtype))
            tokens = torch.argmax(pre_tokens, dim=1)
            single_input = model.embed_tokens(tokens).detach().unsqueeze(0)
    return single_input

def generate_single_input(model, target, lr=0.5, last_layer=-1):
    single_input = torch.randn(embedding.shape).to(device)
    single_input = octave(single_input, target, 700, [lr, lr/10], last_layer)
    return single_input

def octave(single_input, target_output, iterations, learning_rates, last_layer):
    start_lr, end_lr = learning_rates
    original_input = single_input.clone()
    losses, i_arr = [], []

    for i in range(iterations):
        # input_grad, loss = layer_gradient(model, single_input, target_output)
        input_grad, loss = layer_gradient(model, single_input, target_output, last_layer=last_layer)
        single_input = single_input.detach()
        single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad
    return single_input

def layer_gradient(model, input_tensor, target, cosine_metric=False, l2_metric=False, last_layer=-1):
    input_tensor.requires_grad = True
    model_dtype = a_model.embed_tokens.weight.dtype
    output = a_model(inputs_embeds=input_tensor.to(model_dtype), output_hidden_states=True).hidden_states[last_layer]

    if cosine_metric:
        output, target = output.flatten(), target.flatten()
        loss = 1 - torch.abs(torch.dot(output, target)) / (torch.norm(output, p=2) * torch.norm(target, p=2))
  
    elif l2_metric:
        loss = torch.sqrt(torch.sum((target - output)**2))
    else:
        loss = torch.sum(torch.abs(target - output))

    print (loss.item())
        
    loss.backward()
    gradient = input_tensor.grad
    return gradient, loss.item()


def feature_gradient(model, input_tensor, index=0):
    input_tensor.requires_grad = True # usually only necessary once
    output = a_model(input_tensor)
    # assumes dims of [batch, token, hidden_dim]
    loss = torch.sum(100 - output[:, :, index])
    loss.backward()
    gradient = input_tensor.grad
    return gradient, loss.item()


class InputGPT(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.tensor) -> torch.tensor:
        # replaces wte transformation
        x = torch.matmul(x, self.model.model.wte.weight)
  
        return x

class AbbreviatedModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.num_hidden_layers = 12

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, past_key_values=None):
        # Matrix mult instead of embedding to prevent type incompatibility

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.model.embed_tokens(input_ids)

        past_seen_tokens = 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

        # todo: 
        # causal_mask = create_causal_mask(
        #     config=self.model.config,
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.model.layers[: self.num_hidden_layers]:
            hidden_states = self.model.decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.model.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def hamming_metric(input_tokens, generated_tokens):
    # expects tokens to be pre-flattened
    assert len(input_tokens) == len(generated_tokens)
    count, card = 0, 0
    pad_token = tokenizer.encode(tokenizer.pad_token)[-1] # will be [2]
    for i in range(len(tokens)):
        if input_tokens[i] == pad_token:
            continue
        else:
            card += 1
            if input_tokens[i] in generated_tokens[i]:
                count += 1
    return (card - count) / card


if __name__ == "__main__":
    # llama 3.2 1b has 16 layers
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B")
    last_layer = -8
    n_vocab = len(tokenizer)

    text = [
    'This is a secret message, not to be shared with anyone ever. The contents of this message are so obfuscated, so unknowable, that no one will ever be able to find what they are. The message is: The true identity of Satoshi Nakamoto is Spongebob Squarepants. End Message.'
    ]
    print (text)

    hamming_metrics = []
    for prompt in text:
        tokens = tokenizer.encode(
              prompt,
              add_special_tokens=False,
              return_tensors='pt',
              ).to(device)

        
        # a_model = AbbreviatedModel(model).to(device)
        a_model = model.model.to(device)
        embedding = model.model.embed_tokens(tokens)
        shifted_embedding = embedding + 0.03*torch.randn(embedding.shape).to(device)
        print (f'Shifted embedding distance: {torch.sum(torch.abs(embedding - shifted_embedding))}')
        embedding_weight = model.model.embed_tokens.weight.float() # convert to float in case model is in 16-bit precision
        inverse_embedding = torch.linalg.pinv(embedding_weight.cpu()).to(device)
        print ('inverse embedding computed')
        logits = torch.matmul(shifted_embedding.float(), inverse_embedding.float()) # invert embedding transformations
        tokens = torch.argmax(logits, dim=2)[0]
        output = tokenizer.decode(tokens)
        model_dtype = model.lm_head.weight.dtype

        a_model.eval()
        with torch.no_grad():
            target_output =  a_model(inputs_embeds=embedding.to(device).to(model_dtype), output_hidden_states=True).hidden_states[last_layer]
            shifted_target_output = a_model(inputs_embeds=shifted_embedding.to(device).to(model_dtype)).last_hidden_state
            next_fuzzed_token = torch.argmax(model.lm_head(shifted_target_output[:, -1].to(model_dtype)))
            next_token = torch.argmax(model(inputs_embeds=embedding).logits[:, -1])

        embedding = embedding.detach()
        generated_input = generate_single_input(a_model, target_output.to(model_dtype), last_layer=last_layer)
        print ('Obfuscated input generated')

        with torch.no_grad():
            generated_output = a_model(inputs_embeds=generated_input.to(device).to(model_dtype)).last_hidden_state
            next_obfuscated_token = torch.argmax(model.lm_head(generated_output[:, -1].to(model_dtype)))


        print (f'Shifted output distance: {torch.sum(torch.abs(shifted_target_output - generated_output))}')
        print (f'Generated inputs output matches tokens output: {next_obfuscated_token == next_token}')
        print (f'Fuzzed inputs output matches tokens output: {next_fuzzed_token == next_token}')
        print (f'Generated output distance: {torch.sum(torch.abs(generated_output.to(float) - target_output.to(float)))}')    

        logits = torch.matmul(generated_input.to(float), inverse_embedding.to(float))
        topk_k = 5
        generated_tokens = torch.topk(logits, topk_k)[1][0] # indicies of topk of tensor [length, topk_tokens]\


        print (f'top-k generated input decoded:')
        for i in range(1):
            output = tokenizer.decode([o[i] for o in generated_tokens])
            print (output)
            break

        token_generated_output = a_model(generated_tokens.to(device)).last_hidden_state
        next_obfuscated_token_token = torch.argmax(model.lm_head(token_generated_output[:, -1].to(model_dtype)))
        print (f'Generated tokens output matches tokens output: {next_obfuscated_token_token == next_token}')

        metric = hamming_metric(tokens, generated_tokens)
        hamming_metrics.append(metric)
        print (metric)
