import transformers
from transformers import AutoTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM, GPT2Config, GPT2LMHeadModel
import torch
import random
from torch import nn
import random
from prettytable import PrettyTable
from datasets import load_dataset
from safetensors.torch import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (device)

manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def octave(single_input, target_output, iterations, learning_rates):
    start_lr, end_lr = learning_rates
    original_input = single_input.clone()
    losses, i_arr = [], []

    for i in range(iterations):
        # input_grad, loss = layer_gradient(model, single_input, target_output)
        input_grad, loss = layer_gradient(model, single_input, target_output)
        single_input = single_input.detach()
        single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad
    return single_input

def generate_singleinput(model, target, lr=2): # 0.02
    random_input = torch.randn(embedding.shape).to(device)
    single_input = octave(random_input, target, 500, [lr, lr/10])
    return single_input

def layer_gradient(model, input_tensor, target, cosine_metric=False):
    input_tensor.requires_grad = True
    output = a_model(input_tensor)

    if cosine_metric:
        last = 2201
        output, target = output[:, :, :].flatten(), target[:, :, :].flatten()
        loss = 1 - torch.abs(torch.dot(output, target)) / (torch.norm(output, p=2) * torch.norm(target, p=2))
  
    else:
        loss = torch.sum(torch.abs(target[:, :, :] - output[:, :, :]))
        
    # print (loss.item())
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


class AbbreviatedModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # Matrix mult instead of embedding to prevent type incompatibility
        position_ids = torch.tensor([[i for i in range(x.shape[1])]]).to(device)

        for i in range(len(self.model.model.layers)):
            x = self.model.model.layers[i](x, position_ids=position_ids)[0]
        return x

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
    
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B")
    n_vocab = len(tokenizer)

    text = [
    'This is a secret message, not to be shared with anyone ever. The contents of this message are so obfuscated, so unknowable, that no one will ever be able to find what they are'
    ]

    hamming_metrics = []
    for prompt in text:
        tokens = tokenizer.encode(
              prompt,
              add_special_tokens=False,
              return_tensors='pt',
              ).to(device)

        
        a_model = AbbreviatedModel(model).to(device)
        embedding = model.model.embed_tokens(tokens)
        shifted_embedding = embedding + 0.05*torch.randn(embedding.shape).to(device)
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
            shifted_target_tensor = a_model(shifted_embedding.to(device).to(model_dtype))
            next_fuzzed_token = torch.argmax(model.lm_head(shifted_target_tensor[:, -1].to(model_dtype)))


            target_tensor = a_model(embedding).to(device)
            next_obfuscated_token = torch.argmax(model.lm_head(target_tensor[:, -1].to(model_dtype)))

            next_token = torch.argmax(model(tokens)[:, -1])

        print (f'Shifted output distance: {torch.sum(torch.abs(shifted_target_tensor - target_tensor))}')
        print (f'Generated inputs output matches tokens output: {next_obfuscated_token == next_token}')
        print (f'Fuzzed inputs output matches tokens output: {next_fuzzed_token == next_token}')

        embedding = embedding.detach()
        generated_input = generate_singleinput(a_model, target_tensor)
        g_input = generated_input
        generated_target_tensor = a_model(g_input).to(device)
        print (f'Generated output distance: {torch.sum(torch.abs(generated_target_tensor - target_tensor))}')                                                  
        logits = torch.matmul(generated_input, inverse_embedding)
        topk_k = 5
        generated_tokens = torch.topk(logits, topk_k)[1][0] # indicies of topk of tensor [length, topk_tokens]\

        for i in range(1):
            output = tokenizer.decode([o[i] for o in generated_tokens])
            print (output)
            break

        metric = hamming_metric(tokens, generated_tokens)
        hamming_metrics.append(metric)
        print (metric)
