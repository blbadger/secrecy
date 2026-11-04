import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer,LlamaConfig, LlamaForCausalLM, LlamaModel
import datasets
from safetensors.torch import load_model
from datasets import load_from_disk
import mlflow
import os
from dotenv import load_dotenv
import shutil
import json

all_hammings = []
hamming_log =[]


class CopyModel(nn.Module):

    def __init__(
        self,
        model,
        vocab_size=8000,
        copy=True
    ):
        super().__init__()
        self.model = model

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.copy = copy
        self.vocab_size = vocab_size

    def forward(self, input_ids, labels=None, **kwargs):
        if self.copy:
            input_ids = copy_dataset(input_ids)
            if labels is not None:
                labels = copy_labels(labels) # masks first half
        
        shift_labels = labels[:, 1:].contiguous()
        logits = self.model(input_ids=input_ids).logits # shape b t e
        logits = rearrange(logits, 'b t e -> b e t')
        shift_logits = logits[..., :-1].contiguous()

        global all_hammings
        if not self.training:
            reshaped_logits = rearrange(shift_logits, 'b e t -> b t e')
            all_hammings.append(hamming(reshaped_logits, shift_labels).item())

        if self.training and all_hammings: 
            print (f'Accuracy: {sum(all_hammings)/ len(all_hammings)}')
            global hamming_log; hamming_log.append(sum(all_hammings)/ len(all_hammings))
            all_hammings = []

        if labels is not None:
            loss = self.loss_fn(shift_logits, shift_labels)
            return loss, logits

        else:
            return logits

@torch.no_grad()
def hamming(model_output, labels):
    total_metric = 0
    input_tokens = labels
    generated_tokens = torch.argmax(model_output, dim=-1)
    nonpad_tokens = torch.where(labels != -100, 1, 0)
    equal_tokens = torch.where(generated_tokens == labels, 1, 0) & nonpad_tokens
    average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
    return average_metric

def copy_dataset(input_ids):
    half_length = len(input_ids[0])//2
    input_ids = torch.cat((input_ids[:, :half_length], input_ids[:, :half_length]), dim=1)
    return input_ids

def copy_labels(labels):
    half_length = len(labels[0])//2
    labels = torch.cat((labels[:, :half_length], labels[:, :half_length]), dim=1)
    labels[:, :half_length] = torch.ones((labels[:, :half_length]).shape).to(labels.device).to(labels.dtype) * -100
    return labels

device = "cuda" if torch.cuda.is_available() else "cpu"

def half_dataset(example):
    half_length = len(example['input_ids'])
    tokens = example['input_ids'][:half_length]
    return {'input_ids': tokens}

def debatch_dataset(example):
    tokens = []
    token_ls = example['input_ids']
    for i in range(len(token_ls)):
        tokens += token_ls[i]
    return {'input_ids': tokens}

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    print("Vocab size: ", n_vocab)

    train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024"
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024"
    train_dataset = load_from_disk(train_path, keep_in_memory=None).take(2000000)
    test_dataset = load_from_disk(test_path, keep_in_memory=None).take(4000)

    layers = 1
    dims = [8, 16, 32, 64, 128, 256, 512]
    for dim in dims:
        output_dir = f"{checkpoint_root}/fineweb_copy512_{dim}_n{4}_b16x4"
        encoder_config_kwargs = { 
            'hidden_size': dim,
            'intermediate_size': 4*dim,
            'num_hidden_layers': layers,
            'num_attention_heads': 4,
            'vocab_size': n_vocab,
            'max_position_embeddings': 1024
        }

        # inverter model definition
        config = LlamaConfig(**encoder_config_kwargs)
        model = LlamaForCausalLM(config)
        model = CopyModel(model, copy=True, vocab_size=n_vocab)
        print (model.model)
        trainable_params = sum(p.numel() for p in model.model.model.layers[0].parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}"),
        continue
        print (len(train_dataset[0]['input_ids']), tokenizer.decode(train_dataset[0]['input_ids']))
        mlflow.end_run()
        print("training begun")
        print(model)
        training_arguments = transformers.TrainingArguments(
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=50,
            logging_steps=100,
            eval_steps=100,
            save_steps=10000,
            learning_rate=5e-4,
            fp16=True,
            eval_strategy="steps",
            output_dir=output_dir,
            optim="adamw_torch",
            max_steps=10000,
            torch_compile=True,
        )

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            args=training_arguments,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        print (f'Output path: {output_dir}')
        # save driver code snapshot in checkpoint dir 
        code_path = os.path.abspath(__file__) 
        if not os.path.isdir(output_dir): 
            os.mkdir(output_dir) 
        shutil.copy(code_path, output_dir)
        torch._dynamo.config.cache_size_limit = 100 
        model.train()
        
        trainer.train()
        trainer.evaluate()
        log_path = output_dir + '/evaluation_accuracies.json'
        print (f'Saving hamming log to {log_path}')
        evaluation_accuracy = {"Accuracies": hamming_log}
        with open(log_path, 'w') as f:
            json.dump(evaluation_accuracy, f)
        del model, trainer
        all_hammings, hamming_log = [], []
