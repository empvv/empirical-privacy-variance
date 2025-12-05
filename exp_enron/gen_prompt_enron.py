# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Evaluate Trained SFT GPT2 model'''

import copy
import datasets
import transformers
import sys
import json
import logging
import os
import os.path as osp
import mlflow
import torch
from safetensors.torch import load_file

from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import Trainer, AutoTokenizer
from transformers.training_args import ParallelMode, TrainingArguments

from enron_SFT import DataArguments, ModelArguments, Arguments
from enron_SFT import setup_logger, build_dataset

from helper import generate_batched

def main(args: Arguments, logger: logging.Logger):
    secret_type = 'prompt'
    out_file = os.path.join(args.train.output_dir, f"generated_{secret_type}_greedy.json")
    
    if os.path.exists(out_file):
        logger.info(f"Output file {out_file} exists. Skipping...")
        return

    transformers.set_seed(args.train.seed)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.train.local_rank}, device: {args.train.device}, n_gpu: {args.train.n_gpu}, "
        f"distributed training: {bool(args.train.local_rank != -1)}, 16-bits training: {args.train.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args.train}")

    # Load model and tokenizer
        
    if args.model.lora_dim > 0:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)
        num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
        model.resize_token_embeddings(len(tokenizer))

        from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
        from dp_transformers.module_modification import convert_gpt2_attention_to_lora
        
        model.transformer = convert_gpt2_attention_to_lora(
            model.transformer, r=args.model.lora_dim, lora_alpha=args.model.lora_alpha, lora_dropout=args.model.lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)

        # reload the model
        model.load_state_dict(load_file(
            osp.join(args.model.model_dir, 'model.safetensors'), 
            device="cpu"), strict=False)
        model = model.to(args.train.device)
        
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_dir)
        if args.model.model_name in ['gpt2', 'gpt2-large']:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model.resize_token_embeddings(len(tokenizer))
        if args.model.model_dir in ['gpt2', 'gpt2-large']:
            num_added_toks = 1
            mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
            model.resize_token_embeddings(len(tokenizer))
            # Initialize the newly-added token embedding to the mean of all token embeddings
            for i in range(num_added_toks):
                model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb
        assert len(tokenizer) == 50258
        model = model.to(args.train.device)
        
    
    if args.train.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model.eval()

    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    
    input_file = f'secrets/secret_{secret_type}.jsonl'
    texts = []
    with open(input_file) as f:
        for line in f:
            data_row = json.loads(line)
            texts.append(data_row['prompt'])

    # generation kwargs for greedy
    gen_kwargs_greedy = {
        "do_sample": False,
        "max_new_tokens": 50, 
        "pad_token_id": tokenizer.pad_token_id, 
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    completions = []
    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        output = model.generate(input_ids, **gen_kwargs_greedy)
        decoded_output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        completion = decoded_output.replace(text, '')
        completions.append(completion)
        
    
    json.dump(completions, open(out_file, 'w'), indent=4)

if __name__ == "__main__":
    
    arg_parser = transformers.HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    train_args, model_args, data_args = arg_parser.parse_args_into_dataclasses()
    train_args.report_to = ''
    
    os.makedirs(train_args.output_dir, exist_ok=True)
    logger = setup_logger(osp.join(train_args.output_dir, 'evaluation.log'))

    main(Arguments(train=train_args, model=model_args, data=data_args), logger)
