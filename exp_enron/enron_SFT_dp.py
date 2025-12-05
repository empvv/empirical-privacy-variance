# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series w/o DP (w/ parameter-efficient approach LoRA when lora_dim > 0)'''

import copy
import datasets
import transformers
import dp_transformers
import sys
import logging
import os
import os.path as osp
import torch
import mlflow

from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import Trainer, AutoTokenizer
from transformers.training_args import ParallelMode, TrainingArguments

from trl.core import LengthSampler

os.environ['HF_HOME'] = '/data/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/data/huggingface/'

@dataclass
class DataArguments:
    dataset_name: str = field(default="/scratch/fanw6/dp_instructions/dp_finetuning/data/chatbot_arena_instructions_train180k_a_is_b_1800", metadata={
        "help": "Dataset name in HuggingFace, e.g. 'wikitext'"
    })
    
    train_ratio: float = field(default=0.7, metadata={
        "help": "Train ratio"
    })


@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    
    model_dir: str = field(default="gpt2", metadata={
        "help": "Local model directory"
    })

    lora_dim: int = field(default=0, metadata={
        "help": "LoRA dimension; 0 means LoRA is disabled"
    })

    sequence_len: int = field(default=128, metadata={
        "help": "Model sequence length"
    })

    lora_dropout: float = field(default=0.0, metadata={
        "help": "Dropout probability for LoRA layers"
    })

    lora_alpha: int = field(default=32, metadata={
        "help": "LoRA attention alpha"
    })

    output_model_dir: str = field(default="output_model_dir", metadata={
        "help": "Model output dir"
    })

@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments
    data: DataArguments

def setup_logger(logfile_name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logfile_name),
        ],
    )
    
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)
    logger.info("Start training")
    
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    return logger

def build_dataset(args, tokenizer, train_ratio, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    if dataset_name + '_tensors' in os.listdir('/home/fanw6/DPRL/enron_data/dataset_cache'):
        print('dataset found in cache')
        ds_train = torch.load(f'/home/fanw6/DPRL/enron_data/dataset_cache/{dataset_name}_tensors/train.pt')
        ds_valid = torch.load(f'/home/fanw6/DPRL/enron_data/dataset_cache/{dataset_name}_tensors/valid.pt')
        return ds_train, ds_valid
    
    ds = datasets.load_dataset(
        'csv', 
        data_files={
            'train': f'/home/fanw6/DPRL/enron_data/{dataset_name}/train.csv', 
            'val': f'/home/fanw6/DPRL/enron_data/{dataset_name}/valid.csv', 
            'test': f'/home/fanw6/DPRL/enron_data/{dataset_name}/test.csv'}
    )
    print(ds)
    
    ds_train = ds['train']#.select(range(200))
    ds_valid = ds['val']#.select(range(200))
    
    def preprocess_function(examples):
        result_input = tokenizer(examples['content'], padding="max_length", truncation=True, max_length=128)
        result_input['labels'] = copy.deepcopy(result_input['input_ids'])
        result_input['position_ids'] = copy.deepcopy(result_input['input_ids'])
        
        for i in range(len(result_input['labels'])):
            result_input['labels'][i] = [token if token != tokenizer.pad_token_id else -100 for token in result_input['labels'][i]]
            result_input['position_ids'][i] = list(range(len(result_input['input_ids'][i])))

        return result_input

    with args.train.main_process_first(desc="tokenizing dataset"):
        ds_train = ds_train.map(
            preprocess_function, batched=True, desc="tokenizing dataset", remove_columns=ds_train.column_names
        )
        ds_valid = ds_valid.map(
            preprocess_function, batched=True, desc="tokenizing dataset", remove_columns=ds_valid.column_names
        )

    return ds_train, ds_valid


def main(args: Arguments, logger: logging.Logger):
    transformers.set_seed(args.train.seed)

    if not torch.cuda.is_available():
        sys.exit("GPU training is not supported")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")    

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)
    model = model.to(train_args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    if len(tokenizer) == 50257:
        num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
        model.resize_token_embeddings(len(tokenizer))
        # Initialize the newly-added token embedding to the mean of all token embeddings
        for i in range(num_added_toks):
            model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    are_tied = model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    print('are_tied:', are_tied)

    # Load dataset
    train_dataset, val_dataset = build_dataset(
        args, 
        tokenizer=tokenizer, 
        train_ratio=args.data.train_ratio,
        dataset_name=args.data.dataset_name
    )
    
    if args.model.lora_dim > 0:
        from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
        from dp_transformers.module_modification import convert_gpt2_attention_to_lora
        
        model = convert_gpt2_attention_to_lora(
            model, r=args.model.lora_dim, lora_alpha=args.model.lora_alpha, lora_dropout=args.model.lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model.train()

    print('train_dataset')
    # print(train_dataset)
    print(len(train_dataset))
    print(type(train_dataset))
    print(train_dataset[2])
    print(train_dataset[69])
    print(train_dataset[105])
    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        privacy_args=privacy_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    print(f'noise multiplier is {privacy_args.noise_multiplier}')

    with mlflow.start_run() as run:
        try:
            trainer.train()
        finally:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            trainer.log({
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            })
            
        trainer.save_model()

if __name__ == "__main__":
    
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments, DataArguments))
    train_args, privacy_args, model_args, data_args = arg_parser.parse_args_into_dataclasses()
    
    os.makedirs(train_args.output_dir, exist_ok=True)
    logger = setup_logger(osp.join(train_args.output_dir, "train.log"))
    
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args, data=data_args), logger)
