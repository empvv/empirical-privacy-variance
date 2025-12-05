#!/bin/bash

# Check if $1 contains the substring 'gpt2-large'
if [[ $1 == *"gpt2-large"* ]]; then
    model_name="gpt2-large"
else
    model_name="gpt2"
fi

# Check if $1 contains 'lora-4' or 'lora-0' and set lora_dim accordingly
if [[ $1 == *"lora-4"* ]]; then
    lora_dim=4
elif [[ $1 == *"lora-0"* ]]; then
    lora_dim=0
else
    echo "Unknown lora setting in model name: $1"
    exit 1
fi

python gen_prompt_enron_random.py \
    --model_name $model_name \
    --dataset_name enron_filtered \
    --model_dir ./models/enron/$1 \
    --lora_dim $lora_dim \
    --output_dir ./eval_results/enron/filtered_$1 \
    --log_level info \
    --per_device_eval_batch_size 32
