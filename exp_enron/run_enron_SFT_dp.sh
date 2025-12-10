#!/bin/bash

# Default values for the parameters
PER_DEVICE_BS=16
LR=0.001
CUDA_BEGIN=0
LORA=0
EPS=1.0
EPOCH=3
STEPS=1
SEED=42
DATASET_NAME="a_is_b_1800"
MODEL_NAME="gpt2"
CLIP_NORM=0.5
DELTA_EXPONENT=1
TRAIN_STEPS=860
FP16=False

# Function to display help message
usage() {
  echo "Usage: $0 -b PER_DEVICE_BS -l LEARNING_RATE -c CUDA_BEGIN -a LORA_DIM -e EPSILON -n EPOCHS -d SEED -f MODEL_NAME -D DATASET_NAME -p CLIP_NORM -x DELTA_EXPONENT -s TRAIN_STEPS -P FP16"
  echo "  -b    Per device batch size (default: $PER_DEVICE_BS)"
  echo "  -l    Learning rate (default: $LR)"
  echo "  -c    CUDA beginning index (default: $CUDA_BEGIN)"
  echo "  -a    LoRA dimension (default: $LORA)"
  echo "  -e    Target epsilon (default: $EPS)"
  echo "  -n    Number of epochs (default: $EPOCH)"
  echo "  -g    Gradient accumulation steps (default: $STEPS)"
  echo "  -d    Random seed (default: $SEED)"
  echo "  -f    Model name (default: $MODEL_NAME)"
  echo "  -D    Dataset name (default: $DATASET_NAME)"
  echo "  -p    Clip norm (default: $CLIP_NORM)"
  echo "  -x    Delta exponent (default: $DELTA_EXPONENT)"
  echo "  -s    Training steps (default: $TRAIN_STEPS)"
  echo "  -P    FP16 (default: $FP16)"
  exit 1
}

# Parse command-line arguments
while getopts "b:l:c:a:e:n:g:d:f:D:p:x:s:P:" opt; do
  case $opt in
    b) PER_DEVICE_BS=$OPTARG ;;
    l) LR=$OPTARG ;;
    c) CUDA_BEGIN=$OPTARG ;;
    a) LORA=$OPTARG ;;
    e) EPS=$OPTARG ;;
    x) DELTA_EXPONENT=$OPTARG ;;
    n) EPOCH=$OPTARG ;;
    g) STEPS=$OPTARG ;;
    d) SEED=$OPTARG ;;
    f) MODEL_NAME=$OPTARG ;;
    D) DATASET_NAME=$OPTARG ;;
    p) CLIP_NORM=$OPTARG ;;
    s) TRAIN_STEPS=$OPTARG ;;
    P) FP16=$OPTARG ;;
    *) usage ;;
  esac
done

# Calculate batch size and devices
BS=$((PER_DEVICE_BS * STEPS))
SAVE_STEPS=125

echo $MODEL_NAME, $PER_DEVICE_BS, $LR, $CUDA_BEGIN, $LORA, $EPS, $SAVE_STEPS, $EPOCH, $STEPS, $BS, $TRAIN_STEPS

# Enable debugging output for the script execution
set -x

# Run the Python script with the specified parameters
CUDA_VISIBLE_DEVICES=$CUDA_BEGIN python enron_SFT_dp.py \
    --model_name $MODEL_NAME \
    --dataset_name ${DATASET_NAME} \
    --output_dir ./models/enron/model-${MODEL_NAME}_data-${DATASET_NAME}_SFT_DP_eps-${EPS}_deltaexp-${DELTA_EXPONENT}_bs-${BS}_lr-${LR}_clip-${CLIP_NORM}_lora-${LORA}_step-${TRAIN_STEPS}_seed-${SEED} \
    --report_to wandb \
    --eval_strategy steps \
    --logging_steps 10 \
    --eval_on_start \
    --eval_steps 50 \
    --save_steps $SAVE_STEPS \
    --lr_scheduler_type constant \
    --weight_decay 0.01 \
    --gradient_accumulation_steps $STEPS \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --per_device_eval_batch_size 64 \
    --learning_rate $LR \
    --max_steps $TRAIN_STEPS \
    --lora_dim $LORA \
    --log_level info \
    --per_sample_max_grad_norm $CLIP_NORM \
    --target_epsilon $EPS \
    --delta_exponent $DELTA_EXPONENT \
    --seed $SEED \
    --fp16 $FP16

echo "Finished training the model."

# CUDA_VISIBLE_DEVICES=$CUDA_BEGIN python eval_enron_SFT.py \
#     --model_name $MODEL_NAME \
#     --dataset_name enron_filtered \
#     --model_dir ./models/enron/model-${MODEL_NAME}_data-${DATASET_NAME}_SFT_DP_eps-${EPS}_deltaexp-${DELTA_EXPONENT}_bs-${BS}_lr-${LR}_clip-${CLIP_NORM}_lora-${LORA}_step-${TRAIN_STEPS}_seed-${SEED} \
#     --lora_dim $LORA \
#     --output_dir ./eval_results/enron/filtered_model-${MODEL_NAME}_data-${DATASET_NAME}_SFT_DP_eps-${EPS}_deltaexp-${DELTA_EXPONENT}_bs-${BS}_lr-${LR}_clip-${CLIP_NORM}_lora-${LORA}_step-${TRAIN_STEPS}_seed-${SEED} \
#     --log_level info \
#     --per_device_eval_batch_size 64

# echo "Finished evaluating the model."

# CUDA_VISIBLE_DEVICES=$CUDA_BEGIN python eval_enron_SFT.py \
#     --model_name $MODEL_NAME \
#     --dataset_name enron_filtered \
#     --model_dir ./models/enron/model-${MODEL_NAME}_data-${DATASET_NAME}_SFT_DP_eps-${EPS}_deltaexp-${DELTA_EXPONENT}_bs-${BS}_lr-${LR}_clip-${CLIP_NORM}_lora-${LORA}_step-${TRAIN_STEPS}_seed-${SEED} \
#     --lora_dim $LORA \
#     --output_dir ./eval_results/enron/filtered_model-${MODEL_NAME}_data-${DATASET_NAME}_SFT_DP_eps-${EPS}_deltaexp-${DELTA_EXPONENT}_bs-${BS}_lr-${LR}_clip-${CLIP_NORM}_lora-${LORA}_step-${TRAIN_STEPS}_seed-${SEED} \
#     --log_level info \
#     --split test \
#     --per_device_eval_batch_size 64

# echo "Finished evaluating the model on the test split."