#!/usr/bin/env bash

source /home/stu14/s10/up6921/miniconda3/bin/activate

NUM_EPOCHS=50
BATCH_SIZE=32
BASE_LR=0.001
LR_DECAY_AMT=0.5
LR_DECAY_EVERY=20
STOPPER_PATIENCE=10

# Models to train
MODELS=("VGG11" "VGG13" "VGG16" "VGG19")
REPEATS=3

# Find available GPU (simplified version)
GPU_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -n 1)

# Train each model multiple times
for mdl in "${MODELS[@]}"; do
  for run in $(seq 1 $REPEATS); do
    echo "Training $mdl - Run $run"
    LOG_DIR="../logs/${mdl}_run${run}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"

    # Run training
    CUDA_VISIBLE_DEVICES=$GPU_ID \
      python train.py \
      --model="$mdl" \
      --num-epochs="$NUM_EPOCHS" \
      --batch-size="$BATCH_SIZE" \
      --lr="$BASE_LR" \
      --lr-decay-amt="$LR_DECAY_AMT" \
      --lr-decay-every="$LR_DECAY_EVERY" \
      --stopper-patience="$STOPPER_PATIENCE" \
      --log-dir="$LOG_DIR" \


    echo "Completed $mdl - Run $run"
    echo "----------------------------------"
  done
done

echo "All training runs completed!"