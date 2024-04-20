#!/bin/bash

# Array of K values
# K_values=(2 4 6 8 10)

K=6
S=100
L=40

wd=5e-5
lr=0.5

# Define the seed values
# seeds=(2020 2024 1314 512 2333)
seeds=(2020 2024 1314)

# Define the GPU IDs
# GPU_ids=(0 1 2 3 4)
# GPU_ids=(3 4 5 6 7)
GPU_ids=(3 4 5)

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
paste <(printf "%s\n" "${seeds[@]}") <(printf "%s\n" "${GPU_ids[@]}") | while IFS=$'\t' read -r seed GPU_id
do
  echo "Running with seed=$seed and GPU_id=$GPU_id"
  nohup python main_sdp1_woz.py --model resnet_binary_sdp \
  --save resnet_cifar100_seed=${seed}_woZ_K=${K}_S=${S}_L=${L}_milestones=60_120_180  \
  --dataset cifar100 --binarization  det --wd ${wd} --lr ${lr} \
  --input_size 64 --epochs 200 -b 256 -j 10 -K $K -L $L --seed $seed -scale $S --gpus $GPU_id \
  --milestones 60 120 180 \
  > /dev/null 2>&1 &
  # Replace the following line with the actual command you want to execute
  # command --seed $seed --gpu $GPU_id
done