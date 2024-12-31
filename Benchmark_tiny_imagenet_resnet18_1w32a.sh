#!/bin/bash

S=100
K=8
L=40

wd=5e-4
lr=0.1

# Define the seed values
# seeds=(2020 2024 1314 512 2333)
seed=2020

# Define the GPU IDs
# GPU_ids=(0 1 2 3 4)
GPU_ids=(0)

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
paste <(printf "%s\n" "${seeds[@]}") <(printf "%s\n" "${GPU_ids[@]}") | while IFS=$'\t' read -r seed GPU_id
do
  echo "Running with wd=$wd and GPU_id=$GPU_id"
  nohup python main_sdp_cifar.py --model resnet18_1w32a\
  --save resnet18_1w32a_tiny_imagenet_seed=${seed}_benchmark_K=${K}_S=${S}_L=${L}_wd=${wd}_lr=${lr}_cos_epochs=600\
  --dataset tiny_imagenet --binarization  det --wd ${wd} --lr ${lr} --lr_decay cos\
  --input_size 64 --epochs 500 -b 256 -j 10 -K $K -L $L --seed $seed -scale $S --gpus $GPU_id \
  > /dev/null 2>&1 &
done
