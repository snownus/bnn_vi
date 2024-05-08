#!/bin/bash

# Array of K values
# K_values=(2 4 6 8 10)

S=100
K=4
L=40

# wd=5e-5
lr=0.1

# Define the seed values
# seeds=(2020 2024 1314 512 2333)
seed=2020
wds=(5e-5 1e-4 1e-5 5e-4)
# wds=(5e-5)
# Define the GPU IDs
# GPU_ids=(0 1 2 3 4)
# GPU_ids=(3 4 5 6 7)
GPU_ids=(0 4 5 6)

# GPU_ids=(0)

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
paste <(printf "%s\n" "${wds[@]}") <(printf "%s\n" "${GPU_ids[@]}") | while IFS=$'\t' read -r wd GPU_id
do
  echo "Running with wd=$wd and GPU_id=$GPU_id"
  CUDA_VISIBLE_DEVICES=$GPU_id nohup python main_sdp1_wo_gaussian_qudrature.py --model resnet18_1w1a\
  --save resnet18_1w1a_cifar10_seed=${seed}_benchmark_K=${K}_S=${S}_L=${L}_wd=${wd}_lr=${lr}_cos_epochs=600\
  --dataset cifar10 --binarization  det --wd ${wd} --lr ${lr} --lr_decay cos\
  --input_size 32 --epochs 600 -b 256 -j 10 -K $K -L $L --seed $seed -scale $S --gpus 0 \
  > /dev/null 2>&1 &
  # Replace the following line with the actual command you want to execute
  # command --seed $seed --gpu $GPU_id
done


# --model resnet18_1w1a --save resnet18_1w1a --dataset cifar10 --binarization det -L 40 --input_size 32 -b 256 -j 20 -K 4 
# --seed 2020 --lr 0.1 --epochs 150 --wd 5e-5 --lr_decay cos -scale 100 --gpus 5