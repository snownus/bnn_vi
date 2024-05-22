#!/bin/bash

seeds=(2020 2024 1314 512 2333)
GPU_ids=(0 1 2 3 4)

paste <(printf "%s\n" "${seeds[@]}") <(printf "%s\n" "${GPU_ids[@]}") | while IFS=$'\t' read -r seed GPU_id
do
    echo "Running with seed=$seed and GPU_id=$GPU_id"
    CUDA_VISIBLE_DEVICES=$GPU_id nohup python robustness_main.py --seed $seed > /dev/null 2>&1 &
done