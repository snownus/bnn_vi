#!/bin/bash

# Array of K values
# K_values=(2 4 6 8 10)

S=100
K=4
L=40

wd=1e-5
lr=0.5

lr_decay=cos

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --master_port 10088 --nproc_per_node 4 main_imagenet.py -a birealnet18 --dali_cpu \
-save imagenet_birealnet18_benchmark_lr=${lr}_wd=${wd}_K=${K}_S=${S}_L=${L}_lr-decay=${lr_decay} --wd ${wd} --lr ${lr} \
--epochs 150 -b 256 -j 20 -K $K -L $L -scale $S --lr_decay cos /home/gengxue/data/imagenet \
> /dev/null 2>&1 &