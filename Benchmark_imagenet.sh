#!/bin/bash

# Array of K values
# K_values=(2 4 6 8 10)

S=100
K=4
L=40

wd=5e-5
lr=0.5

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node 4 main_imagenet.py -a alexnet --dali_cpu \
-save imagenet_alexnet_benchmark_lr=${lr}_wd=${wd}_K=${K}_S=${S}_L=${L}_milestones_30,60,90 --wd ${wd} --lr ${lr} \
--epochs 100 -b 256 -j 20 -K $K -L $L -scale $S /home/gengxue/data/imagenet \
> /dev/null 2>&1 &