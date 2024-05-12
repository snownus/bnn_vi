#!/bin/bash

# Array of K values
# K_values=(2 4 6 8 10)

S=100
K=4
L=40

wd=1e-4
lr=0.1

lr_decay=cos

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --master_port 10088 --nproc_per_node 4 main_imagenet.py -a resnet18_1w32a_recu --dali_cpu \
-save imagenet_birealnet1w32a_recu_benchmark_lr=${lr}_wd=${wd}_K=${K}_S=${S}_L=${L}_lr-decay=${lr_decay}_${binary_opt}_epochs=200  --wd ${wd} --lr ${lr} \
--epochs 200 -b 256 -j 20 -K $K -L $L -scale $S --lr_decay cos /home/gengxue/data/imagenet \
> /dev/null 2>&1 &
