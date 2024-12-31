#!/bin/bash

S=100
K=2
L=40

wd=1e-5
lr=0.5

lr_decay=cos

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --master_port 10087 --nproc_per_node 4 main_sdp_imagenet.py -a alexnet_1w1a --dali_cpu \
-save imagenet_alexnet1w1a_benchmark_lr=${lr}_wd=${wd}_K=${K}_S=${S}_L=${L}_lr-decay=${lr_decay}_${binary_opt}_epochs=100  --wd ${wd} --lr ${lr} \
--epochs 100 -b 256 -j 20 -K $K -L $L -scale $S --lr_decay cos /home/gengxue/data/imagenet \
> /dev/null 2>&1 &

