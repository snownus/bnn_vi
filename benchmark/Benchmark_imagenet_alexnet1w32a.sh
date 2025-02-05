#!/bin/bash

S=100
K=4
L=40

lr_decay=cos

wd=5e-5
lr=0.1

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --master_port 10088 --nproc_per_node 4 ./main_sdp_imagenet.py \
-a alexnet_1w32a --dali_cpu \
-save imagenet_alexnet1w32a_benchmark_lr=${lr}_wd=${wd}_K=${K}_S=${S}_L=${L}_lr-decay=${lr_decay}_epochs=100  --wd ${wd} --lr ${lr} \
--epochs 100 -b 256 -j 20 -K $K -L $L -scale $S --lr_decay cos ${HOME}/data/imagenet \
> /dev/null 2>&1 &