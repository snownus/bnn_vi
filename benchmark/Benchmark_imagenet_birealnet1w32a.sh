#!/bin/bash

S=100
K=4
L=40

wd=5e-5
lr=0.1

lr_decay=cos
epochs=200

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --master_port 10087 --nproc_per_node 4 ./main_sdp_imagenet.py -a resnet18_1w32a --dali_cpu \
-save imagenet_resnet18_1w32a_benchmark_lr=${lr}_wd=${wd}_K=${K}_S=${S}_L=${L}_lr-decay=${lr_decay}_epochs=${epochs}  --wd ${wd} --lr ${lr} \
--epochs ${epochs} -b 256 -j 20 -K $K -L $L -scale $S --lr_decay cos ${HOME}/data/imagenet \
> /dev/null 2>&1 &
