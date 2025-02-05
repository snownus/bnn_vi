#!/bin/bash

S=100
K=2
L=40

wd=1e-5
lr=0.5

lr_decay=cos
epochs=200

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --master_port 10088 --nproc_per_node 4 ./main_sdp_imagenet.py -a resnet18_1w1a_recu --dali_cpu \
-save imagenet_resnet18_1w1a_recu_benchmark_lr=${lr}_wd=${wd}_K=${K}_S=${S}_L=${L}_lr-decay=${lr_decay}_epochs=${epochs}  --wd ${wd} --lr ${lr} \
--epochs ${epochs} -b 256 -j 20 -K $K -L $L -scale $S --lr_decay cos ${HOME}/data/imagenet \
> /dev/null 2>&1 &
