#!/bin/bash

S=100
K=2
L=40

wd=1e-5
lr=0.5

lr_decay=cos
epochs=200

model_dir=results/imagenet_resnet18_1w1a_recu_benchmark_lr=0.5_wd=1e-5_K=2_S=100_L=40_lr-decay=cos_epochs=200

CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --master_port 10087 --nproc_per_node 4  main_sdp_imagenet.py \
-a resnet18_1w1a_recu --dali_cpu --wd ${wd} --lr ${lr} \
--epochs 100 -b 256 -j 20 -K $K -L $L -scale $S --lr_decay cos \
--evaluate ${model_dir}/model_best.pth.tar \
${HOME}/data/imagenet 