#!/bin/bash

model_dir=results/resnet18_1w32a_cifar100_seed=512_benchmark_K=8_S=100_L=40_wd=5e-4_lr=0.1_cos_epochs=500

python main_sdp_cifar.py --model resnet18_1w32a_cifar  \
--save test/  \
--dataset cifar100 --binarization  det --wd 5e-4 --evaluate ${model_dir}/model_best.pth.tar \
--lr 0.1 --lr_decay cos  --input_size 32 --epochs 500 \
-b 256 -j 10 -K 8 -L 40 --seed 512 -scale 100 --gpus 0
