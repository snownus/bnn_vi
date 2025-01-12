#!/bin/bash

model_dir=results/resnet18_1w1a_cifar10_seed=2020_benchmark_K=4_S=100_L=40_wd=1e-5_lr=0.5_cos_epochs=600

python main_sdp_cifar.py --model resnet18_1w1a_cifar  \
--save test/  \
--dataset cifar10 --binarization  det --wd 5e-4 --evaluate ${model_dir}/model_best.pth.tar \
--lr 0.5 --lr_decay cos  --input_size 32 --epochs 600 \
-b 256 -j 10 -K 4 -L 40 --seed 2020 -scale 100 --gpus 0
