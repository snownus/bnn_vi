#!/bin/bash

model_dir=results/vgg16_cifar100_1w32a_seed=2333_benchmark_K=8_S=100_L=40_wd=5e-4_lr=0.1_cos_epochs=500

python main_sdp_cifar.py --model vgg16_cifar100_sdp   \
--save test/   --dataset cifar100 --binarization  det --wd 5e-4 \
--lr 0.1 --lr_decay cos  --input_size 32 --epochs 500 -b 256 -j 10 \
-K 8 -L 40 --seed 2333 -scale 100 --gpus 0 --evaluate ${model_dir}/model_best.pth.tar