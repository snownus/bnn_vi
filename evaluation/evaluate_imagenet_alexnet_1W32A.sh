S=100
K=4
L=40

lr_decay=cos

wd=5e-5
lr=0.1

model_dir=results/imagenet_alexnet1w32a_benchmark_lr=0.1_wd=5e-5_K=4_S=100_L=40_lr-decay=cos_epochs=100

CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --master_port 10087 --nproc_per_node 4  main_sdp_imagenet.py \
-a alexnet_1w32a --dali_cpu --wd ${wd} --lr ${lr} \
--epochs 100 -b 256 -j 20 -K $K -L $L -scale $S --lr_decay cos \
--evaluate ${model_dir}/model_best.pth.tar \
${HOME}/data/imagenet 