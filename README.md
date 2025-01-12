This repository shows our NIPS paper "Training Binary Neural Networks via Gaussian Variational Inference and Low-Rank Semidefinite Programming".

## Requirements

### OS Environment
- **Linux**: Ubuntu 20.04.1 LTS

### CUDA Version
- **CUDA**: Version 12.0

### Python Environment

1. **Install Python Virtual Environment**:
    ```bash
    apt-get update
    apt-get install -y --no-install-recommends build-essential curl libfreetype6-dev libzmq3-dev pkg-config python3-pip software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt update
    apt-get install -y python3-venv
    python3.8 -m venv py3.8

2. **Install Requirements**:
    ```bash
    source py3.8/bin/activate
    pip install -r requirements.txt
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    ```


## Dataset Preparation

1. **Download CIFAR Dataset**:
    - Download CIFAR-10
        ```bash
        python main_sdp_cifar.py --model vgg16_cifar10_sdp --download_data --save test/ --dataset cifar10 --input_size 32 --gpus 0
        ```

    - Download CIFAR-100
        ```bash
        python main_sdp_cifar.py --model vgg16_cifar100_sdp --download_data --save test/ --dataset cifar100 --input_size 32 --gpus 0
        ```
    
2. **Download tiny-imagenet**
        
        i. download http://cs231n.stanford.edu/tiny-imagenet-200.zip and save it to folder ```datasets/```

        ii. run ```python datasets/process_tiny_imagenet.py```
    
3. **Download imagenet**


## Training

We provide all codes shown in our paper.

1. **Benchmarking**

All training code are shown in the ``benchmark/`` and it is easy to figure out the run settings from the name of the bash file. All hyper-parameter settings and running procedures are included in the respective bash files. Here is an example.

    - Run Training with 1W1A settings on CIFAR-10:
    ```bash
    bash benchmark/Benchmark_cifar10_resnet18_1w1a.sh
    ```

## Results and Pre-trained Models

You can download pretrained CIFAR-100 + ResNet18 for binarized weights and real-valued activations here: 

| Model name                     |   Accuracy   |    Pretrained Model  |
| ------------------------------ |------------- | ---------------------|
| VGG16 + CIFAR-100 (1W32A)      | 72.09 ± 0.17 |  [Click Here](https://drive.google.com/file/d/1G2p3RIQCE5UQi0dIAU-DsOkVSt3brLiC/view?usp=sharing) |
| ResNet18 + CIFAR-100 (1W32A)   | 77.05 ± 0.41 |  [Click Here](https://drive.google.com/file/d/1DuL1WkIzWOBciliyWClomcdq7r5Ldzn4/view?usp=sharing)|
| VGG16 + CIFAR-10 (1W32A)       | 93.25 ± 0.11 |  [Click Here](https://drive.google.com/file/d/1M4dFejHJU5jQDtF3bP_k_KwijvIy_n7I/view?usp=sharing) |
| ResNet18 + CIFAR-10 (1W32A)    | 95.05 ± 0.10 |  [Click Here](https://drive.google.com/file/d/1BasTsjZgWrFnaLrv-d__KqoBimUqUnxy/view?usp=sharing) |
| ResNet18 + Tiny-ImageNet (1W32A) |  58.98 ± 0.28 | [Click Here](https://drive.google.com/file/d/1CzwoAXbUOvQ4ZQcrt5BFaoB_etgVNDAZ/view?usp=sharing) |
| -------------------------------- | -------------- | ------------------ |
| VGG-Small + CIFAR-10 (1W1A) | 92.7 ± 0.1 | [Click Here](https://drive.google.com/file/d/1Jz8qlvQp1uHJCWHvhkjwkA6aVLods5XT/view?usp=sharing) |
| ResNet18 + CIFAR-10 (1W1A)  | 92.8 ± 0.2 | [Click Here](https://drive.google.com/file/d/15dq4ucMZ1VeOAU9JhVzkc7bFvFTXbggA/view?usp=sharing) |
|------------------------------------| -------------- | ------------------ |
| AlexNet + ImageNet (1W1A) | 51.1 | [Click Here](https://drive.google.com/file/d/1GQhWWdwbQk8TPNxB22Uou5dpYePVw-r7/view?usp=sharing) |
| AlexNet + ImageNet (1W32A) | 59.4 | [Click Here](https://drive.google.com/file/d/1H0OB7_X6c6tNha5eMQYHEaMkwlj3Bt_s/view?usp=sharing) |
|------------------------------------| -------------- | -------------------- |
| BiRealNet + ImageNet (1W1A) | 62.1 | [Click Here](https://drive.google.com/file/d/1ew9nP-M_GwfQPSXjWkZUts_cyIQN2yib/view?usp=sharing) | 
| ResNet18 + ImageNet (1W32A) | 68.2 | [Click Here](https://drive.google.com/file/d/1WAC18mpcQx5DHzMgJgRZ-InL1oMLcTXe/view?usp=sharing) | 


## Evaluation

1. Evaluation with Training.
After completing the training, the log files will be saved in the ```results/``` folder. At the end of each log file, you will see the best precision for the validation dataset.

For example, you could see the best accuracy at the end of the log file:
```results/resnet18_1w32a_cifar100_seed=2020_benchmark_K=8_S=100_L=40_wd=5e-4_lr=0.1_cos_epochs=500.txt.```


2. Direct Evaluation: Download the shared pre-trained model and save it to your local directory. Folder ```evaluation``` contains all bash file to evaluate pretrained models. Please check the model path and data path correctly.
    - Evaluate CIFAR-100 with ResNet18, run the command:
    ```bash
    bash evaluation/evaluate_cifar100_resnet18_1W32A.sh
    ```

    - Evaluate CIFAR-100 with VGG16, run the command:
    ```bash
    bash evaluation/evaluate_cifar100_vgg16_1W32A.sh
    ```