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

## Pre-trained Models

You can download pretrained CIFAR-100 + ResNet18 for binarized weights and real-valued activations here: 

1. VGG16 + CIFAR100 (vgg16_1w32A_model_best.pth.tar):
https://drive.google.com/file/d/1fpCf84TS8UbJa3KB7VgxWjo190_4UPjY/view?usp=drive_link


2. ResNet18 + CIFAR100 (resnet18_1w32A_model_best.pth.tar)
https://drive.google.com/file/d/1bHZ-3HXe4EO7sGDx5XpHSENUdKfcUrV0/view?usp=drive_link


## Evaluation

1. Evaluation with Training.
After completing the training, the log files will be saved in the ```results/``` folder. At the end of each log file, you will see the best precision for the validation dataset.

For example, you could see the best accuracy at the end of the log file:
```results/resnet18_1w32a_cifar100_seed=2020_benchmark_K=8_S=100_L=40_wd=5e-4_lr=0.1_cos_epochs=500.txt.```


2. Direct Evaluation: Download the shared pre-trained model and save it to your local directory. 
    - Evaluate CIFAR-100 with ResNet18, run the command:
    ```bash
    bash evaluate_cifar100_resnet18_1W32A.sh
    ```

    - Evaluate CIFAR-100 with VGG16, run the command:
    ```bash
    bash evaluate_cifar100_vgg16_1W32A.sh
    ```


## Results

### [Image Classification on CIFAR-100 with Binarized Weights and Real-Valued Activations]

Our model achieves the following performance :
| Model name                     |   Accuracy   |
| ------------------------------ |------------- |
| VGG16 + CIFAR-100 (1W32A)      | 72.09 ± 0.17 |
| ResNet18 + CIFAR-100 (1W32A)   | 77.05 ± 0.41 |
