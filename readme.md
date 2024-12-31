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

## Training

We provide the code for training and evaluating our methods on the commonly used network architectures ResNet18 and VGG16 for CIFAR-100 datasets with binarized weights and real-valued activations (denoted as 1W32A).

1. **Download CIFAR-100 Dataset**:
    ```bash
    bash download_data.sh
    ```

2. **Run Training on VGG16 + CIFAR-100 (1W32A) with 5 Runs**:
    ```bash
    bash benchmark_cifar100_vgg16_1W32A.sh
    ```
    Note: This script runs 5 instances in parallel on 5 different GPUs. You can specify your own GPU IDs.

3. **Run Training on ResNet18 + CIFAR-100 (1W32A) with 5 Runs**:
    ```bash
    bash benchmark_cifar100_resnet18_1W32A.sh
    ```
    Note: This script runs 5 instances in parallel on 5 different GPUs. You can specify your own GPU IDs.

All hyper-parameter settings and running procedures are included in the respective bash files.


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
