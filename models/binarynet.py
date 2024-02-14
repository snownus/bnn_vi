import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
# from .binarized_modules import  BinarizeLinear,BinarizeConv2d
import math

from .recu_modules import BinarizeConv2d


def init_model(model):
    for m in model.modules():
        # if isinstance(m, BinarizeConv2d):
        #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     m.weight.data.normal_(0, math.sqrt(2. / n))
        # elif isinstance(m, nn.BatchNorm2d):
        #     m.weight.data.fill_(1)
        #     m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class BinaryNet(nn.Module):

    def __init__(self, K, input_size=32, num_classes=100):
        super(BinaryNet, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,bias=True),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(K, 128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(K, 128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(K, 256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(K, 256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(K, 512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            # BinarizeLinear(512 * int(input_size/8) * int(input_size/8), 1024, bias=False),
            BinarizeConv2d(K, 512 * int(self.input_size/8) * int(self.input_size/8), 1024, kernel_size=1, padding=0, linear=True, bias=False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            # nn.Dropout(0.5),
            # BinarizeLinear(1024, 1024, bias=False),
            BinarizeConv2d(K, 1024, 1024, kernel_size=1, padding=0, linear=True, bias=False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )
        init_model(self)
        # self.regime = {
        #     0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
        #     # 40: {'lr': 1e-3},
        #     # 80: {'lr': 5e-4},
        #     # 100: {'lr': 1e-4},
        #     # 120: {'lr': 5e-5},
        #     # 140: {'lr': 1e-5}
        # }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * int(self.input_size/8) * int(self.input_size/8))
        x = self.classifier(x)
        return x


def vgg_cifar10_binary(**kwargs):
    input_size = kwargs.get('input_size', 32)
    num_classes = kwargs.get( 'num_classes', 10)
    K = kwargs.get( 'K', 2)
    return BinaryNet(K, input_size, num_classes)


def vgg_cifar100_binary(**kwargs):
    input_size = kwargs.get('input_size', 32)
    num_classes = kwargs.get( 'num_classes', 100)
    K = kwargs.get( 'K', 2)
    return BinaryNet(K, input_size, num_classes)

def vgg_tiny_imagenet_binary(**kwargs):
    input_size = kwargs.get('input_size', 64)
    num_classes = kwargs.get( 'num_classes', 200)
    K = kwargs.get( 'K', 2)
    return BinaryNet(K, input_size, num_classes)
