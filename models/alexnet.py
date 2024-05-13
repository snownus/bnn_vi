import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from .sdp_wo_entropy import BinarizeConv2dSDP

__all__ = ['AlexNet', 'AlexNet_1W32A', 'alexnet_1w1a', 'alexnet_1w32a']

class AlexNet(nn.Module):

    def __init__(self, K, scale, binarize_a=True, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinarizeConv2dSDP(K, scale, 96, 256, kernel_size=5, stride=1, padding=2, bias=True, binarize_a=binarize_a),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinarizeConv2dSDP(K, scale, 256, 384, kernel_size=3, stride=1, padding=1, bias=True, binarize_a=binarize_a),
            nn.BatchNorm2d(384, eps=1e-3, momentum=0.1, affine=True),
            BinarizeConv2dSDP(K, scale, 384, 384, kernel_size=3, stride=1, padding=1, bias=True, binarize_a=binarize_a),
            nn.BatchNorm2d(384, eps=1e-3, momentum=0.1, affine=True),
            BinarizeConv2dSDP(K, scale, 384, 256, kernel_size=3, stride=1, padding=1, bias=True, binarize_a=binarize_a),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            BinarizeConv2dSDP(K, scale, 256 * 6 * 6, 4096, kernel_size=1, padding=0, linear=True, bias=True, binarize_a=binarize_a),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(0.1),
            BinarizeConv2dSDP(K, scale, 4096, 4096, kernel_size=1, padding=0, linear=True, bias=True, binarize_a=binarize_a),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNet_1W32A(nn.Module):

    def __init__(self, K, scale, num_classes=1000):
        super(AlexNet_1W32A, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinarizeConv2dSDP(K, scale, 96, 256, kernel_size=5, stride=1, padding=2, bias=True, binarize_a=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinarizeConv2dSDP(K, scale, 256, 384, kernel_size=3, stride=1, padding=1, bias=True, binarize_a=False),
            nn.BatchNorm2d(384, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            BinarizeConv2dSDP(K, scale, 384, 384, kernel_size=3, stride=1, padding=1, bias=True, binarize_a=False),
            nn.BatchNorm2d(384, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            BinarizeConv2dSDP(K, scale, 384, 256, kernel_size=3, stride=1, padding=1, bias=True, binarize_a=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            BinarizeConv2dSDP(K, scale, 256 * 6 * 6, 4096, kernel_size=1, padding=0, linear=True, bias=True, binarize_a=False),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            BinarizeConv2dSDP(K, scale, 4096, 4096, kernel_size=1, padding=0, linear=True, bias=True, binarize_a=False),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x



def alexnet_1w1a(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model


def alexnet_1w32a(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_1W32A(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])

    return model