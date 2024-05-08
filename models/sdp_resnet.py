'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .sdp_wo_entropy import BinarizeConv2dSDP


__all__ =['resnet18_1w1a', 'resnet18_1w32a']

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_1w32a(nn.Module):
    expansion = 1

    def __init__(self, K, scale, in_planes, planes, stride=1):
        super(BasicBlock_1w32a, self).__init__()
        self.conv1 = BinarizeConv2dSDP(K, scale, in_planes, planes, kernel_size=3, stride=stride, padding=1, binarize_a=False, binarize_out=False, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2dSDP(K, scale, planes, planes, kernel_size=3, stride=1, padding=1, binarize_a=False, binarize_out=False, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        pad = 0 if planes == self.expansion*in_planes else planes // 4
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion*planes, affine=False, eps=1e-5, momentum=0.2)
                        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out



class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, K, scale, in_planes, planes, stride=1):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = BinarizeConv2dSDP(K, scale, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2dSDP(K, scale, planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        pad = 0 if planes == self.expansion*in_planes else planes // 4
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                        nn.AvgPool2d((2,2)), 
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.hardtanh(out, inplace=True)
        return out


class Bottleneck_1w1a(nn.Module):
    expansion = 4

    def __init__(self, K, scale, in_planes, planes, stride=1):
        super(Bottleneck_1w1a, self).__init__()
        self.conv1 = BinarizeConv2dSDP(K, scale, in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2dSDP(K, scale, planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2dSDP(K, scale, planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2dSDP(K, scale, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = F.hardtanh(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out


class ResNet(nn.Module):
    def __init__(self, K, scale, block, num_blocks, num_channel, binarize_a=True, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channel[0]
        self.binarize_a = binarize_a

        self.conv1 = nn.Conv2d(3, num_channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel[0])
        self.layer1 = self._make_layer(K, scale, block, num_channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(K, scale, block, num_channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(K, scale, block, num_channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(K, scale, block, num_channel[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(num_channel[3]*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_channel[3]*block.expansion)

        self.apply(_weights_init)

    def _make_layer(self, K, scale, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(K, scale, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.binarize_a:
            out = self.bn1(self.conv1(x))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out 


def resnet18_1w1a(**kwargs):
    K = kwargs.get( 'K', 2)
    scale = kwargs.get('scale', 100)
    dataset = kwargs.get('dataset', 'cifar10')
    num_classes = -1
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    return ResNet(K, scale, BasicBlock_1w1a, [2,2,2,2], [64,128,256,512], num_classes=num_classes)


def resnet18_1w32a(**kwargs):
    K = kwargs.get( 'K', 2)
    scale = kwargs.get('scale', 100)
    dataset = kwargs.get('dataset', 'cifar10')
    num_classes = -1
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    return ResNet(K, scale, BasicBlock_1w32a, [2,2,2,2], [64,128,256,512], num_classes=num_classes)


def resnet34_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [3,4,6,3],[64,128,256,512],**kwargs)

def resnet50_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3,4,6,3],[64,128,256,512],**kwargs)

def resnet101_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3,4,23,3],[64,128,256,512],**kwargs)

def resnet152_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3,8,36,3],[64,128,256,512],**kwargs)
