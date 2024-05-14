import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import sys

from .sdp_wo_entropy import BinarizeConv2dSDP

BN = None

__all__ = ['ResNet', 'resnet18_1w32a', 'resnet34_1w32a']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def conv3x3Binary(K, scale, in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2dSDP(K, scale, in_planes, out_planes, kernel_size=3, 
                             stride=stride, padding=1, binarize_a=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, K, scale, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3Binary(K, scale, inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.nonlinear = nn.ReLU(inplace=True)
        self.conv2 = conv3x3Binary(K, scale, planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinear(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.nonlinear(out)    

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, K, scale, num_classes=1000, deep_stem=False,
                 avg_down=False, bypass_last_bn=False,
                 bn_group_size=1,
                 bn_group=None,
                 bn_sync_stats=False,
                 use_sync_bn=True):

        global BN, bypass_bn_weight_list

        BN = nn.BatchNorm2d

        bypass_bn_weight_list = []


        self.inplanes = 64
        super(ResNet, self).__init__()

        self.deep_stem = deep_stem
        self.avg_down = avg_down

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        self.nonlinear1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(K, scale, block, 64, layers[0])
        self.layer2 = self._make_layer(K, scale, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(K, scale, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(K, scale, block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.nonlinear2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            print('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def _make_layer(self, K, scale, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BN(planes * block.expansion),
                )

        layers = []
        layers.append(block(K, scale, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(K, scale, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)  
        x = self.maxpool(x) 

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)   
        x = self.fc(x)          
        
        return x


def resnet18_1w32a(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_1w32a(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model