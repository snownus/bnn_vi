import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init

from .sdp_wo_entropy import BinarizeConv2dSDP

# from .sdp_wo_z import BinarizeConv2dSDP

BN = None

__all__ = ['resnet18_1w1a_recu', 'resnet18_1w32a_recu', 'resnet34_1w1a_recu', 'resnet34_1w32a_recu']


def conv3x3Binary(K, scale, binarize_a, in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    # return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=1, bias=False)
    return BinarizeConv2dSDP(K, scale, in_planes, out_planes, kernel_size=3, 
                             stride=stride, padding=1, binarize_a=binarize_a)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, K, scale, binarize_a, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3Binary(K, scale, binarize_a, inplanes, planes, stride)
        self.bn1 = BN(planes)
        if binarize_a:
            # self.nonlinear1 = nn.Hardtanh(inplace=True)
            self.nonlinear1 = nn.PReLU()
        else:
            self.nonlinear1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3Binary(K, scale, binarize_a, planes, planes)
        self.bn2 = BN(planes)
        if binarize_a:
            self.nonlinear2 = nn.PReLU()
            # self.nonlinear2 = nn.Hardtanh(inplace=True)
        else:
            self.nonlinear2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.nonlinear1(out)
        residual = out
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.nonlinear2(out)
        
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, K, scale, binarize_a, num_classes=1000, 
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

        self.avg_down = avg_down
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        if binarize_a:
            self.nonlinear = nn.PReLU()
            # self.nonlinear = nn.Hardtanh(inplace=True)
        else:
            self.nonlinear = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(K, scale, binarize_a, block, 64, layers[0])
        self.layer2 = self._make_layer(K, scale, binarize_a, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(K, scale, binarize_a, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(K, scale, binarize_a, block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1e-8)
                m.bias.data.zero_()

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            print('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def _make_layer(self, K, scale, binarize_a, block, planes, blocks, stride=1, avg_down=False):
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
        layers.append(block(K, scale, binarize_a, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(K, scale, binarize_a, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x) 
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


def resnet18_1w1a_recu(**kwargs):
    """Constructs a ResNet-18 model. """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs, binarize_a=True)
    return model


def resnet18_1w32a_recu(**kwargs):
    """Constructs a ResNet-18 model. """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs, binarize_a=False)
    return model


def resnet34_1w1a_recu(**kwargs):
    """Constructs a ResNet-34 model. """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs, binarize_a=True)
    return model


def resnet34_1w32a_recu(**kwargs):
    """Constructs a ResNet-34 model. """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs, binarize_a=False)
    return model
