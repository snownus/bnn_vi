import torch.nn as nn

from .sdp_wo_entropy import BinarizeConv2dSDP


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BinaryNet(nn.Module):

    def __init__(self, K, scale, num_classes=100):
        super(BinaryNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear0 = nn.PReLU(128)
        self.conv1 = BinarizeConv2dSDP(K, scale, 128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear1 = nn.PReLU(128)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = BinarizeConv2dSDP(K, scale, 128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.nonlinear2 = nn.PReLU(256)
        self.conv3 = BinarizeConv2dSDP(K, scale, 256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.nonlinear3 = nn.PReLU(256)
        self.conv4 = BinarizeConv2dSDP(K, scale, 256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.nonlinear4 = nn.PReLU(512)
        self.conv5 = BinarizeConv2dSDP(K, scale, 512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.nonlinear5 = nn.PReLU(512)
        self.fc = nn.Linear(512*4*4, num_classes)


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear0(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear4(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def vgg_small_cifar10_sdp(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    K = kwargs.get( 'K', 2)
    scale = kwargs.get('scale', 100)
    return BinaryNet(K, scale, num_classes)


def vgg_small_cifar100_sdp(**kwargs):
    num_classes = kwargs.get( 'num_classes', 100)
    K = kwargs.get( 'K', 2)
    scale = kwargs.get('scale', 100)
    return BinaryNet(K, scale, num_classes)