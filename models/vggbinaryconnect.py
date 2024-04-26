import torch
import torch.nn as nn
import torch.nn.functional as F
from .sdp_wo_entropy import BinarizeConv2dSDP



class VGGBinaryConnect_SDP(nn.Module):
    """VGG-like net used for Cifar10.
       This model is the Conv architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, K, scale, in_features, out_features, eps=1e-5, momentum=0.2, batch_affine=False):
        super(VGGBinaryConnect_SDP, self).__init__()
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_features, 128, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)
        # self.bn1 = nn.BatchNorm2d(128)

        # self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=False)
        self.conv2 = BinarizeConv2dSDP(K, scale, 128, 128, kernel_size=3, padding=1,bias=False, binarize_a=False)
        self.bn2 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)
        # self.bn2 = nn.BatchNorm2d(128)

        #self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=False)
        self.conv3 = BinarizeConv2dSDP(K, scale, 128, 256, kernel_size=3, padding=1,bias=False, binarize_a=False)
        self.bn3 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)
        # self.bn3 = nn.BatchNorm2d(256)

        #self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False)
        self.conv4 = BinarizeConv2dSDP(K, scale, 256, 256, kernel_size=3, padding=1,bias=False, binarize_a=False)
        self.bn4 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)
        # self.bn4 = nn.BatchNorm2d(256)


        #self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=False)
        self.conv5 = BinarizeConv2dSDP(K, scale, 256, 512, kernel_size=3, padding=1,bias=False, binarize_a=False)
        self.bn5 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        # self.bn5 = nn.BatchNorm2d(512)

        #self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.conv6 = BinarizeConv2dSDP(K, scale, 512, 512, kernel_size=3, padding=1,bias=False, binarize_a=False)
        self.bn6 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        # self.bn6 = nn.BatchNorm2d(512)


        #self.fc1 = nn.Linear(512 * 4 * 4, 1024, bias=False)
        self.fc1 = BinarizeConv2dSDP(K, scale, 512 * 4 * 4, 1024, kernel_size=1, padding=0, linear=True, bias=False, binarize_a=False)
        self.bn7 = nn.BatchNorm1d(1024,affine=batch_affine)
        # self.bn7 = nn.BatchNorm1d(1024)

        #self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.fc2 = BinarizeConv2dSDP(K, scale, 1024, 1024, kernel_size=1, padding=0, linear=True, bias=False, binarize_a=False)
        self.bn8 = nn.BatchNorm1d(1024, affine=batch_affine)
        # self.bn8 = nn.BatchNorm1d(1024)


        self.fc3 = nn.Linear(1024, out_features, bias=False)
        self.bn9 = nn.BatchNorm1d(out_features,affine=batch_affine)
        # self.bn9 = nn.BatchNorm1d(out_features)



    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(x))

        x = F.relu(self.bn3(self.conv3(x)))


        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn4(x))

        x = F.relu(self.bn5(self.conv5(x)))

        x = self.conv6(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn6(x))

        x = x.view(-1, 512 * 4 * 4)

        x = self.fc1(x)
        x = F.relu(self.bn7(x))

        x = self.fc2(x)
        x = F.relu(self.bn8(x))

        x = self.fc3(x)
        x = self.bn9(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),


def vggbinaryconnect_cifar10(**kwargs):
    input_size = kwargs.get('input_size', 32)
    num_classes = kwargs.get( 'num_classes', 10)
    K = kwargs.get( 'K', 2)
    scale = kwargs.get('scale', 100)
    return VGGBinaryConnect_SDP(K, scale, 3, num_classes)


def vggbinaryconnect_cifar100(**kwargs):
    input_size = kwargs.get('input_size', 32)
    num_classes = kwargs.get( 'num_classes', 100)
    K = kwargs.get( 'K', 2)
    scale = kwargs.get('scale', 100)
    return VGGBinaryConnect_SDP(K, scale, 3, num_classes)

