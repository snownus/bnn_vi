import torch.nn as nn

from .sdp_wo_entropy import BinarizeConv2dSDP

import torch.nn.functional as F


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class VGG16(nn.Module):
    """VGG16 used for Cifar10.
       This model is the Conv architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, K, scale, in_features, out_features, eps=1e-5, momentum=0.2, batch_affine=False):
        super(VGG16, self).__init__()
        self.in_features = in_features
        # self.conv1 = nn.Conv2d(in_features, 64, kernel_size=3, padding=1,bias=False)
        self.conv1 = BinarizeConv2dSDP(K, scale, in_features, 64, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn1 = nn.BatchNorm2d(64, eps=eps, momentum=momentum,affine=batch_affine)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1,bias=False)
        self.conv2 = BinarizeConv2dSDP(K, scale, 64, 64, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn2 = nn.BatchNorm2d(64, eps=eps, momentum=momentum,affine=batch_affine)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=False)
        self.conv3 = BinarizeConv2dSDP(K, scale, 64, 128, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn3 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=False)
        self.conv4 = BinarizeConv2dSDP(K, scale, 128, 128, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn4 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)


        # self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=False)
        self.conv5 = BinarizeConv2dSDP(K, scale, 128, 256, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn5 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)
        # self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False)
        self.conv6 = BinarizeConv2dSDP(K, scale, 256, 256, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn6 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)
        # self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False)
        self.conv7 = BinarizeConv2dSDP(K, scale, 256, 256, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn7 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)

        # self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=False)
        self.conv8 = BinarizeConv2dSDP(K, scale, 256, 512, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn8 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        # self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.conv9 = BinarizeConv2dSDP(K, scale, 512, 512, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn9 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        # self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.conv10 = BinarizeConv2dSDP(K, scale, 512, 512, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn10 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)


        # self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.conv11 = BinarizeConv2dSDP(K, scale, 512, 512, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn11 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        # self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.conv12 = BinarizeConv2dSDP(K, scale, 512, 512, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn12 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        # self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.conv13 = BinarizeConv2dSDP(K, scale, 512, 512, kernel_size=3, padding=1, 
                                       bias=False, binarize_a=False, binarize_out=False)
        self.bn13 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)

        # self.fc14 = nn.Linear(512, 512, bias=False)
        self.fc14 = BinarizeConv2dSDP(K, scale, 512, 512, kernel_size=1, padding=0, linear=True, 
                                      bias=False, binarize_a=False, binarize_out=False)
        self.bn14 = nn.BatchNorm1d(512,affine=batch_affine)

        # self.fc15 = nn.Linear(512, 512, bias=False)
        self.fc15 = BinarizeConv2dSDP(K, scale, 512, 512, kernel_size=1, padding=0, linear=True, 
                                      bias=False, binarize_a=False, binarize_out=False)
        self.bn15 = nn.BatchNorm1d(512,affine=batch_affine)

        # self.fc16 = nn.Linear(512, out_features, bias=False)
        self.fc16 = BinarizeConv2dSDP(K, scale, 512, out_features, kernel_size=1, padding=0, linear=True, 
                                      bias=False, binarize_a=False, binarize_out=False)
        self.bn16 = nn.BatchNorm1d(out_features,affine=batch_affine)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # layer 1: outsize 64
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2, 2))) # layer 2: outsize 128

        x = F.relu(self.bn3(self.conv3(x))) # layer 3: outsize 128
        x = F.relu(self.bn4(F.max_pool2d(self.conv4(x), 2, 2))) # layer 4: outsize 256

        x = F.relu(self.bn5(self.conv5(x))) # layer 5: outsize 256
        x = F.relu(self.bn6(self.conv6(x))) # layer 6: outsize 256
        x = F.relu(self.bn7(F.max_pool2d(self.conv7(x), 2, 2))) # layer 7: outsize 512

        x = F.relu(self.bn8(self.conv8(x))) # layer 8: outsize 512
        x = F.relu(self.bn9(self.conv9(x))) # layer 9: outsize 512
        x = F.relu(self.bn10(F.max_pool2d(self.conv10(x), 2, 2))) # layer 10: outsize 512

        x = F.relu(self.bn11(self.conv11(x))) # layer 8: outsize 512
        x = F.relu(self.bn12(self.conv12(x))) # layer 9: outsize 512
        x = F.relu(self.bn13(F.max_pool2d(self.conv13(x), 2, 2))) # layer 10: outsize 512
        
        x = x.view(-1, 512)

        x = self.fc14(x)
        x = F.relu(self.bn14(x))

        x = self.fc15(x)
        x = F.relu(self.bn15(x))

        x = self.fc16(x)
        x = self.bn16(x)
        
        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),



def vgg16_cifar10_sdp(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    K = kwargs.get( 'K', 2)
    scale = kwargs.get('scale', 100)
    return VGG16(K, scale, 3, out_features=num_classes)


def vgg16_cifar100_sdp(**kwargs):
    num_classes = kwargs.get( 'num_classes', 100)
    K = kwargs.get( 'K', 2)
    scale = kwargs.get('scale', 100)
    return VGG16(K, scale, 3, out_features=num_classes)