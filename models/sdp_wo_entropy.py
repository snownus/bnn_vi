import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np
from utils import binarize

import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn import init


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        
        out_forward = torch.sign(x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3
        '''
        mask1 = x < -1
        mask2 = x > 1
        out1 = (-1) * mask1.type(torch.float32) + (x) * (1-mask1.type(torch.float32))
        out2 = (1) * mask2.type(torch.float32) + (x) * (1-mask2.type(torch.float32))
        out = out_forward.detach() - out2.detach() + out2
        '''

        return out


class BinarizeConv2dSDP(nn.Module):

    def __init__(self, K, scale, in_chn, out_chn, dropout=0, kernel_size=3, stride=1, padding=1, bias=False, linear=False):
        super(BinarizeConv2dSDP, self).__init__()
        self.alpha = nn.Parameter(torch.rand(out_chn, 1, 1), requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape_w = (K, out_chn, in_chn, kernel_size, kernel_size)
        self.shape_sum_w = (out_chn, in_chn, kernel_size, kernel_size)
        # self.weights = nn.Parameter(torch.zeros((self.shape_sum_w)), requires_grad=True)
        # self.weights = nn.Parameter(torch.zeros((1, self.number_of_weights)), requires_grad=True)
        
        self.K = K
        self.scale = scale

        self.M = nn.Parameter(torch.zeros((self.shape_sum_w)), requires_grad=True)
        self.Z = nn.Parameter(torch.zeros((self.shape_w)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.M.data)
        # torch.nn.init.xavier_uniform_(self.Z.data)
        torch.nn.init.xavier_normal_(self.Z.data)

        #torch.nn.init.xavier_normal_(self.weights.data)
        #self.weights.data = torch.normal(0.0, 1.0,size=self.shape_sum_w).cuda() 
        self.linear = linear
        self.in_chn = in_chn
        self.out_chn = out_chn
        
        # get the i^th column of Z.
        self.sample = nn.Parameter(torch.zeros((1, K)), requires_grad=False)
        # self.register_buffer('sample', torch.zeros((1, K), requires_grad=False))

    def forward(self, input):
        # bw = BinaryQuantize().apply(self.weights)

        # real_weights = self.weights.view(self.shape_sum_w)
        # bw = BinaryQuantize().apply(real_weights)
        
        m = self.M.view(self.number_of_weights)
        z = self.Z.view(self.K, self.number_of_weights)

        with torch.no_grad():
            A = m*m + torch.sum(z.T**2,dim=1)/self.scale
            # print(f'A.shape: {A.shape}, self.M.shape: {self.M.shape}')
            m.data = m.data / torch.sqrt(A)
            z.data = z.data / torch.sqrt(A)

        # w = sdp(m, z, torch.tensor(self.scale), torch.tensor(self.K))
        w = SDP().apply(m, z, self.sample, torch.tensor(self.scale), torch.tensor(self.K), self.training)
        real_weights = w.view(self.shape_sum_w)
        bw = BinaryQuantize().apply(real_weights)

        a = input
        if self.linear:
            a = a[:, :, None, None]
        ba = BinaryQuantize_a().apply(a)
        output = F.conv2d(ba, bw, stride=self.stride, padding=self.padding, bias=None)
        #* scaling factor
        output = output * self.alpha
        if self.linear:
            output = torch.squeeze(output)

        return output


class SDP(Function):
    @staticmethod
    def forward(ctx, m, z, sample, scale, K, training=True):
        sample_sum = torch.sum(sample)
        rv = torch.normal(0.0, 1.0/np.sqrt(scale.item()), size=(1, K.item())).cuda()
        if sample_sum == 1 or sample_sum == -1:
            rv = sample
        elif sample_sum != 0:
            print(f'errors in sample Z!!!')
        zz = torch.mm(rv, z)
        w = zz + m
        # if training is True:
        #     w = zz + m
        # else:
        #     w = m
        grad_scaling = np.sqrt(scale)
        if sample_sum == 1 or sample_sum == -1:
            grad_scaling = torch.tensor(1.0).cuda()
        ctx.save_for_backward(K, sample, rv, grad_scaling)
        
        return w

    @staticmethod
    def backward(ctx, grad_output):
        # print(f'aaa') two positive gradients -> 12.61; 
        K, sample, rv, grad_scaling = ctx.saved_tensors
        grad_input1, grad_input2 = grad_output.clone(), grad_output.clone()
        grad_m = grad_input1
        # print(f'grad_m: {grad_m}')
        grad_z = torch.mm(rv.T, grad_input2) * grad_scaling
        # print(f'grad_z: {grad_z}')
        return grad_m, grad_z, None, None, None, None


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()

        return grad_input