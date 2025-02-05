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

        return out


class BinarizeConv2dSDP(nn.Module):

    def __init__(self, K, scale, in_chn, out_chn, kernel_size=3, 
                 stride=1, padding=1, bias=False, linear=False, 
                 binarize_a=True, binarize_out=False):
        #TODO: BIAS, scaling factor; and dropout;
        super(BinarizeConv2dSDP, self).__init__()
        self.Alpha = nn.Parameter(torch.rand(out_chn, 1, 1), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.rand(out_chn), requires_grad=True)
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape_w = (K, out_chn, in_chn, kernel_size, kernel_size)
        self.shape_sum_w = (out_chn, in_chn, kernel_size, kernel_size)
        
        self.K = K
        self.scale = scale

        self.M = nn.Parameter(torch.zeros((self.shape_sum_w)), requires_grad=True)
        self.Z = nn.Parameter(torch.zeros((self.shape_w)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.M.data)
        torch.nn.init.xavier_normal_(self.Z.data)

        self.linear = linear
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.binarize_a = binarize_a
        self.binarize_out = binarize_out
        
    def forward(self, input):
        m = self.M.view(self.number_of_weights)
        z = self.Z.view(self.K, self.number_of_weights)

        with torch.no_grad():
            A = m*m + torch.sum(z.T**2,dim=1)/self.scale
            m.data = m.data / torch.sqrt(A)
            z.data = z.data / torch.sqrt(A)

        w = SDP().apply(m, z, torch.tensor(self.scale), torch.tensor(self.K), self.training)
        real_weights = w.view(self.shape_sum_w)
        bw = BinaryQuantize().apply(real_weights)

        a = input
        if self.binarize_a:
            ba = BinaryQuantize_a().apply(a)
        else:
            ba = a
        
        alpha = self.Alpha
        if self.linear:
            bw = torch.squeeze(bw)
            output = F.linear(ba, bw, bias=self.bias)
            alpha = torch.squeeze(alpha)
        else:
            output = F.conv2d(ba, bw, stride=self.stride, padding=self.padding, bias=self.bias)
            
        output = output * alpha

        out_a = output
        if self.binarize_out:
            out_a = BinaryQuantize_a().apply(output)

        return out_a


class SDP(Function):
    @staticmethod
    def forward(ctx, m, z, scale, K, training=True):
        rv = torch.normal(0.0, 1.0/np.sqrt(scale.item()), size=(1, K.item())).cuda()
        zz = torch.mm(rv, z)
        w = zz + m
        grad_scaling = np.sqrt(scale)
        ctx.save_for_backward(K, rv, grad_scaling)
        
        return w

    @staticmethod
    def backward(ctx, grad_output):
        K, rv, grad_scaling = ctx.saved_tensors
        grad_input1, grad_input2 = grad_output.clone(), grad_output.clone()
        grad_m = grad_input1
        grad_z = torch.mm(rv.T, grad_input2) * grad_scaling
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



class BinaryQuantize_IRNet(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None



class BinaryQuantize_RBNN(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None



def get_ab(N):
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i
