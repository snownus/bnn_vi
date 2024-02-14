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


class BinarizeConv2dSDP(nn.Module):

    def __init__(self, K, in_chn, out_chn, dropout=0, kernel_size=3, stride=1, padding=1, bias=False, linear=False):
        super(BinarizeConv2dSDP, self).__init__()
        self.alpha = nn.Parameter(torch.rand(out_chn, 1, 1), requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape_w = (K, out_chn, in_chn, kernel_size, kernel_size)
        self.shape_sum_w = (out_chn, in_chn, kernel_size, kernel_size)
        self.K = K

        self.scale = 10000
        self.M = nn.Parameter(torch.zeros((self.shape_sum_w)), requires_grad=True)
        self.Z = nn.Parameter(torch.zeros((self.shape_w)), requires_grad=True)
        # torch.nn.init.xavier_uniform_(self.M.data)
        torch.nn.init.xavier_normal_(self.Z.data)
        # torch.nn.init.normal_(self.Z.data, mean=0.0, std=0.5)

        self.lambdaa = 1e-15
        self.Sigma = nn.Parameter(torch.zeros((self.shape_sum_w)), requires_grad=True)
        # torch.nn.init.xavier_normal_(self.Sigma.data)
        # torch.nn.init.xavier_uniform_(self.Sigma.data)

        self.linear = linear
        self.in_chn = in_chn
        self.out_chn = out_chn


    def forward(self, input):
        # bw = BinaryQuantize().apply(self.weights)

        # real_weights = self.weights.view(self.shape_sum_w)
        # bw = BinaryQuantize().apply(real_weights)
        
        m = self.M.view(self.number_of_weights)
        z = self.Z.view(self.K, self.number_of_weights)
        sigma = self.Sigma.view(self.number_of_weights)

        # print(f'before norm: sigma:{sigma}')
        with torch.no_grad():
            A = m*m + torch.sum(z.T**2,dim=1)/self.scale + sigma*sigma
            # print(f'A.shape: {A.shape}, self.M.shape: {self.M.shape}')
            m.data = m.data / torch.sqrt(A)
            z.data = z.data / torch.sqrt(A)
            sigma.data = sigma.data / torch.sqrt(A)
        # print(f'after norm: sigma:{sigma}')

        w = SDP().apply(m, z, sigma, torch.tensor(self.lambdaa), torch.tensor(self.scale), torch.tensor(self.K))
        # sigma.data += 2*self.lambdaa
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
    def forward(ctx, m, z, sigma, lambdaa, scale, K):
        rv = torch.normal(0.0, 1.0/np.sqrt(scale.item()), size=(1, K.item())).cuda()
        G = torch.normal(0.0, 1.0, size=(1, m.shape[0])).cuda()
        zz = torch.mm(rv, z)
        # print(f'sigma.shape: {sigma.shape}, G.shape: {G.shape}, sigma*G.shape: {(sigma*G).shape}')
        # print(f'sigma: {sigma}')
        w = zz + m + sigma * G
        ctx.save_for_backward(rv, scale, lambdaa)

        return w

    @staticmethod
    def backward(ctx, grad_output):
        rv, scale, lambdaa = ctx.saved_tensors
        # grad_input1, grad_input2, grad_input3 = grad_output.clone(), grad_output.clone(), grad_output.clone()
        grad_input1, grad_input2 = grad_output.clone(), grad_output.clone()
        grad_m = grad_input1
        grad_z = torch.mm(rv.T, grad_input2) * np.sqrt(scale.item())
        grad_sigma = torch.ones_like(grad_input2) * (-2) * lambdaa
        # print(f'grad_sigma: {grad_sigma}')

        return grad_m, grad_z, grad_sigma, None, None, None


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
