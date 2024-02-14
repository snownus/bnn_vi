import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class BinarizeConv2d(nn.Module):

    def __init__(self, K, in_chn, out_chn, dropout=0, kernel_size=3, stride=1, padding=1, bias=False, linear=False):
        super(BinarizeConv2d, self).__init__()
        self.alpha = nn.Parameter(torch.rand(out_chn, 1, 1), requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        # self.shape = (K, self.number_of_weights, 1, 1)
        self.shape_w = (K, out_chn, in_chn, kernel_size, kernel_size)
        self.shape_sum_w = (out_chn, in_chn, kernel_size, kernel_size)
        self.K = K
        self.weights = nn.Parameter(torch.zeros((self.shape_w))*0.001, requires_grad=True)
        self.weights.binary = True
        torch.nn.init.xavier_uniform_(self.weights.data)

        self.register_buffer('RV', torch.ones(K+1, requires_grad=False))
        val = torch.empty(1, K+1)
        torch.nn.init.xavier_uniform_(val)
        self.RV = val[0]

        self.fc1 = nn.Linear(K+1, 2*(K+1))
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2*(K+1), K)
        self.dropout_ratio = dropout

        self.linear = linear

        self.binary_activation = BinaryActivation()

        self.in_chn = in_chn
        self.out_chn = out_chn

    def forward(self, input):
        if self.training:
            val = torch.rand(1).cuda()
            self.RV[-1] = val[0]
            rv = self.fc1(self.RV)
            rv = self.relu1(rv)
            rv = self.fc2(rv)
            rv =  torch.nn.functional.sigmoid(rv) - 0.5
            # rv = torch.nn.functional.softmax(rv) - 0.5
            RV_temp = self.RV.clone()
            RV_temp[0:self.K] = rv.detach()
            self.RV.data = RV_temp
            # self.RV[0:self.K].data = rv.detach()
        else:
            rv = self.RV[0:self.K]

        rv = rv.view(-1, self.K)
        w = self.weights.view(self.K, self.number_of_weights)
        w = torch.mm(rv, w)
        real_weights = w.view(self.shape_sum_w)

        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        bw = scaling_factor * BinaryQuantize().apply(real_weights)

        a = input
        if self.linear:
            a = a[:, :, None, None]
        ba = self.binary_activation(a)
        output = F.conv2d(ba, bw, stride=self.stride, padding=self.padding, bias=None)
        #* scaling factor
        output = output * self.alpha
        if self.linear:
            output = torch.squeeze(output)

        return output


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        out = torch.sign(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        # input = ctx.saved_tensors[0]
        # grad_input = grad_output.clone()
        # grad_input = torch.where(abs(input)>1, torch.tensor(0, dtype=grad_input.dtype).cuda(), grad_input)
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        out = torch.sign(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*inputs))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input

