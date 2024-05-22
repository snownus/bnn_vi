import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import random
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn
import networkx as nx
from networkx import grid_graph
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from models.sdp_wo_entropy import BinarizeConv2dSDP

class MLP(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels):
        super().__init__()
        self.fc1 = Linear(input_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, output_channels)
        
    def forward(self, x):
        #print(x.shape)
        x = F.sigmoid(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        return x


class MLP_SDP(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels, K=4, scale=100):
        super().__init__()
        self.fc1 = BinarizeConv2dSDP(K, scale, input_channels, hidden_channels, kernel_size=1, padding=0, linear=True, bias=False)
        self.fc2 = BinarizeConv2dSDP(K, scale, hidden_channels, output_channels, kernel_size=1, padding=0, linear=True, bias=False)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def models(dataset, methods, hidden_channels, K, scale, input_channels=-1):
    if dataset == 'complete':
        input_channels, output_channels = 19900, 200
    elif dataset == 'random':
        input_channels, output_channels = input_channels, 500
    elif dataset == 'grid':
        input_channels, output_channels = 19800, 10000

    if methods == 'SDP':
        return MLP_SDP(input_channels=input_channels, 
                    output_channels=output_channels, 
                    hidden_channels=hidden_channels, K=K, scale=scale)
    elif methods == 'BASE':
        return MLP(input_channels, output_channels, hidden_channels)

