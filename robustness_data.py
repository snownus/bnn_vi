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

def generate_complete_graph_data():
    G = nx.complete_graph(200)
    for node in G.nodes:
        G.nodes[node]['weight'] = random.choice([-1,1])
    for edge in G.edges:
        G.edges[edge]['weight'] = G.nodes[edge[0]]['weight'] * G.nodes[edge[1]]['weight']

    ### Create training Dataset
    train_data_input = []
    train_data_output_label = []
    for i in range(500):
        data_point_feature = []
        data_point_label = []
        for edge in G.edges:
            G.edges[edge]['weight'] = G.nodes[edge[0]]['weight'] * G.nodes[edge[1]]['weight']
            data_point_feature.append(G.edges[edge]['weight'])
        data_point_feature = np.array(data_point_feature)
        for node in G.nodes:
            G.nodes[node]['weight'] = random.choice([-1,1])
            data_point_label.append(G.nodes[node]['weight'])
        data_point_label = np.array(data_point_label)
        train_data_input.append(data_point_feature)
        train_data_output_label.append(data_point_label)
    feature = torch.Tensor(train_data_input)
    label = torch.Tensor(train_data_output_label)
    dataset = TensorDataset(feature,label)
    dataloader = DataLoader(dataset, batch_size=10)

    ### Create testing Dataset
    test_data_input = []
    test_data_output_label = []
    for i in range(100):
        data_point_feature = []
        data_point_label = []
        for edge in G.edges:
            G.edges[edge]['weight'] = G.nodes[edge[0]]['weight'] * G.nodes[edge[1]]['weight']
            data_point_feature.append(G.edges[edge]['weight'])
        data_point_feature = np.array(data_point_feature)
        for node in G.nodes:
            G.nodes[node]['weight'] = random.choice([-1,1])
            data_point_label.append(G.nodes[node]['weight'])
        data_point_label = np.array(data_point_label)
        test_data_input.append(data_point_feature)
        test_data_output_label.append(data_point_label)
    test_feature = torch.Tensor(test_data_input)
    test_label = torch.Tensor(test_data_output_label)
    test_dataset = TensorDataset(test_feature,test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=10)

    return dataloader, test_dataloader, feature.shape[1]

from random import sample
def generate_random_graph_data():
    G = nx.Graph()
    G.add_nodes_from([x for x in range(500)])
    for node in G.nodes:
        node_list = [x for x in range(len(G.nodes))]
        node_sample_list = node_list.pop(node)
        neighbor = sample([x for x in range(100)], 10)
        for adj in neighbor:
            G.add_edge(node, adj)
    for node in G.nodes:
        G.nodes[node]['weight'] = random.choice([-1,1])
    
    j=0
    for edge in G.edges:
        G.edges[edge]['weight'] = G.nodes[edge[0]]['weight'] * G.nodes[edge[1]]['weight']
        G.edges[edge]['label'] = j
        j += 1
    
    ### Create training Dataset
    train_data_input = []
    train_data_output_label = []
    for i in range(500):
        data_point_feature = []
        data_point_label = []
        for edge in G.edges:
            G.edges[edge]['weight'] = G.nodes[edge[0]]['weight'] * G.nodes[edge[1]]['weight']
            data_point_feature.append(G.edges[edge]['weight'])
        data_point_feature = np.array(data_point_feature)
        for node in G.nodes:
            G.nodes[node]['weight'] = random.choice([-1,1])
            data_point_label.append(G.nodes[node]['weight'])
        data_point_label = np.array(data_point_label)
        train_data_input.append(data_point_feature)
        train_data_output_label.append(data_point_label)
    feature = torch.Tensor(train_data_input)
    label = torch.Tensor(train_data_output_label)
    dataset = TensorDataset(feature,label)
    dataloader = DataLoader(dataset, batch_size=10)

    ### Create testing Dataset
    test_data_input = []
    test_data_output_label = []
    for i in range(100):
        data_point_feature = []
        data_point_label = []
        for edge in G.edges:
            G.edges[edge]['weight'] = G.nodes[edge[0]]['weight'] * G.nodes[edge[1]]['weight']
            data_point_feature.append(G.edges[edge]['weight'])
        data_point_feature = np.array(data_point_feature)
        for node in G.nodes:
            G.nodes[node]['weight'] = random.choice([-1,1])
            data_point_label.append(G.nodes[node]['weight'])
        data_point_label = np.array(data_point_label)
        test_data_input.append(data_point_feature)
        test_data_output_label.append(data_point_label)
    test_feature = torch.Tensor(test_data_input)
    test_label = torch.Tensor(test_data_output_label)
    test_dataset = TensorDataset(test_feature,test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=10)

    return dataloader, test_dataloader, feature.shape[1]


def generate_grid_graph_data():
    G = grid_graph(dim=(100, 100))
    i = 0
    for node in G.nodes:
        G.nodes[node]['label'] = i
        G.nodes[node]['weight'] = random.choice([-1,1])
        i+=1
        
    j=0
    for edge in G.edges:
        G.edges[edge]['weight'] = G.nodes[edge[0]]['weight'] * G.nodes[edge[1]]['weight']
        G.edges[edge]['label'] = j
        j += 1
    
    ### Create training Dataset
    train_data_input = []
    train_data_output_label = []
    for i in range(500):
        data_point_feature = []
        data_point_label = []
        for edge in G.edges:
            G.edges[edge]['weight'] = G.nodes[edge[0]]['weight'] * G.nodes[edge[1]]['weight']
            data_point_feature.append(G.edges[edge]['weight'])
        data_point_feature = np.array(data_point_feature)
        for node in G.nodes:
            G.nodes[node]['weight'] = random.choice([-1,1])
            data_point_label.append(G.nodes[node]['weight'])
        data_point_label = np.array(data_point_label)
        train_data_input.append(data_point_feature)
        train_data_output_label.append(data_point_label)
    feature = torch.Tensor(train_data_input)
    label = torch.Tensor(train_data_output_label)
    dataset = TensorDataset(feature,label)
    dataloader = DataLoader(dataset, batch_size=10)

    ### Create testing Dataset
    test_data_input = []
    test_data_output_label = []
    for i in range(50):
        data_point_feature = []
        data_point_label = []
        for edge in G.edges:
            G.edges[edge]['weight'] = G.nodes[edge[0]]['weight'] * G.nodes[edge[1]]['weight']
            data_point_feature.append(G.edges[edge]['weight'])
        data_point_feature = np.array(data_point_feature)
        for node in G.nodes:
            G.nodes[node]['weight'] = random.choice([-1,1])
            data_point_label.append(G.nodes[node]['weight'])
        data_point_label = np.array(data_point_label)
        test_data_input.append(data_point_feature)
        test_data_output_label.append(data_point_label)
    test_feature = torch.Tensor(test_data_input)
    test_label = torch.Tensor(test_data_output_label)
    test_dataset = TensorDataset(test_feature,test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=10)

    return dataloader, test_dataloader, feature.shape[1]

