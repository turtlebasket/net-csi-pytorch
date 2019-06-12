# CLEVERWALL - MAIN
# ** Currently unusable, and still a MASSIVE WIP.

import torch
from torch import nn
import os
from scapy.utils import rdpcap

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ok so it looks like torch.device() got thanos snapped by the pytorch devs
# ** RESOLVE LATER

# load dataset from wireshark PCAP
class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.capture_reader= rdpcap(filename, 1000)
        self.packets = []
        for (pkt_data, pkt_metadata) in self.capture_reader:
            # Append array to array. Should make data/metadata easier to track
            self.packets.append(tuple((pkt_data, pkt_metadata))) # pass in tuple

    # return length of packet list
    def __len__(self): 
        return len(self.packets)# FIX LATER.
    
    # return item to whoever's reading from dataset
    def __getitem__(self, index):
        return self.packets[index]
        
train_dataset = TrafficDataset('./data/train.pcap')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset)

# basic nn class setup
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out