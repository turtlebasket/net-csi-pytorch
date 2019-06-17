"""
CLEVERWALL - MAIN
** Currently unusable, and still a MASSIVE WIP.
"""

import torch
from torch import nn
from scapy.all import rdpcap, hexdump

# Hyper-parameters
INPUT_SIZE = 784
HIDDEN_SIZE = 500
NUM_CLASSES = 10
NUM_EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ok so it looks like torch.device() got thanos snapped by the pytorch devs
# ** RESOLVE LATER

"""
Extend torch's default Dataset class
load dataset from wireshark PCAP
"""
class TrafficDataset(torch.utils.data.Dataset):

    def __init__(self, filename):
        self.capture_reader = rdpcap(filename, 1000)

    # return length of packet list
    def __len__(self):
        return len(self.capture_reader)# FIX LATER.

    # return item to whoever's reading from dataset
    def __getitem__(self, index):
        return hexdump(self.capture_reader)

TRAIN_DATASET = TrafficDataset('./data/train.pcap')
TRAIN_LOADER = torch.utils.data.DataLoader(dataset=TRAIN_DATASET)

# basic nn class setup
class NeuralNet(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
