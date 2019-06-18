"""
CLEVERWALL - MAIN
** Currently unusable, and still a MASSIVE WIP.
"""

# import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scapy.all import rdpcap, hexdump

# Hyper-parameters
INPUT_SIZE = 784    # Size of input layer
HIDDEN_SIZE = 500   # Size of output layer
NUM_CLASSES = 10
NUM_EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ok so it looks like torch.device() got thanos snapped by the pytorch devs
# ** RESOLVE LATER

class TrafficDataset(Dataset):
    """
    Extends torch's default Dataset class
    load dataset from wireshark PCAP
    """

    def __init__(self, filename):
        self.capture_reader = rdpcap(filename, 1000)

    # return length of packet list
    def __len__(self):
        return len(self.capture_reader)# FIX LATER.

    # return item to whoever's reading from dataset
    def __getitem__(self, index):
        return hexdump(self.capture_reader)

TRAIN_DATASET = TrafficDataset('./data/train_capture.pcap')
TRAIN_LOADER = DataLoader(dataset=TRAIN_DATASET)


class NeuralNet(nn.Module):
    """
    basic nn model setup (fully connected feed-forward neural net).
    Inherits from nn.Module, which is for storing machine state
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Let's work with some convolutional layers later, as they're
        # supposedly better at detecting spatial features
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
print("=== NETWORK INSTANTIATED ===\n{}".format(net))

# start on ML algorithm later
# for epoch in range(NUM_EPOCHS)
