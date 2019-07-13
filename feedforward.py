"""
CLEVERWALL - MAIN
** Currently unusable, and still a MASSIVE WIP.
"""

from scapy.all import rdpcap, hexdump
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Hyper-parameters
INPUT_SIZE = 784    # Size of input layer
HIDDEN_SIZE = 500   # Size of output layer
NUM_CLASSES = 10
NUM_EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# For some reason, linter can't find `torch.device`. Ignore for now
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrafficDataset(Dataset):
    """
    Extends torch's default Dataset class
    load data from dataset from entire PCAP,
    check with data from suricata's log (EVE.JSON format)
    """

    def __init__(self, traffic_filename, rules_filename):
        # read pcap data for packets
        self.capture_reader = rdpcap(traffic_filename, 1000)
        # indices that suricata has flagged
        self.flagged_indices = []
        scan_file = open(rules_filename, 'r')

        # optimize later :-/
        # for now, get 'pcap_cnt' int value
        for line in scan_file:
            self.flagged_indices.append(json.loads(line))

    # return length of packet list
    def __len__(self):
        return len(self.capture_reader)# FIX LATER.

    # return item to whoever's reading from dataset
    def __getitem__(self, index):
        target_entry = self.capture_reader[index]
        dump = str(hexdump(target_entry, dump=True))

        # Suricata logs packet count starting from zero, whereas
        # scapy starts from 1
        flagged = True if index+1 in self.flagged_indices else False

        # pass value back as dict
        # return {'dump':dump, 'flagged': flagged}
        return dump, flagged


class NeuralNet(nn.Module):
    """
    basic nn model setup (fully connected feed-forward neural net).
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Let's work with some convolutional layers later, as they're
        # supposedly better at detecting spatial features
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # at some point, probably wanna learn what all these layers actually do :c
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net_obj = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
print("=== NETWORK INSTANTIATED ===\n{}".format(net_obj))

# start on ML algorithm later
# for epoch in range(NUM_EPOCHS)

# Learn the details of how the algorithm works later :P
criterion = nn.CrossEntropyLoss()

# Adam (from Optimizer) - specialized optimization algorithm
optimizer = torch.optim.Adam(net_obj.parameters())

# Training Phase
total_steps = len(train_loader)

"""
for epoch in range(NUM_EPOCHS):
    for i, captures in enumerate(train_loader):
        # Read capture item through tensors
        # Step forward
        # backpropagate and correct
"""
