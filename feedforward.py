#!/usr/bin/env python3

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
# INPUT_SIZE = 784    # Size of input layer
INPUT_SIZE = 25000 # Size of input layer
HIDDEN_SIZE = 500   # Size of output layer
NUM_CLASSES = 2
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

    def __init__(self, traffic_filename, rules_filename, input_size):
        # read pcap data for packets
        self.capture_reader = rdpcap(traffic_filename, 1000)
        scan_file = open(rules_filename, 'r')

        # indices that suricata has flagged
        flagged_entries = []
        for line in scan_file:
            flagged_entries.append(json.loads(line))

        self.flagged_indices = []
        for e in flagged_entries:
            # find a proper way to get EVERYTHING later.
            try:
                self.flagged_indices.append(e['pcap_cnt'])
            except KeyError:
                pass

    # return length of packet list
    def __len__(self):
        return len(self.capture_reader)# FIX LATER.

    # return item to whoever's reading from dataset
    def __getitem__(self, index):
        target_entry = self.capture_reader[index]
        dump = str(hexdump(target_entry, dump=True))
        # dump_tensor = torch.zeros(len(dump)).float()
        dump_tensor = torch.zeros(INPUT_SIZE).float()
        for c in range(len(dump)):
            # get 
            dump_tensor[c] = ord(dump[c])

        # IMPORTANT: Suricata logs packet count starting from zero, whereas scapy starts from 1
        flagged = (index+1 in self.flagged_indices)

        # pass value back as tuple
        return dump_tensor, flagged

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

if __name__ == '__main__':

    print(
        "      __                             ____  \n"
        " ____/ /__ _  _____ _____    _____ _/ / /  \n"
        "/ __/ / -_) |/ / -_) __/ |/|/ / _ `/ / /   \n"
        "\\__/_/\\__/|___/\\__/_/  |__,__/\\_,_/_/_/\n"
    )

    print("Create Dataloader...", end=" ")
    train_dataset = TrafficDataset('./data/train_capture.pcap', './data/train_rules.json', INPUT_SIZE)
    train_loader = DataLoader(dataset=train_dataset)
    print("Done.")

    """
    # TESTY ZONE
    dataiter = iter(train_loader)
    packet, flag = dataiter.next()
    print("Packets:\n{}\nFlag:\n{}\n".format(packet, flag))
    """

    print("Create Network Instance...", end=" ")
    net_obj = NeuralNet(INPUT_SIZE, INPUT_SIZE, NUM_CLASSES)
    print("Done.")

    print("Set parameters...", end=" ")
    # Learn the details of how Cross Entropy Loss works later
    criterion = nn.CrossEntropyLoss()
    # Adam (from Optimizer) - specialized optimization algorithm
    optimizer = torch.optim.Adam(net_obj.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader)
    print("Done.")

    # Trainy algorithm
    print("Start analysis.")

    for epoch in range(NUM_EPOCHS):
        for i, (inputs, flags) in enumerate(train_loader, 0):

            # IMPORTANT: Transform `inputs` tensor to match machine interface
            # print("inputs: {}".format(inputs))

            # clear gradients
            optimizer.zero_grad()

            # backtrack & optimize
            outputs = net_obj(inputs)
            loss = criterion(outputs, flags.long())
            # print("loss calculated.")
            loss.backward()
            # print("backpropagation completed.")
            optimizer.step()
            # print("stepped forward.")

            # log
            if (i+1) % 100 == 0: # if iteration over dataset is complete...
                print("[ Epoch {}/{} ] [ Step {}/{} ] [ Loss = {:.5f} ]".format(epoch, NUM_EPOCHS, i+1, total_steps, loss.item()))

        print("Done.")

    # Testy code down here