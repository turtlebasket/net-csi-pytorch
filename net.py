#!/usr/bin/env python3

"""
CLEVERWALL - MAIN
** Currently unusable, and still a MASSIVE WIP.
"""

# TODO: 
# - [ ] Find a way to map hex dump onto a smaller array!!

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import dataset

# Hyper-parameters
INPUT_SIZE = 200
HIDDEN_SIZE = 300
NUM_LAYERS = 3

NUM_CLASSES = 2
NUM_EPOCHS = 6
BATCH_SIZE = 100
LEARNING_RATE = 0.00000005 # important if the network's gonna pick up on patterns

# STORAGE FILE NAME
DATA_STOR="dataset1.pkl"

# For some reason, linter can't find `torch.device`. Ignore for now
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNet(nn.Module):
    """
    Basic RNN network.
    """

    # ok bois we goin with a RNN model
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(NeuralNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        out, _ = self.lstm(x, (hidden_state, cell_state))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':

    print(
        "               __                      __ \n"
        " .-----.-----.|  |_ ______.----.-----.|__|\n"
        " |     |  -__||   _|______|  __|__ --||  |\n"
        " |__|__|_____||____|      |____|_____||__|\n"
    )

    print("Import dataset...", end=" ")

    try:
        train_dataset = dataset.load_dataset(DATA_STOR)
    except (FileNotFoundError, EOFError) as e:
        train_dataset = dataset.TrafficDataset('./data/train_capture.pcap', './data/train_rules.json')
        dataset.store_dataset(train_dataset, DATA_STOR)
    print("Done.")

    print("Create dataloader...", end=" ")
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.multisize_collate_fn)
    print("Done.")

    # testy
    # print(train_dataset[1])

    print("Create neural network instance...", end=" ")
    net_obj = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    print("Done.")

    print("Set network parameters...", end=" ")
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

            # print("outputs: {}".format(outputs))
            loss = criterion(outputs, flags.long())
            # print("loss calculated, {}".format(loss.item()))

            loss.backward()
            optimizer.step()

            # log
            if (i+1) % 100 == 0: # if iteration over dataset is complete...
                print("Epoch {}/{}\t\tStep {}/{}\t\tLoss = {:.5f}".format(epoch+1, NUM_EPOCHS, i+1, total_steps, loss.item()))

    print("Done.")

    # Testy code down here
    # At the end, I'll want to write to a checkpoint (.ckpt)