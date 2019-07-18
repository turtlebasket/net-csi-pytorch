#!/usr/bin/env python3

"""
CLEVERWALL - MAIN
** Currently unusable, and still a MASSIVE WIP.
"""

# TODO: 
# - [ ] Move tensor getter to init
# - [ ] Pickle dataset, load from pkl file!
# - [ ] Find out what that issue is with loss being zero

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import dataset

# Hyper-parameters
# INPUT_SIZE = 784    # Size of input layer
INPUT_SIZE = 25000 # Size of input layer
HIDDEN_SIZE = 500   # Size of output layer
NUM_CLASSES = 2
NUM_EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# STORAGE FILE NAME
DATA_STOR="dataset1.pkl"

# For some reason, linter can't find `torch.device`. Ignore for now
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    print("Import Dataset...", end=" ")

    try:
        train_dataset = dataset.load_dataset(DATA_STOR)
    except (FileNotFoundError, EOFError) as e:
        train_dataset = dataset.TrafficDataset('./data/train_capture.pcap', './data/train_rules.json', INPUT_SIZE)
        dataset.store_dataset(train_dataset, DATA_STOR)
    print("Done.")

    print("Create Dataloader...", end=" ")
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
    # At the end, I'll want to write to a checkpoint (.ckpt)