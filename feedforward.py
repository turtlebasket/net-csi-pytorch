# CLEVERWALL - MAIN
# ** Currently unusable, and still a MASSIVE WIP.

import pandas
import torch
from torch import nn

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

# load dataset from wireshark CSV
class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir):
        self.csv_file = pandas.read_csv(csv_file)
        self.root_dir = root_dir

    # return length given by CSV reader
    def __len__(self): 
        return len(self.csv_file)
    
    # return item to whoever's reading from dataset
    def __getitem(self, index):
        # NOT FINISHED. figure out indexing
        return 0

train_dataset = TrafficDataset('train.csv', './data/')
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