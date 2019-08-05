from scapy.all import rdpcap, hexdump
import json
import pickle
import torch
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    """
    Extends torch's default Dataset class. Loads data from dataset from entire PCAP,
    checks with data from suricata's log (EVE.JSON)
    """

    def __init__(self, traffic_filename, rules_filename):

        # Read from files
        capture_reader = rdpcap(traffic_filename, 1000)
        self.dataset_length = len(capture_reader)
        scan_file = open(rules_filename, 'r')

        # Get packet hex dumps & transform onto tensors
        self.dump_tensors = []
        for item in capture_reader:
            dump = str(hexdump(item, dump=True))
            dump_tensor = torch.zeros(len(dump)).float()
            for c in range(len(dump)):
                # try:
                dump_tensor[c] = ord(dump[c])
                # except IndexError: # stop copying once out of range. find better solution later
                #     break
            self.dump_tensors.append(dump_tensor)

        # Determine whether nor not packet was flagged by suricata
        flagged_entries = []
        for line in scan_file:
            flagged_entries.append(json.loads(line))

        flagged_indices = []
        for e in flagged_entries:
            # find a proper way to get ALL ENTRIES later.
            try:
                flagged_indices.append(e['pcap_cnt'])
            except KeyError:
                pass

        # IMPORTANT: Suricata logs packet count starting from zero, whereas scapy starts from 1
        self.flags = []
        for index in range(self.dataset_length):
            self.flags.append(index+1 in flagged_indices)

    def __len__(self):
        return self.dataset_length

    # return item to whoever's reading from dataset
    def __getitem__(self, index):
        # get items from respective arrays

        # pass value back as tuple (dump: Tensor, flag: Tensor)
        return self.dump_tensors[index], self.flags[index]

#easier-to-manage wrappers

def store_dataset(dataset: TrafficDataset, filename: str):
    with open(filename, 'wb+') as stor:
        pickle.dump(dataset, stor)

def load_dataset(filename: str):
    with open(filename, 'rb') as stor:
        return pickle.load(stor)

# custom collate_fn for multi-size batch processing (used by dataloader)

def multisize_collate_fn(batch):
    # values passed back as (dump, flag)
    # dumps = [item[0] for item in batch]
    # flags = [item[1] for item in batch]

    dumps = []
    flags = []

    for item in batch:
        dumps.append(item[0])
        flags.append(item[1])

    print(dumps)
    print(flags)

    dumps = torch.LongTensor(dumps)
    flags = torch.LongTensor(flags)
    # return torch.stack(batch, 0)
    # return torch.as_tensor(batch)
    return [dumps, flags]