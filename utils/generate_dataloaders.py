from torch.utils.data import DataLoader
from utils.constants import *
from torch.utils.data import TensorDataset
import torch

def sparse_collate_fn(batch):
    X_batch, Y_batch = zip(*batch)
    return torch.stack([x.to_dense() for x in X_batch]), torch.stack(Y_batch)

def generate_dataloaders(BATCH_SIZE,train_dataset=None, val_dataset=None, test_dataset=None):
    dataloaders = []
    for dataset in [train_dataset, val_dataset, test_dataset]:
        new_dataloder = None
        if dataset:
            new_dataloder = DataLoader(
                    TensorDataset(dataset[0], dataset[1]), 
                    batch_size=BATCH_SIZE, 
                    shuffle=False, 
                    collate_fn=sparse_collate_fn)
        dataloaders.append(new_dataloder)
    return dataloaders
