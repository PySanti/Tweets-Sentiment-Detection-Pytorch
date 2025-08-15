from torch.utils.data import DataLoader
from utils.constants import *
from torch.utils.data import TensorDataset
import torch

def sparse_collate_fn(batch):
    X_batch, Y_batch = zip(*batch)
    return torch.stack([x.to_dense() for x in X_batch]), torch.stack(Y_batch)



def generate_dataloaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE):
    train_dataloader, validation_dataloader, test_dataloader = None, None, None
    if train_dataset:
        train_dataloader = DataLoader(
                TensorDataset(train_dataset[0], train_dataset[1]), 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                collate_fn=sparse_collate_fn)
    if val_dataset:
        validation_dataloader = DataLoader(
                TensorDataset(val_dataset[0], val_dataset[1]), 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                collate_fn=sparse_collate_fn)

    if test_dataset:
        test_dataloader = DataLoader(
                TensorDataset(test_dataset[0], test_dataset[1]), 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                collate_fn=sparse_collate_fn)
    return train_dataloader, validation_dataloader, test_dataloader

