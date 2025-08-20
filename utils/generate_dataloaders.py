from torch.utils.data import DataLoader
from utils.constants import *
from torch.utils.data import TensorDataset
from utils.TweetDataset import TweetDataset
import torch
from utils.constants  import BATCH_SIZE

def generate_dataloaders(tokenizer,train_dataset=None, val_dataset=None, test_dataset=None):
    dataloaders = []
    for dataset in [train_dataset, val_dataset, test_dataset]:
        new_dataloder = None
        if dataset:
            new_dataloder = DataLoader(
                    TweetDataset(tweets=dataset[0], labels=dataset[1], tokenizer=tokenizer), 
                    batch_size=BATCH_SIZE, 
                    shuffle=False,
                    num_workers=5,
                    pin_memory=True,
                    persistent_workers=True)
        dataloaders.append(new_dataloder)
    return dataloaders
