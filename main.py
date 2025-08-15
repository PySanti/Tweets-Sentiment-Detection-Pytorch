from torch.utils.data import TensorDataset
from utils.constants import *
from torch.utils.data import DataLoader
import torch
from utils.get_preprocessed_data import get_preprocessed_data


(X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_preprocessed_data()

train_dataloader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=False)
validation_dataloader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

