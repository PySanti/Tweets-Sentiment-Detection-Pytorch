from logging import critical
from sklearn.model_selection import train_test_split
from utils.constants import BATCH_SIZE
from utils.get_preprocessed_data import get_preprocessed_data
import torch
from utils.MLP import MLP
from utils.generate_dataloaders import generate_dataloaders

if __name__ == "__main__":
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_preprocessed_data()
    train_dataloader, validation_dataloader,_= generate_dataloaders(
            train_dataset=(X_train, Y_train),
            val_dataset=(X_val, Y_val),
            BATCH_SIZE=128
            )
    mlp = MLP(
            input_shape=X_train.shape[1],
            hidden_sizes=[300,300,300], 
            out_size=4).to("cuda")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=1e-4)
    for i in range(100):
        for (X_batch, Y_batch) in train_dataloader:
            X_batch = X_batch.float().to("cuda")
            Y_batch = Y_batch.to("cuda")
            optimizer.zero_grad()

            output = mlp(X_batch)
            loss = criterion(output, Y_batch)

            loss.backward()
            optimizer.step()

            print(loss.item())

