import enum
import numpy as np
from logging import critical
from sklearn.model_selection import train_test_split
from utils.constants import BATCH_SIZE
from utils.get_preprocessed_data import get_preprocessed_data
import torch
from utils.MLP import MLP
from utils.generate_dataloaders import generate_dataloaders
from tokenizers import Tokenizer, trainers, models

if __name__ == "__main__":
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_preprocessed_data()

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer()
    tokenizer.train_from_iterator(X_train,trainer=trainer)

    train_dataloader, validation_dataloader,_= generate_dataloaders(
            train_dataset=(X_train, Y_train),
            val_dataset=(X_val, Y_val),
            tokenizer=tokenizer
            )
    mlp = MLP(embed_dim=256,hidden_sizes=[300,300,300],out_size=4).to("cuda")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=1e-3)
    for i in range(100):
        train_loss = []
        for i, (X_batch, Y_batch) in enumerate(train_dataloader):
            print(f"Batch : {i}/{len(X_train)//BATCH_SIZE}", end="\r")
            X_batch = X_batch.to("cuda")
            Y_batch = Y_batch.to("cuda")
            optimizer.zero_grad()

            output = mlp(X_batch)
            loss = criterion(output, Y_batch)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print("\n\n")
        print(np.mean(train_loss))

