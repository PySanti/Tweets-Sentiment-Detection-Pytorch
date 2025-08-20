import numpy as np
from utils.constants import BATCH_SIZE
from utils.get_preprocessed_data import get_preprocessed_data
import torch
from utils.MLP import MLP
from utils.generate_dataloaders import generate_dataloaders
from utils.initialize_tokenizer import initialize_tokenizer
import time

if __name__ == "__main__":
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_preprocessed_data()

    tokenizer = initialize_tokenizer(X_train)

    train_dataloader, validation_dataloader,_= generate_dataloaders(
            train_dataset=(X_train, Y_train),
            val_dataset=(X_val, Y_val),
            tokenizer=tokenizer
            )
    mlp = MLP(embed_dim=256,hidden_sizes=[300,300,300],out_size=4).to("cuda")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=5e-3)
    for i in range(15):
        t1 = time.time()
        print(f"Epoch : {i}")
        train_loss = []
        val_loss = []
        val_accuracy = []

        mlp.train()
        for i, (X_batch, Y_batch) in enumerate(train_dataloader):
            print(f"Batch : {i}/{len(X_train)//BATCH_SIZE}", end="\r") 
            X_batch, Y_batch = X_batch.to('cuda'), Y_batch.to('cuda')
            optimizer.zero_grad()

            output = mlp(X_batch)
            loss = criterion(output, Y_batch)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())


        mlp.eval()

        with torch.no_grad():
            for i, (X_batch, Y_batch) in enumerate(validation_dataloader):
                print(f"Batch : {i}/{len(X_val)//BATCH_SIZE}", end="\r")
                X_batch, Y_batch = X_batch.to('cuda'), Y_batch.to('cuda')

                output = mlp(X_batch)
                loss = criterion(output, Y_batch)
                _, predicted = torch.max(output, 1)
                print(predicted)

                val_accuracy.append((Y_batch == predicted).cpu().sum()/BATCH_SIZE)
                val_loss.append(loss.item())

        print(f'Train loss : {np.mean(train_loss)}')
        print(f'Val loss : {np.mean(val_loss)}')
        print(f'Val accuracy : {np.mean(val_accuracy)}')
        print(f'Overfitting : {100 - (np.mean(val_loss)*100/np.mean(train_loss))}')
        print(f"Tiempo de procesamiento de la epoca : {time.time()-t1}")


