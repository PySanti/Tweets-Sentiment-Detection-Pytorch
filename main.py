import numpy as np
from pandas import test
from utils.constants import BATCH_SIZE
from utils.get_preprocessed_data import get_preprocessed_data
import torch
from utils.MLP import MLP
from utils.generate_dataloaders import generate_dataloaders
from utils.initialize_tokenizer import initialize_tokenizer
import time
from utils.plot_model_performance import plot_model_performance

if __name__ == "__main__":
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_preprocessed_data()
    mlp = MLP(embed_dim=256,hidden_sizes=[300,300,300],out_size=4, drop_rate=0.2).to("cuda")
    tokenizer = initialize_tokenizer(X_train)

    train_dataloader, validation_dataloader,test_dataloader= generate_dataloaders(
            train_dataset=(X_train, Y_train),
            val_dataset=(X_val, Y_val),
            test_dataset=(X_test, Y_test),
            tokenizer=tokenizer
            )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=5e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    epochs_train_loss = []
    epochs_val_loss = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,              #  Optimizador al que se le ajustará el LR
        mode='max',             #  Reduce el LR si la métrica (ej. pérdida) deja de disminuir
        factor=0.1,             #  Multiplica el LR por este factor cuando se activa (ej. 0.001 → 0.0001)
        patience=4,             #  Número de épocas sin mejora antes de reducir el LR
        threshold=1e-3,         #  Mínimo cambio considerado como mejora significativa
        threshold_mode='rel',   #  'rel' compara de forma relativa al mejor valor; 'abs' es absoluto
        cooldown=0,             #  Épocas de espera tras reducir el LR antes de volver a evaluar
        min_lr=1e-6,            #  Valor mínimo que puede alcanzar el LR
        eps=1e-8                #  Mínima diferencia entre LR actual y nuevo para aplicar el cambio
    )

    for i in range(25):
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

            with torch.amp.autocast(device_type='cuda'):
                output = mlp(X_batch)
                loss = criterion(output, Y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())


        mlp.eval()

        with torch.no_grad():
            for i, (X_batch, Y_batch) in enumerate(validation_dataloader):
                print(f"Batch : {i}/{len(X_val)//BATCH_SIZE}", end="\r")
                X_batch, Y_batch = X_batch.to('cuda'), Y_batch.to('cuda')

                output = mlp(X_batch)
                loss = criterion(output, Y_batch)
                _, predicted = torch.max(output, 1)

                val_accuracy.append((Y_batch == predicted).cpu().sum()/BATCH_SIZE)
                val_loss.append(loss.item())

        scheduler.step(np.mean(val_accuracy))        # Se llama con la métrica que se quiere monitorizar
        epochs_train_loss.append(np.mean(train_loss))
        epochs_val_loss.append(np.mean(val_loss))
        print(f'Train loss : {np.mean(train_loss):.3f}')
        print(f'Val loss : {np.mean(val_loss):.3f}')
        print(f'Val accuracy : {np.mean(val_accuracy):.3f}')
        print(f'Overfitting : {100 - (np.mean(val_loss)*100/np.mean(train_loss)):.3f}')
        print(f"Tiempo de procesamiento de la epoca : {time.time()-t1:.3f}")
        print("_______________")

    plot_model_performance(epochs_train_loss,epochs_val_loss)
    

    mlp.eval()
    test_accuracy = []
    with torch.no_grad():
        for i, (X_batch, Y_batch) in enumerate(test_dataloader):
            print(f"Batch : {i}/{len(X_val)//BATCH_SIZE}", end="\r")
            X_batch, Y_batch = X_batch.to('cuda'), Y_batch.to('cuda')

            output = mlp(X_batch)
            _, predicted = torch.max(output,1)
            test_accuracy.append((Y_batch == predicted).to('cpu').sum()/BATCH_SIZE)
        print(f"Test accuracy : {np.mean(test_accuracy):.3f}")

