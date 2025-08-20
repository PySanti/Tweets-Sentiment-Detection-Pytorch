import matplotlib.pyplot as plt
def plot_model_performance(train_loss, val_loss):
    """
    Grafica las curvas de pérdida de entrenamiento y validación.

    Args:
        train_loss (list or array): Lista de pérdidas por época en entrenamiento.
        val_loss (list or array): Lista de pérdidas por época en validación.
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='red', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', color='blue', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
