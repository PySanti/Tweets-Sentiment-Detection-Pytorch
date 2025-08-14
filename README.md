# Tweets Sentiment Detection with Pytorch


El objetivo de este proyecto será implementar la librería `PyTorch` para resolver el problema propuesto por el siguiente [dataset](https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets/data).


Se implementará una red neuronal con arquitectura tipo MLP.

Además, se implementarán técnicas avanzadas para la mejora del rendimiento del modelo, entre estas: Learning Rate Scheduling, Hypertunning (con ray tune), Batch Normalization, He Initialization, Regularization (L1 o L2 con Dropout), etc.


# Visualización del Dataset

Usando el siguiente código cargamos el dataset y revisamos su estructura:

```python
import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("tariqsays/sentiment-dataset-with-1-million-tweets")
x = pd.DataFrame(pd.read_csv(path + "/dataset.csv"))
print(x.shape)
print(x.iloc[0])

```

Resultado:

```python
(937854, 3)
Text        @Charlie_Corley @Kristine1G @amyklobuchar @Sty...
Language                                                   en
Label                                               litigious
Name: 0, dtype: object
```

El dataset base contiene 937.854 registros, donde cada registro cuenta con 3 características: Text, Language y Label.


# Preprocesamiento

