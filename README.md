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

Es importante destacar que existen 4 etiquetas: positive,negative,uncertainty,litigious.
Es importante ademas destacar que la frecuencia de aparicion de idiomas es asi:

```
en : 93%
fr : 1%
Other (53453) : 6%
```

Es importante destacar que se comprobo que las clases estuvieran relativamente balanceadas:

```
positive : 28.21%
negative : 27.96%
uncertainty : 22.07%
litigious : 21.77%
```


Esto lo tomaremos en cuenta para transformar el texto.

# Preprocesamiento

Primeramente dividimos el dataframe basico verticalmente para obtener dos dataframes : uno con una sola columna (las etiquetas) y otro con dos columnas (para cada fila, el texto y su lenguaje).

Luego creamos las funciones convert_base_X y clean_text para convertir el Dataframe de texto + lenguaje en un arreglo de numpy donde para cada posicion se tuviera el texto con el lenguaje y el texto base limpiado. Este es el codigo.


```python

# main.py

from utils.convert_base_X import convert_base_X
from utils.load_base_data import load_base_data

print("Cargando datos base")
base_dataset = load_base_data()
X = base_dataset.drop(labels=['Label'], axis=1)
Y = base_dataset['Label'].to_numpy()

print("Convirtiendo X base")
X = convert_base_X(X)

print("Generando corpus")
corpus = set([w.strip().lower() for w in " ".join(X).split(" ")])

print(corpus)
print(len(corpus))
```

```python

# utils/load_base_data.py

import kagglehub
import pandas as pd

def load_base_data():
    # Download latest version
    path = kagglehub.dataset_download("tariqsays/sentiment-dataset-with-1-million-tweets")
    base_dataset = pd.DataFrame(pd.read_csv(path + "/dataset.csv"))
    return base_dataset

```

```python

# utils/convert_base_X
import numpy as np
from .clean_text import clean_text

def convert_base_X(X, SUPPORTED_LAN=["en", "fr"]):
    """
        Recibe el dataframe basico cargado que contiene ~1M de registros
        con dos columnas : "Text" y "Language" y las combina para
        crear un dataframe con solo registros de texto.

        Ademas, elimina palabras eliminables de los textos llamando a la funcion
        clean_text.
    """
    def process_row(row):
        print(f"\t{row.name}/{len(X)}", end="\r")
        lan_tag = "lan_" + (row.Language if row.Language in SUPPORTED_LAN else "other")
        new_row = lan_tag + " " + clean_text(row.Text, row.Language)
        return new_row
    return X.apply(process_row, axis=1).values



```

```python
# utils/clean_text.py

import re
from nltk.stem import LancasterStemmer, SnowballStemmer
import emoji
import string

language_map = {
    "el": "greek",
    "pt": "portuguese",
    "ca": "catalan",
    "tl": "tagalog",
    "da": "danish",
    "hu": "hungarian",
    "ht": "haitian creole",
    "fr": "french",
    "qht": "haitian creole (Twitter-specific)",
    "is": "icelandic",
    "th": "thai",
    "pa": "punjabi",
    "am": "amharic",
    "und": "undetermined",
    "qst": "spanish (Twitter-specific)",
    "bn": "bengali",
    "en": "english",
    "cs": "czech",
    "sl": "slovene",
    "ro": "romanian",
    "eu": "basque",
    "vi": "vietnamese",
    "fi": "finnish",
    "ur": "urdu",
    "sv": "swedish",
    "cy": "welsh",
    "nl": "dutch",
    "qme": "meitei (Twitter-specific)",
    "it": "italian",
    "iw": "hebrew",  # Deprecated, use 'he' in modern standards
    "ta": "tamil",
    "zh": "chinese",
    "es": "spanish",
    "ne": "nepali",
    "sr": "serbian",
    "sd": "sindhi",
    "fa": "persian",
    "lt": "lithuanian",
    "et": "estonian",
    "in": "indonesian",  # Deprecated, use 'id' in modern standards
    "ja": "japanese",
    "tr": "turkish",
    "ar": "arabic",
    "ru": "russian",
    "ko": "korean",
    "de": "german",
    "zxx": "no linguistic content",
    "ckb": "central kurdish",
    "qam": "armenian (Twitter-specific)",
    "ml": "malayalam",
    "no": "norwegian",
    "pl": "polish",
    "lv": "latvian",
    "art": "artificial language",
    "bg": "bulgarian",
    "or": "oriya",
    "uk": "ukrainian",
    "mr": "marathi",
    "hi": "hindi",
    "te": "telugu",
    "si": "sinhala",
    "kn": "kannada",
    "gu": "gujarati"
}


def clean_text(text, lan) -> str:
    """
        Elimina emojis, @s y urls del texto.
    """

    # precompilar los patrones y los steemers
    text = text.strip()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r'(@\w+|http\S+|#\w+)', '', text)
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    try:
        if type(lan) == str and 'http' not in lan:
            if lan == "en":
                steemer = LancasterStemmer()
            else:
                steemer = SnowballStemmer(language_map[lan])
            text = " ".join([steemer.stem(w).lower() for w in text.split(" ")])
    except:
        pass
    finally:
        return text


```

Con el codigo actual, la longitud del corpus es de ~750.000 palabras, demasiado grande. Buscaremos reducirlo aun mas y optimizar el codigo.
