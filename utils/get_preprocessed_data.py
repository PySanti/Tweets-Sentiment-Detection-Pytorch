import torch
from utils.convert_to_sparse_tensor import convert_to_sparse_tensor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from utils.convert_base_X import convert_base_X
from utils.load_base_data import load_base_data
import os
from utils.constants import *
import pandas as pd
import torch


def get_preprocessed_data():

    base_dataset = None

    # carga de tweets + targets
    if os.path.exists(f"./{PROCESSED_DATA_FILENAME}"):
        base_dataset = pd.read_csv(f"./{PROCESSED_DATA_FILENAME}")
    else:
        base_dataset = load_base_data()
        X = base_dataset.drop(labels=['Label'], axis=1)
        Y = base_dataset['Label'].to_numpy()
        X = convert_base_X(X)
        base_dataset = pd.DataFrame({"tweets" : pd.Series(X), "target":pd.Series(Y) })
        base_dataset.to_csv(f"./{PROCESSED_DATA_FILENAME}",index=False)

    X = base_dataset["tweets"]
    Y = base_dataset['target'].map(TAGS_MAP)

    # division en conjuntos
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=42, test_size=0.1, stratify=Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, random_state=42, test_size=0.1, stratify=Y_train)

    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    X_test = X_test.to_numpy()

    Y_train = Y_train.to_numpy()
    Y_val = Y_val.to_numpy()
    Y_test = Y_test.to_numpy()


    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


