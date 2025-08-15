from sklearn.model_selection import train_test_split
from utils.convert_base_X import convert_base_X
from utils.load_base_data import load_base_data
import os
from utils.constants import *
import pandas as pd

from utils.show_dataset_info import show_dataset_info


base_dataset = None
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
Y = base_dataset['target']

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=42, test_size=0.1, stratify=Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, random_state=42, test_size=0.1, stratify=Y_train)

show_dataset_info(X_train, Y_train, "Train")
show_dataset_info(X_val, Y_val, "Val")
show_dataset_info(X_test, Y_test, "Test")
