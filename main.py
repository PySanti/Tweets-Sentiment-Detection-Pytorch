import torch
from utils.convert_to_sparse_tensor import convert_to_sparse_tensor
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from utils.convert_base_X import convert_base_X
from utils.load_base_data import load_base_data
import os
from utils.constants import *
import pandas as pd
from torch.utils.data import DataLoader
import torch
from sklearn.feature_extraction.text import CountVectorizer

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
Y = base_dataset['target'].map(TAGS_MAP)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=42, test_size=0.1, stratify=Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, random_state=42, test_size=0.1, stratify=Y_train)

# Solo se admitiran palabras en el corpus que aparezcan +5 veces
vectorizer = CountVectorizer(min_df=5)

X_train = convert_to_sparse_tensor(vectorizer.fit_transform(X_train))
X_val = convert_to_sparse_tensor(vectorizer.transform(X_val))
X_test = convert_to_sparse_tensor(vectorizer.transform(X_test))

Y_train = torch.tensor(Y_train.to_numpy()).unsqueeze(1)
Y_val = torch.tensor(Y_val.to_numpy()).unsqueeze(1)
Y_test = torch.tensor(Y_test.to_numpy()).unsqueeze(1)

train_dataloader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=False)
validation_dataloader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)



print(X_train[100].to_dense())
print(Y_train[100])
