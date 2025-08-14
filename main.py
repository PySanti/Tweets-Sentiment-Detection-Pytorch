import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("tariqsays/sentiment-dataset-with-1-million-tweets")
x = pd.DataFrame(pd.read_csv(path + "/dataset.csv"))
print(x.shape)
print(x.iloc[0])

