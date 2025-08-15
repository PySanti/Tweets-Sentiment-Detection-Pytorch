import kagglehub
import pandas as pd

def load_base_data():
    # Download latest version
    path = kagglehub.dataset_download("tariqsays/sentiment-dataset-with-1-million-tweets")
    base_dataset = pd.DataFrame(pd.read_csv(path + "/dataset.csv"))
    return base_dataset



