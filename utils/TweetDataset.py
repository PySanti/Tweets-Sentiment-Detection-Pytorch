import torch
from torch.utils.data import Dataset
from utils.encode_text import encode_text
from utils.constants import MAX_LEN



class TweetDataset(Dataset):

    def __init__(self, tweets, labels, tokenizer):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        text = self.tweets[idx]
        label = self.labels[idx]
        tokens = encode_text(text,self.tokenizer, MAX_LEN)
        tokens = tokens + [0] * (MAX_LEN - len(tokens)) # padding
        return torch.tensor(tokens), torch.tensor(label)


