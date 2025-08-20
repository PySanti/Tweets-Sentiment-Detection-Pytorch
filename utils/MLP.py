import torch
from utils.constants import VOCAB_SIZE


class MLP(torch.nn.Module):
    def __init__(self, embed_dim, hidden_sizes, out_size, drop_rate):
        super(MLP,self).__init__()
        self.layers = torch.nn.ModuleList()

        self.embed = torch.nn.Embedding(VOCAB_SIZE, embed_dim,padding_idx=0)
        self.embed_norm = torch.nn.BatchNorm1d(embed_dim)
        current_size = embed_dim
        for layer in hidden_sizes:
            self.layers.append(torch.nn.Linear(current_size, layer))
            self.layers.append(torch.nn.BatchNorm1d(layer))
            self.layers.append(torch.nn.LeakyReLU())
            self.layers.append(torch.nn.Dropout(p=drop_rate))
            current_size = layer

        self.layers.append(torch.nn.Linear(current_size, out_size))

    def forward(self, X):
        out = self.embed(X) # (batch, seq, embed)
        masked = out * (X != 0).unsqueeze(-1).float() 
        pooled = masked.mean(dim=1) # (batch, embed)
        out = pooled
        out = self.embed_norm(out)
        for layer in self.layers:
            out = layer(out)
        return out

    def _init_weights(self):
        pass
