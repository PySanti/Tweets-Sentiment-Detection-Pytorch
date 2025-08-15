import torch


class MLP(torch.nn.Module):
    def __init__(self, hidden_sizes, input_shape, out_size):
        super(MLP,self).__init__()
        self.layers = torch.nn.ModuleList()

        current_size = input_shape
        for layer in hidden_sizes:
            self.layers.append(torch.nn.Linear(current_size, layer))
            self.layers.append(torch.nn.ReLU())
            # normalizacion
            # dropout
            current_size = layer

        self.layers.append(torch.nn.Linear(current_size, out_size))

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def _init_weights(self):
        pass
