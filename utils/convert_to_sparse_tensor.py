import numpy as np
import torch

def convert_to_sparse_tensor(sparse_matrix):
    """
        Recibe una matriz dispersa generada por una instancia
        de CountVectorizer y lo transforma en un tensor de pytorch
    """
    sparse_coo = sparse_matrix.tocoo()
    indices = torch.LongTensor(np.vstack((sparse_coo.row, sparse_coo.col)))
    values = torch.FloatTensor(sparse_coo.data)
    shape = torch.Size(sparse_coo.shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor


