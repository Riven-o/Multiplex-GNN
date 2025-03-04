import numpy as np
import torch
from scipy.sparse import coo_matrix


def coototensor(A):
    """
    Convert a coo_matrix to a torch sparse tensor
    """
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def adj_matrix_weight_merge(A, adj_weight):
    """
    Multiplex Relation Aggregation
    """

    N = A[0][0].shape[0]
    temp = coo_matrix((N, N))
    temp = coototensor(temp)


    a = coototensor(A[0][0].tocoo())
    b = coototensor(A[0][1].tocoo())
    c = coototensor(A[0][2].tocoo())
    A_t = torch.stack([a, b, c], dim=2).to_dense()

    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp, 2)

    return temp + temp.transpose(0, 1)