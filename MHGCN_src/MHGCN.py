import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from .MHGCN_utils import adj_matrix_weight_merge


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class MHGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, out, dropout = 0.5):
        super(MHGCN, self).__init__()

        self.out = out
        
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(nfeat, nhid))
        for _ in range(nlayers - 1):
            self.layers.append(GraphConvolution(nhid, nhid))
        self.output = nn.Linear(nhid, out)
       
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

    def forward(self, A_batch: torch.Tensor, feature: torch.Tensor):
        A_batch = A_batch.permute(0, 2, 3, 1)
        final_A = (A_batch @ self.weight_b).squeeze()
        embeds = []
        for layer in self.layers:
            feature = self.relu(self.dropout(layer(feature, final_A)))
            embeds.append(feature)
        embeds = torch.stack(embeds, dim=0).mean(0)
        embeds = embeds.mean(dim=1)
        embeds = self.output(embeds)

        if self.out == 1: # regression task
            embeds.squeeze()
        # Average pooling
        return embeds