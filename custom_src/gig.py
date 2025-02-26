import numpy as np
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Batch
from torch_geometric.transforms import SIGN
from utils import model_infer

class GIG(nn.Module):
    def __init__(self, gnn, gnn_name, in_feats, hidden, out_feats, n_layers, n_head=2, dropout=0.5):
        super(GIG, self).__init__()
        self.n_layers = n_layers
        self.gnn = gnn
        self.gnn_name = gnn_name
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_feats, nhead=n_head, 
                                       dim_feedforward=hidden, dropout=dropout, 
                                       batch_first=True),
            num_layers=n_layers
        )
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.transformer:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, **kwargs):
        """
        x: (num_graphs, num_feats)
        """
        x = model_infer(model=self.gnn, model_name=self.gnn_name, **kwargs)
        res = self.transformer(x)
        return res