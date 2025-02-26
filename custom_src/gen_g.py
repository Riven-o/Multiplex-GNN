import numpy as np
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Batch
from torch_geometric.transforms import SIGN

class GenG(nn.Module):
    def __init__(self, no_sc_idx, no_fc_idx):
        super(GenG, self).__init__()
        self.no_sc_idx, self.no_fc_idx = \
            no_sc_idx, no_fc_idx
        
    def forward(self, adjs):
        """
        data: num_graphs, num_graph_layers, num_nodes, num_nodes
        """
        pass
        
