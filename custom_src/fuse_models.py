import torch
from torch import nn

from typing import Optional, Tuple, Union

from torch_geometric.nn import GCNConv, SAGEConv, SGConv, GATConv, GINConv, MessagePassing
from torch_scatter import scatter_mean, scatter_max, scatter

import torch.nn.functional as F

class FuseBase(nn.Module):
    def __init__(self, model_name, in_dim, hid_dim, nlayers, 
                dropout, nclass, 
                reduce_nodes, reduce_fuse) -> None:
        super().__init__()

        if 'fuse_embed' in model_name:
            self.fuse_type = 'fuse_embed'
            assert reduce_fuse in ['mean', 'concat', 'sum']
        elif 'fuse_graph' in model_name:
            self.fuse_type = 'fuse_graph'
            assert reduce_fuse in ['mean', 'or', 'and', 'sum']
        if 'fuse_pred' in model_name:
            self.fuse_type = 'fuse_pred'
            assert reduce_fuse in ['mean', 'sum']

        self.dropout = dropout
        self.reduce_nodes = reduce_nodes  # Control reduction method
        self.reduce_fuse = reduce_fuse

        if self.fuse_type == 'fuse_embed' and self.reduce_fuse == 'concat':
            self.output = nn.Linear(hid_dim * 2, nclass)
        else:
            self.output = nn.Linear(hid_dim, nclass)

    def forward_fuse_graph(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = net(x, edge_index)
            else:
                x = net(x)
        x = self.output(x)
        return scatter(x, batch.batch, dim=0, reduce=self.reduce_nodes)

    def forward_fuse_embed(self, batch):
        x_0, x_1 = batch.x, batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x_0 = net(x_0, batch.edge_index_sc)
                x_1 = net(x_1, batch.edge_index_fc)
            else:
                x_0 = net(x_0)
                x_1 = net(x_1)
        if self.reduce_fuse == 'mean':
            x = (x_0 + x_1) / 2
        elif self.reduce_fuse == 'concat':
            x = torch.concat([x_0, x_1], dim=-1)
        elif self.reduce_fuse == 'sum':
            x = x_0 + x_1
        x = self.output(x)
        return scatter(x, batch.batch, dim=0, reduce=self.reduce_nodes)
    
    def forward_fuse_pred(self, batch):
        x_0, x_1 = batch.x, batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x_0 = net(x_0, batch.edge_index_sc)
                x_1 = net(x_1, batch.edge_index_fc)
            else:
                x_0 = net(x_0)
                x_1 = net(x_1)
        x_0 = self.output(x_0)
        x_1 = self.output(x_1)
        
        x_0 = scatter(x_0, batch.batch, dim=0, reduce=self.reduce_nodes)
        x_1 = scatter(x_1, batch.batch, dim=0, reduce=self.reduce_nodes)

        if self.reduce_fuse == 'mean':
            x = (x_0 + x_1) / 2
        elif self.reduce_fuse == 'sum':
            x = x_0 + x_1
        
        return x
    
    def forward(self, *args):
        if self.fuse_type == 'fuse_graph':
            return self.forward_fuse_graph(*args)
        elif self.fuse_type == 'fuse_embed':
            return self.forward_fuse_embed(*args)
        elif self.fuse_type == 'fuse_pred':
            return self.forward_fuse_pred(*args)

class VanillaFuse(FuseBase):
    # siamese network for now
    def __init__(self, model_name, in_dim, hid_dim, nlayers, 
                dropout, nclass, 
                reduce_nodes, reduce_fuse) -> None:
        super().__init__(model_name, in_dim, hid_dim, nlayers, 
                         dropout, nclass, 
                         reduce_nodes, reduce_fuse)
        

        if 'GCN' in model_name:
            model_class = GCNConv
        elif 'SAGE' in model_name:
            model_class = SAGEConv
        elif 'SGC' in model_name:
            model_class = SGConv
        elif 'GIN' in model_name:  # not work yet
            model_class = GINConv

        self.net = nn.ModuleList()

        # Input layer
        self.net.append(model_class(in_dim, hid_dim))
        self.net.append(nn.BatchNorm1d(hid_dim))
        self.net.append(nn.ReLU())
        self.net.append(nn.Dropout(p=dropout))

        # Hidden layers
        for _ in range(nlayers - 1):
            self.net.append(model_class(hid_dim, hid_dim))
            self.net.append(nn.BatchNorm1d(hid_dim))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(p=dropout))

        self.net.append(model_class(hid_dim, hid_dim))

class GATFuse(FuseBase):
    def __init__(self, model_name, in_dim, hid_dim, nlayers, 
                dropout, nclass, 
                reduce_nodes, reduce_fuse) -> None:
        super().__init__(model_name, in_dim, hid_dim, nlayers, 
                         dropout, nclass, 
                         reduce_nodes, reduce_fuse)
        
        self.net = nn.ModuleList()

        # Input layer
        self.net.append(GATConv(in_dim, hid_dim, heads=2))
        self.net.append(nn.BatchNorm1d(hid_dim * 2))
        self.net.append(nn.ReLU())
        self.net.append(nn.Dropout(p=dropout))

        # Hidden layers
        for _ in range(nlayers - 1):
            self.net.append(GATConv(hid_dim * 2, hid_dim, heads=2))
            self.net.append(nn.BatchNorm1d(hid_dim * 2))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(p=dropout))

        # Final layer to hidden dimension
        self.net.append(GATConv(hid_dim * 2, hid_dim))

        