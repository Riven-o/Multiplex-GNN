import torch
from torch import nn

from typing import Optional, Tuple, Union
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing, GCNConv


from .vanilla_models import Classifier

from utils import device

class DetourTransformer(nn.Module):

    def __init__(self, 
                nclass,
                heads: int = 2,
                nlayers: int = 1,
                num_nodes: int=116,
                in_dim: Union[int, Tuple[int, int]] = 10,
                dropout: float = 0.1,
                hid_dim: int = 1024,
                detour_type = 'node',
                ) -> None:
        
        super(DetourTransformer, self).__init__()
        org_in_dim = in_dim
        if in_dim % heads != 0:
            in_dim = in_dim  + heads - (in_dim % heads)
        self.detour_type = detour_type
        self.nlayers = nlayers
        self.num_nodes = num_nodes

        # self.batchnorm = nn.BatchNorm1d(org_in_dim)
            
        self.lin_first = nn.Sequential(
            nn.Linear(org_in_dim, in_dim), 
            nn.BatchNorm1d(in_dim), 
            nn.LeakyReLU()
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_dim, hid_dim), 
            nn.BatchNorm1d(hid_dim), 
            nn.LeakyReLU(),
        )
        
        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=heads, dim_feedforward=hid_dim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None
        ) for _ in range(nlayers)])
        self.net_fc = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=heads, dim_feedforward=hid_dim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None
        ) for _ in range(nlayers)])
        self.heads = heads
        self.in_dim = in_dim
        self.fcsc_loss = nn.MSELoss()

        self.classifier = Classifier(GCNConv, hid_dim, nclass, num_nodes)

    def forward(self, data):
        node_feature = data.x
        # node_feature = self.batchnorm(node_feature)
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), self.in_dim)

        adj = data.adj_sc
        adj_fc = data.adj_fc
        org_adj = adj
        multi_mask = []
        for _ in range(self.heads):
            # mask = torch.zeros(len(adj), adj.shape[1], adj.shape[2], device=adj.device) - torch.inf
            # mask[torch.logical_and(adj, adj_fc)] = 0
            
            # TODO(jiahang): numeric error, weird. Codes are different from paper's descriptions
            mask = torch.zeros(len(adj), adj.shape[1], adj.shape[2], device=adj.device)
            mask[torch.logical_and(adj, adj_fc)] = 1
            adj = (adj.float() @ org_adj.float()) > 0
            multi_mask.append(mask)
        multi_mask = torch.cat(multi_mask)
        
        for i in range(self.nlayers):
            node_feature = self.net[i](node_feature, mask=multi_mask)
        
        # return self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_dim))
        h = self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_dim))
        return self.classifier(h, data.edge_index, data.batch).flatten()


class DetourTransformerSingleFC(nn.Module):

    def __init__(self, 
        heads: int = 2,
        nlayers: int = 1,
        num_nodes: int=116,
        in_dim: Union[int, Tuple[int, int]] = 10,
        out_channel: int = 10,
        concat: bool = False,
        dek: int = 4,
        pek: int = 10,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        hid_dim: int = 1024,
        detour_type = 'node',
        batch_size = 32,
        device='cuda:0',
        *args, **kwargs) -> None:
        
        super(DetourTransformerSingleFC, self).__init__()
        org_in_dim = in_dim
        if in_dim % heads != 0:
            in_dim = in_dim  + heads - (in_dim % heads)
        self.detour_type = detour_type
        self.nlayers = nlayers
        self.num_nodes = num_nodes
            
        self.lin_first = nn.Sequential(
            nn.Linear(org_in_dim, in_dim), 
            nn.BatchNorm1d(in_dim), 
            nn.LeakyReLU()
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_dim, out_channel), 
            nn.BatchNorm1d(out_channel), 
            nn.LeakyReLU(),
        )
        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=heads, dim_feedforward=hid_dim, dropout=dropout, batch_first=True),
            nlayers=1,
            norm=None#nn.LayerNorm(in_dim)
        ) for _ in range(nlayers)])

        self.heads = heads
        self.in_dim = in_dim
        self.out_channel = out_channel
        self.mask_heldout = torch.zeros(batch_size, num_nodes, num_nodes) - torch.inf
        self.mask_heldout = self.mask_heldout.to(device)

    def forward(self, data):
        self.loss = 0
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), self.in_dim)

        adj_fc = data.adj_fc
        adj_fc[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = True
        mask_fc = self.mask_heldout[:len(adj_fc)]
        mask_fc[adj_fc] = 0
        mask_fc = mask_fc.repeat(self.heads, 1, 1)
        for i in range(self.nlayers):
            node_feature = self.net[i](node_feature, mask=mask_fc)

        return self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_dim))



class DetourTransformerSingleSC(nn.Module):

    def __init__(self, 
        heads: int = 2,
        nlayers: int = 1,
        num_nodes: int=116,
        in_dim: Union[int, Tuple[int, int]] = 10,
        out_channel: int = 10,
        concat: bool = False,
        dek: int = 4,
        pek: int = 10,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        hid_dim: int = 1024,
        detour_type = 'node',
        batch_size = 32,
        device='cuda:0',
        *args, **kwargs) -> None:
        
        super(DetourTransformerSingleSC, self).__init__()
        org_in_dim = in_dim
        if in_dim % heads != 0:
            in_dim = in_dim  + heads - (in_dim % heads)
        self.detour_type = detour_type
        self.nlayers = nlayers
        self.num_nodes = num_nodes
            
        self.lin_first = nn.Sequential(
            nn.Linear(org_in_dim, in_dim), 
            nn.BatchNorm1d(in_dim), 
            nn.LeakyReLU()
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_dim, out_channel), 
            nn.BatchNorm1d(out_channel), 
            nn.LeakyReLU(),
        )

        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=heads, dim_feedforward=hid_dim, dropout=dropout, batch_first=True),
            nlayers=1,
            norm=None#nn.LayerNorm(in_dim)
        ) for _ in range(nlayers)])
        self.heads = heads
        self.in_dim = in_dim
        self.out_channel = out_channel
        self.mask_heldout = torch.zeros(batch_size, num_nodes, num_nodes) - torch.inf
        self.mask_heldout = self.mask_heldout.to(device)


    def forward(self, data):
        self.loss = 0
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), self.in_dim)

        adj = data.adj_sc
        adj_fc = data.adj_fc
        adj[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = True
        adj_fc[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = True
        org_adj = adj
        multi_mask = []
        for _ in range(self.heads):
            mask = self.mask_heldout[:len(adj)]
            mask[torch.logical_and(adj, adj_fc)] = 0
            adj = (adj.float() @ org_adj.float()) > 0
            multi_mask.append(mask)
        multi_mask = torch.cat(multi_mask)

        for i in range(self.nlayers):
            node_feature = self.net[i](node_feature, mask=multi_mask)

        return self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_dim))

    