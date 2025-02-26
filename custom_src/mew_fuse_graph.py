import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np

from .mew import SIGN_pred, SIGN_v2

from dgl import DGLError, remove_self_loop, remove_edges, EID, add_self_loop
from dgl.base import dgl_warning
from dgl.transforms.functional import pairwise_squared_distance
from dgl.sampling import sample_neighbors
from dgl.transforms.functional import convert
import dgl.backend as dglF
from dgl.nn.pytorch.conv import GraphConv, SAGEConv, SGConv, GATConv
from .mew_custom import SIGNv2Custom, knn_graph

class MewFuseGraph(SIGN_pred):
    def __init__(self, num_layer, num_feat, emb_dim, drop_ratio, shared, 
                 k=None, knn_on="graph_embed", fuse_on='node_embed', fuse_method="mean",
                 gnn_add_self_loop=False, null_filter=True, fusion_only_on_null=False,
                 *args, **kwargs):
        super().__init__(num_layer, num_feat, emb_dim, drop_ratio=drop_ratio, shared=shared, *args, **kwargs)
        self.k = k
        self.sign = SIGNv2Custom(num_layer, num_feat, emb_dim, dropout=drop_ratio)
        self.sign2 = self.sign if shared else SIGNv2Custom(num_layer, num_feat, emb_dim, dropout=drop_ratio)
        self.knn_on = knn_on
        self.fuse_on = fuse_on
        self.fuse_method = fuse_method
        self.gnn_add_self_loop = gnn_add_self_loop
        self.null_filter = null_filter
        self.dropout = nn.Dropout(drop_ratio)

        self.knn_sc, self.knn_fc = None, None
        self.fusion_only_on_null = fusion_only_on_null

        self.simple_prim = ['mean', 'sum', 'min', 'max']
        self.gnn_dict = {
            'GCN': GraphConv,
            'SAGE': SAGEConv,
            'GAT': GATConv
        }
        if fuse_method in self.gnn_dict.keys():
            self.gnn = nn.ModuleList([]) # shared by two branches?

            configs_first = {
                "in_feats": emb_dim, 
                "out_feats": emb_dim
            }
            if fuse_method == 'GAT':
                configs_first["num_heads"] = 2
            elif fuse_method == 'SAGE':
                configs_first["aggregator_type"] = "mean"
            
            configs_hid = configs_first.copy()
            if fuse_method == 'GAT':
                configs_hid["in_feats"] = emb_dim * 2

            self.gnn.append(
                    self.gnn_dict[fuse_method](**configs_first)
                )  
            for _ in range(num_layer-1):
                self.gnn.append(
                    self.gnn_dict[fuse_method](**configs_hid)
                ) 
            
    def build_knn_graph(self, node_embed_basis, batch, null_mask):
        """
        for single branch
        input: node embedding to build the basis of knn graph, obtain by SIGN
        output: knn graph
        """
        
        if self.knn_on == 'graph_embed': # knn_base == graph_embed
            embed = self.pool(node_embed_basis, batch.to(node_embed_basis.device))
        elif self.knn_on == 'node_embed':
            embed = node_embed_basis.flatten(1)

        knn_g = knn_graph(embed, null_mask, self.k, exclude_self=True, null_filter=self.null_filter)
        
        return knn_g
    
    def fusion(self, node_embed, knn_g, batch, null_mask, num_graphs, num_nodes):
        """
        for single branch
        input: node embedding for fusion, obtain by SIGN
        output: node or graph embedding (if fusing graph embedding)
        """
        
        if self.fuse_on == 'graph_embed':
            embed = self.pool(node_embed, batch.to(node_embed.device))
        elif self.fuse_on == 'node_embed':
            embed = node_embed.reshape(num_graphs, num_nodes, -1)

        if null_mask.sum() > 0:
            if self.fuse_method in self.simple_prim:
                knn_n, target_n = knn_g.in_edges(v=torch.where(null_mask)[0])
                embed[null_mask] = scatter(embed[knn_n], target_n, 
                                        reduce=self.fuse_method, dim=0, 
                                        dim_size=embed.shape[0])[null_mask]
                
            elif self.fuse_method in self.gnn_dict.keys():
                knn_g = add_self_loop(knn_g) if self.gnn_add_self_loop else knn_g
                for layer in self.gnn[:-1]:
                    embed = self.dropout(
                         F.relu(layer(knn_g, embed))
                    )
                    if self.fuse_method == 'GAT':
                        embed = embed.flatten(-2)
                embed = self.dropout(
                            F.relu(self.gnn[-1](knn_g, embed))
                        )
                if self.fuse_method == 'GAT':
                    embed = embed.mean(-2)

        return embed
    
    def gen_graph_embed(self, embed1, embed2, batch, num_graphs, num_nodes):
        """
        combining two branches
        input: node or graph embedding
        output: graph embedding
        """
        if self.fuse_on == 'node_embed':
            embed1 = embed1.reshape(num_graphs * num_nodes, -1)
            embed2 = embed2.reshape(num_graphs * num_nodes, -1)

        if self.attn_weight:
            geom_ = self.leakyrelu(torch.mm(self.w1(embed1), self.attention))
            cell_type_ = self.leakyrelu(torch.mm(self.w2(embed2), self.attention))
        else:
            geom_ = self.leakyrelu(torch.mm(embed1, self.attention))
            cell_type_ = self.leakyrelu(torch.mm(embed2, self.attention))

        values = torch.softmax(torch.cat((geom_, cell_type_), dim=1), dim=1)
        embed = (values[:,0].unsqueeze(1) * embed1) + (values[:,1].unsqueeze(1) * embed2)
        if self.fuse_on == 'node_embed':
            embed = self.pool(embed, batch.to(embed.device))

        return embed

    def forward(self, adjs, feats, no_sc_idx, no_fc_idx):
        """
        fuse_type:
        graph_embed: knn on graph embed, fuse graph embed (mean)
        node_embed_on_graph_embed: knn on graph embed, fuse node embed (mean)
        0_miss: 0 for the missing graph layer
        unit_miss: unit adjacent matrix for the missing graph layer
        baseline (Mew no fuse graph): 0 adjacent matrix for the missing graph layer, use Mew not MewCustom
            note that baseline is different from 0_miss
        """
        batch = torch.repeat_interleave(
            torch.arange(feats.shape[0]), 
            feats.shape[-2]
        ).to(feats.device)
        num_graphs, num_nodes = feats.shape[0], feats.shape[1]

        node_embed_1 = self.sign(adjs[:, 0], feats) # geom
        node_embed_2 = self.sign2(adjs[:, 1], feats) # cell_type

        knn_g = self.build_knn_graph(node_embed_basis=node_embed_2, batch=batch, null_mask=no_sc_idx)
        _embed1 = self.fusion(node_embed_1, knn_g, batch, no_sc_idx, num_graphs, num_nodes)
        if self.fusion_only_on_null:
            _node_embed_1 = node_embed_1.reshape(num_graphs, num_nodes, -1)
            embed1 = torch.zeros_like(_node_embed_1)
            embed1[~no_sc_idx] = _node_embed_1[~no_sc_idx]
            embed1[no_sc_idx] = _embed1[no_sc_idx]
        else:
            embed1 = _embed1
        self.knn_sc = knn_g
        
        knn_g = self.build_knn_graph(node_embed_basis=node_embed_1, batch=batch, null_mask=no_fc_idx)
        _embed2 = self.fusion(node_embed_2, knn_g, batch, no_fc_idx, num_graphs, num_nodes)
        if self.fusion_only_on_null:
            _node_embed_2 = node_embed_2.reshape(num_graphs, num_nodes, -1)
            embed2 = torch.zeros_like(_node_embed_2)
            embed2[~no_fc_idx] = _node_embed_2[~no_fc_idx]
            embed2[no_fc_idx] = _embed2[no_fc_idx]
        else:
            embed2 = _embed2
        self.knn_fc = knn_g

        graph_embed = self.gen_graph_embed(embed1, embed2, batch, num_graphs, num_nodes)
        
        if self.num_graph_tasks > 0:
            graph_pred = self.graph_pred_module(graph_embed)

        return graph_pred

