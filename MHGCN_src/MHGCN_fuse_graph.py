import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch_scatter import scatter
from dgl import add_self_loop
from dgl.nn.pytorch.conv import GraphConv, SAGEConv, GATConv
from .MHGCN import GraphConvolution
from utils import knn_graph
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot

class MHGCNFuseGraph(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, out, dropout=0.5, k=5, fuse_method="mean", 
                 knn_on="graph_embed", fuse_on="node_embed", graph_pooling="mean",
                 shared=False, combine_type="attn_weight"):
        super(MHGCNFuseGraph, self).__init__()
        assert combine_type in ["attn_weight", "attn_no_weight", "elementwise", "global"]
        self.out = out
        self.k = k
        self.fuse_method = fuse_method
        self.knn_on = knn_on
        self.fuse_on = fuse_on
        self.shared = shared
        self.combine_type = combine_type
        self.layers_sc = nn.ModuleList()
        self.layers_sc.append(GraphConvolution(nfeat, nhid))
        for _ in range(nlayers - 1):
            self.layers_sc.append(GraphConvolution(nhid, nhid))
        if shared:
            self.layers_fc = self.layers_sc
        else:
            self.layers_fc = nn.ModuleList()
            self.layers_fc.append(GraphConvolution(nfeat, nhid))
            for _ in range(nlayers - 1):
                self.layers_fc.append(GraphConvolution(nhid, nhid))
        self.output = nn.Linear(nhid, out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.knn_sc, self.knn_fc = None, None
        self.simple_prim = ['mean', 'sum', 'min', 'max']
        self.gnn_dict = {
            'GCN': GraphConv,
            'SAGE': SAGEConv,
            'GAT': GATConv
        }
        if fuse_method in self.gnn_dict.keys():
            self.gnn = nn.ModuleList([])  # shared by two branches?
            configs_first = {
                "in_feats": nhid, 
                "out_feats": nhid
            }
            if fuse_method == 'GAT':
                configs_first["num_heads"] = 2
            elif fuse_method == 'SAGE':
                configs_first["aggregator_type"] = "mean"
            configs_hid = configs_first.copy()
            if fuse_method == 'GAT':
                configs_hid["in_feats"] = nhid * 2
            self.gnn.append(self.gnn_dict[fuse_method](**configs_first))
            for _ in range(nlayers-1):
                self.gnn.append(self.gnn_dict[fuse_method](**configs_hid))

        # Graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(nhid, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            self.pool = Set2Set(nhid, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.leakyrelu = nn.LeakyReLU(0.3)
        self.attention = nn.Parameter(torch.empty(size=(nhid, 1)))
        glorot(self.attention)
        if self.combine_type == 'attn_weight':
            self.w1 = torch.nn.Linear(nhid, nhid)
            self.w2 = torch.nn.Linear(nhid, nhid)
            torch.nn.init.xavier_uniform_(self.w1.weight.data)
            torch.nn.init.xavier_uniform_(self.w2.weight.data)

        if combine_type == "elementwise":
            self.combine_weights = torch.nn.Parameter(torch.FloatTensor(2, 200, 1), requires_grad=True)
            torch.nn.init.uniform_(self.combine_weights, a=0, b=0.1)
        elif combine_type == "global":
            self.combine_weights = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
            torch.nn.init.uniform_(self.combine_weights, a=0, b=0.1)

    def build_knn_graph(self, node_embed_basis, batch, null_mask):
        """
        Build kNN graph based on node embeddings.
        """
        if self.knn_on == 'graph_embed':
            embed = self.pool(node_embed_basis, batch.to(node_embed_basis.device))
        elif self.knn_on == 'node_embed':
            embed = node_embed_basis.flatten(1)

        knn_g = knn_graph(embed, null_mask, self.k, exclude_self=True)
        return knn_g

    def fusion(self, node_embed, knn_g, batch, null_mask, num_graphs, num_nodes):
        """
        Fuse node embeddings using the kNN graph.
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
                for layer in self.gnn[:-1]:
                    embed = self.dropout(F.relu(layer(knn_g, embed)))
                    if self.fuse_method == 'GAT':
                        embed = embed.flatten(-2)
                embed = self.dropout(F.relu(self.gnn[-1](knn_g, embed)))
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

        if self.combine_type == "attn_weight":
            geom_ = self.leakyrelu(torch.mm(self.w1(embed1), self.attention))
            cell_type_ = self.leakyrelu(torch.mm(self.w2(embed2), self.attention))
        elif self.combine_type == "attn_no_weight":
            geom_ = self.leakyrelu(torch.mm(embed1, self.attention))
            cell_type_ = self.leakyrelu(torch.mm(embed2, self.attention))

        if self.combine_type in ["attn_weight", "attn_no_weight"]:
            values = torch.softmax(torch.cat((geom_, cell_type_), dim=1), dim=1)
            embed = (values[:,0].unsqueeze(1) * embed1) + (values[:,1].unsqueeze(1) * embed2)
        elif self.combine_type == "global":
            embed = (embed1 * self.combine_weights[0]) + (embed2 * self.combine_weights[1])
        elif self.combine_type == "elementwise":
            embed1 = embed1.reshape(num_graphs, num_nodes, -1)
            embed2 = embed2.reshape(num_graphs, num_nodes, -1)
            embed = (embed1 * self.combine_weights[0]) + (embed2 * self.combine_weights[1])
            embed = embed.reshape(-1, embed.shape[-1])
        if self.fuse_on == 'node_embed':
            embed = self.pool(embed, batch.to(embed.device))

        return embed

    def forward(self, A_batch: torch.Tensor, feature: torch.Tensor, no_sc_idx, no_fc_idx):
        # TODO: try 200 * 200 * 2 weight matrix for each edge item rather than 2 * 1
        A_batch_sc = A_batch[:, 0, :, :]
        A_batch_fc = A_batch[:, 1, :, :]
        
        embeds_sc = []
        embeds_fc = []
        feature_sc, feature_fc = feature, feature
        for layer_sc, layer_fc in zip(self.layers_sc, self.layers_fc):
            feature_sc = self.relu(self.dropout(layer_sc(feature_sc, A_batch_sc)))
            feature_fc = self.relu(self.dropout(layer_fc(feature_fc, A_batch_fc)))
            embeds_sc.append(feature_sc)
            embeds_fc.append(feature_fc)
        
        embeds_sc = torch.stack(embeds_sc, dim=0).mean(0)
        embeds_sc = embeds_sc.reshape(-1, embeds_sc.shape[-1])
        embeds_fc = torch.stack(embeds_fc, dim=0).mean(0)
        embeds_fc = embeds_fc.reshape(-1, embeds_fc.shape[-1])
        
        # Build kNN graph and fuse node embeddings
        batch = torch.repeat_interleave(torch.arange(feature.shape[0]), feature.shape[-2]).to(feature.device)
        num_graphs, num_nodes = feature.shape[0], feature.shape[1]

        knn_g = self.build_knn_graph(node_embed_basis=embeds_fc, batch=batch, null_mask=no_sc_idx)
        embed1 = self.fusion(embeds_sc, knn_g, batch, no_sc_idx, num_graphs, num_nodes)
        self.knn_sc = knn_g

        knn_g = self.build_knn_graph(node_embed_basis=embeds_sc, batch=batch, null_mask=no_fc_idx)
        embed2 = self.fusion(embeds_fc, knn_g, batch, no_fc_idx, num_graphs, num_nodes)
        self.knn_fc = knn_g

        graph_embed = self.gen_graph_embed(embed1, embed2, batch, num_graphs, num_nodes)

        embeds = self.output(graph_embed)

        if self.out == 1:  # regression task
            embeds = embeds.squeeze()
        return embeds