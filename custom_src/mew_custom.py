import torch
from torch import nn

from torch_scatter import scatter
import numpy as np

from .mew import SIGN_pred, SIGN_v2
from utils import knn_graph


class SIGNv2Custom(SIGN_v2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, adjs: torch.Tensor, feats: torch.Tensor):
        """
        Different from original sign v2:
        original sign v2: given [X, AX, A2X, A3X], apply NN to each, concate, NN transform then output
        current sign v2: given X, apply adj @ X @ NN layer by layer, concate, NN transform then output
        in other words, current sign v2 is online version of original sign v2
        """
        hidden = []
        for i, ff in enumerate(self.inception_ffs):
            feats = adjs.matmul(feats)
            h = feats.reshape(feats.shape[0] * feats.shape[1], -1)
            h = ff(h)
            hidden.append(self.batch_norms[i](h))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out
        
class MewCustom(SIGN_pred):
    def __init__(self, num_layer, num_feat, emb_dim, drop_ratio, shared, 
                 k=None, supp_pool="mean", fuse_type='graph_embed',
                 *args, **kwargs):
        super().__init__(num_layer, num_feat, emb_dim, drop_ratio=drop_ratio, shared=shared, *args, **kwargs)
        self.k = k
        self.fuse_type = fuse_type
        self.sign = SIGNv2Custom(num_layer, num_feat, emb_dim, dropout=drop_ratio)
        self.sign2 = self.sign if shared else SIGNv2Custom(num_layer, num_feat, emb_dim, dropout=drop_ratio)

        self.supp_pool = supp_pool

    def fuse_graph_embed(self, node_embed_1, node_embed_2, batch, no_sc_idx, no_fc_idx):
        graph_embed_1 = self.pool(node_embed_1, batch.to(node_embed_1.device))
        graph_embed_2 = self.pool(node_embed_2, batch.to(node_embed_2.device)) 

        if no_sc_idx.sum() > 0:
            knn_g = knn_graph(graph_embed_2, no_sc_idx, self.k, exclude_self=True) # must be graph_embed_2 not embed_1
            knn_n, target_n = knn_g.in_edges(v=torch.where(no_sc_idx)[0])
            graph_embed_1[no_sc_idx] = scatter(graph_embed_1[knn_n], target_n, 
                                            reduce=self.supp_pool, dim=0, 
                                            dim_size=graph_embed_1.shape[0])[no_sc_idx]
        if no_fc_idx.sum() > 0:
            knn_g = knn_graph(graph_embed_1, no_fc_idx, self.k, exclude_self=True)  # must be graph_embed_1 not embed_2
            knn_n, target_n = knn_g.in_edges(v=torch.where(no_fc_idx)[0])
            graph_embed_2[no_fc_idx] = scatter(graph_embed_2[knn_n], target_n, 
                                                reduce=self.supp_pool, dim=0, 
                                                dim_size=graph_embed_2.shape[0])[no_fc_idx]
            
        if self.attn_weight:
            geom_ = self.leakyrelu(torch.mm(self.w1(graph_embed_1), self.attention))
            cell_type_ = self.leakyrelu(torch.mm(self.w2(graph_embed_2), self.attention))
        else:
            geom_ = self.leakyrelu(torch.mm(graph_embed_1, self.attention))
            cell_type_ = self.leakyrelu(torch.mm(graph_embed_2, self.attention))

        values = torch.softmax(torch.cat((geom_, cell_type_), dim=1), dim=1)
        graph_embed = (values[:,0].unsqueeze(1) * graph_embed_1) + (values[:,1].unsqueeze(1) * graph_embed_2)

        return graph_embed
    
    def fuse_node_embed_on_graph_embed(self, node_embed_1, node_embed_2, 
                                          batch, no_sc_idx, no_fc_idx, 
                                          num_graphs, num_nodes):

        graph_embed_1 = self.pool(node_embed_1, batch.to(node_embed_1.device))
        graph_embed_2 = self.pool(node_embed_2, batch.to(node_embed_2.device))

        node_embed_1 = node_embed_1.reshape(num_graphs, num_nodes, -1)
        node_embed_2 = node_embed_2.reshape(num_graphs, num_nodes, -1)

        if no_sc_idx.sum() > 0:
            knn_g = knn_graph(graph_embed_2, no_sc_idx, self.k, exclude_self=True) # must be graph_embed_2 not embed_1
            knn_n, target_n = knn_g.in_edges(v=torch.where(no_sc_idx)[0])
            node_embed_1[no_sc_idx] = scatter(node_embed_1[knn_n], target_n, 
                                            reduce=self.supp_pool, dim=0, 
                                            dim_size=node_embed_1.shape[0])[no_sc_idx]
        if no_fc_idx.sum() > 0:
            knn_g = knn_graph(graph_embed_1, no_fc_idx, self.k, exclude_self=True)  # must be graph_embed_1 not embed_2
            knn_n, target_n = knn_g.in_edges(v=torch.where(no_fc_idx)[0])
            node_embed_2[no_fc_idx] = scatter(node_embed_2[knn_n], target_n, 
                                            reduce=self.supp_pool, dim=0, 
                                            dim_size=node_embed_2.shape[0])[no_fc_idx]
            
        node_embed_1 = node_embed_1.reshape(num_graphs * num_nodes, -1)
        node_embed_2 = node_embed_2.reshape(num_graphs * num_nodes, -1)
            
        if self.attn_weight:
            geom_ = self.leakyrelu(torch.mm(self.w1(node_embed_1), self.attention))
            cell_type_ = self.leakyrelu(torch.mm(self.w2(node_embed_2), self.attention))
        else:
            geom_ = self.leakyrelu(torch.mm(node_embed_1, self.attention))
            cell_type_ = self.leakyrelu(torch.mm(node_embed_2, self.attention))
        
        values = torch.softmax(torch.cat((geom_, cell_type_), dim=1), dim=1)
        node_embed = (values[:,0].unsqueeze(1) * node_embed_1) + (values[:,1].unsqueeze(1) * node_embed_2)
        graph_embed = self.pool(node_embed, batch.to(node_embed_1.device))

        return graph_embed
    
    def forward_0_miss(self, node_embed_1, node_embed_2, 
                       batch, no_sc_idx, no_fc_idx, 
                       num_graphs, num_nodes):
        if self.attn_weight:
            geom_ = self.leakyrelu(torch.mm(self.w1(node_embed_1), self.attention))
            cell_type_ = self.leakyrelu(torch.mm(self.w2(node_embed_2), self.attention))
        else:
            geom_ = self.leakyrelu(torch.mm(node_embed_1, self.attention))
            cell_type_ = self.leakyrelu(torch.mm(node_embed_2, self.attention))
        values = torch.softmax(torch.cat((geom_, cell_type_), dim=1), dim=1)

        node_embed = torch.zeros_like(node_embed_1)
        if no_sc_idx.sum() > 0:
            null_mask = torch.repeat_interleave(no_sc_idx, num_nodes)
            node_embed[~null_mask] = \
                (values[~null_mask,0].unsqueeze(1) * node_embed_1[~null_mask]) + \
                (values[~null_mask,1].unsqueeze(1) * node_embed_2[~null_mask])
            node_embed[null_mask] = node_embed_2[null_mask]
        if no_fc_idx.sum() > 0:
            null_mask = torch.repeat_interleave(no_fc_idx, num_nodes)
            node_embed[~null_mask] = \
                (values[~null_mask,0].unsqueeze(1) * node_embed_1[~null_mask]) + \
                (values[~null_mask,1].unsqueeze(1) * node_embed_2[~null_mask])
            node_embed[null_mask] = node_embed_1[null_mask]

        graph_embed = self.pool(node_embed, batch.to(node_embed_1.device))

        return graph_embed
    
    def forward_normal(self, node_embed_1, node_embed_2, batch):
        if self.attn_weight:
            geom_ = self.leakyrelu(torch.mm(self.w1(node_embed_1), self.attention))
            cell_type_ = self.leakyrelu(torch.mm(self.w2(node_embed_2), self.attention))
        else:
            geom_ = self.leakyrelu(torch.mm(node_embed_1, self.attention))
            cell_type_ = self.leakyrelu(torch.mm(node_embed_2, self.attention))

        values = torch.softmax(torch.cat((geom_, cell_type_), dim=1), dim=1)
        node_embed = (values[:,0].unsqueeze(1) * node_embed_1) + (values[:,1].unsqueeze(1) * node_embed_2)
        graph_embed = self.pool(node_embed, batch.to(node_embed_1.device))

        return graph_embed

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
        if self.fuse_type == 'unit_miss':
            #TODO[jiahang]: there are many matrices of which adjs are all 0?
            if no_sc_idx.sum() > 0:
                adjs[no_sc_idx, 0] = torch.eye(num_nodes, device=adjs.device)
            if no_fc_idx.sum() > 0:
                adjs[no_fc_idx, 1] = torch.eye(num_nodes, device=adjs.device)

        node_embed_1 = self.sign(adjs[:, 0], feats) # geom
        node_embed_2 = self.sign2(adjs[:, 1], feats) # cell_type

        if self.fuse_type == 'graph_embed':
            graph_embed = \
                self.fuse_graph_embed(node_embed_1, node_embed_2, batch, no_sc_idx, no_fc_idx)
       
        elif self.fuse_type == 'node_embed_on_graph_embed':
            graph_embed = \
                self.fuse_node_embed_on_graph_embed(node_embed_1, node_embed_2, 
                                                    batch, no_sc_idx, no_fc_idx, 
                                                    num_graphs, num_nodes)
        elif self.fuse_type == '0_miss':
            graph_embed = \
                self.forward_0_miss(node_embed_1, node_embed_2, 
                                    batch, no_sc_idx, no_fc_idx, 
                                    num_graphs, num_nodes)
        elif self.fuse_type in ['unit_miss', 'normal']:
            graph_embed = \
                self.forward_normal(node_embed_1, node_embed_2, batch)
        if self.num_graph_tasks > 0:
            graph_pred = self.graph_pred_module(graph_embed)

        return graph_pred