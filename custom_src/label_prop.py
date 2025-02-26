import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np

from dgl import DGLGraph, out_subgraph, node_subgraph

class LabelProp(nn.Module):
    def __init__(self, fuse_method="mean"):
        super().__init__()
        self.fuse_method = fuse_method
        self.simple_prim = ['mean', 'sum', 'min', 'max']
    
    def fusion(self, lbls: torch.Tensor, knn_g: DGLGraph, null_mask: torch.Tensor):
        lbls = lbls.clone()
        if null_mask.sum() > 0:
            if self.fuse_method in self.simple_prim:
                # trick: we only supplement labels for unlabeled samples which have at least one
                # labeled input nodes. The input edges can only come from labeled nodes otherwise
                # the average will induce much noise.

                labeled_samples = torch.where(~null_mask)[0]
                # out edges of unlabeled samples are removed to avoid
                # noisy (null) labels as input
                knn_g = out_subgraph(knn_g, labeled_samples) 
                knn_n, target_n = knn_g.in_edges(v=torch.where(null_mask)[0])
                # idx not equivalent to null_mask since some samples 
                # are removed if they have no input labeled samples.
                idx = torch.unique(target_n) 
                lbls[idx] = scatter(lbls[knn_n], target_n, 
                                        reduce=self.fuse_method, dim=0, 
                                        dim_size=lbls.shape[0])[idx]

        return lbls

    def forward(self, lbls, no_lbl_idx, knn_sc, knn_fc, train_idx):
        # remove valid and test samples from graphs to avoid label leakage
        knn_sc = node_subgraph(knn_sc, train_idx, relabel_nodes=False, store_ids=False)
        knn_fc = node_subgraph(knn_fc, train_idx, relabel_nodes=False, store_ids=False)

        lbls1 = self.fusion(lbls, knn_sc, no_lbl_idx)
        lbls2 = self.fusion(lbls, knn_fc, no_lbl_idx)

        mask = (~torch.isinf(lbls1) * ~torch.isinf(lbls2)) * train_idx
        lbls[mask] = (lbls1[mask] + lbls2[mask]) / 2.

        return lbls, mask
