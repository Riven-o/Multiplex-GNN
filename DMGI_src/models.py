import torch
import torch.nn as nn
from .layers import GCN, Discriminator, Attention
import numpy as np
from torch_geometric.nn.models import GCN


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        # self.gcn = nn.ModuleList([GCN(args.ft_size, args.hid_dim, args.activation, args.drop_prob, args.isBias) for _ in range(args.nb_modal)])
        self.gcn = nn.ModuleList(
            [
                GCN(
                    args.ft_size, args.hid_dim, 1, dropout=args.drop_prob
                ) for _ in range(args.nb_modal)
            ]
        )

        self.disc = Discriminator(args.hid_dim)
        self.Z = nn.Parameter(torch.FloatTensor(args.nb_graphs, args.nb_nodes, args.hid_dim))
        self.readout_func = self.args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        if args.isSemi:
            self.logistic = LogReg(args.hid_dim, args.nb_classes).to(args.device)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.Z)

    def forward(self, feature, data, shuf, sparse, msk, samp_bias1, samp_bias2, idx):
        h_1_all = []; h_2_all = []; c_all = []; logits = []
        result = {}

        for i in range(self.args.nb_modal):
            """
            feature: shape=[num_graph, num_nodes, num_feat]
            shuf = feature
            data: a list of [
                Batch(modal_1), Batch(modal_2), ...
            ], where each Batch(modal_1) a batch of graphs (Batch(modal_1) = num_graph, num_nodes * num_graph, num_feat)
            other arguments are useless
            """
            h_1 = self.gcn[i](feature.reshape(-1, feature.shape[-1]), data[i].edge_index)

            # how to readout positive summary vector
            h_1_ = h_1.reshape(len(data[i]), -1, h_1.shape[-1])
            c = self.readout_func(h_1_)
            c = self.args.readout_act_func(c)
            h_2 = self.gcn[i](shuf.reshape(-1, shuf.shape[-1]), data[i].edge_index)
            h_2_ = h_2.reshape(len(data[i]), -1, h_2.shape[-1])
            logit = self.disc(c, h_1_, h_2_, samp_bias1, samp_bias2)

            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)

        result['logits'] = logits

        # Attention or not
        if self.args.isAttn:
            h_1_all_lst = []; h_2_all_lst = []; c_all_lst = []

            for h_idx in range(self.args.nheads):
                h_1_all_, h_2_all_, c_all_ = self.attn[h_idx](h_1_all, h_2_all, c_all)
                h_1_all_lst.append(h_1_all_); h_2_all_lst.append(h_2_all_); c_all_lst.append(c_all_)

            h_1_all = torch.mean(torch.cat(h_1_all_lst, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all_lst, 0), 0).unsqueeze(0)

        else:
            h_1_all = torch.mean(torch.stack(h_1_all), 0).reshape(len(data[0]), feature.shape[1], -1)
            h_2_all = torch.mean(torch.stack(h_2_all), 0).reshape(len(data[0]), feature.shape[1], -1)


        # consensus regularizer
        Z = self.Z[idx]
        pos_reg_loss = ((Z - h_1_all) ** 2).sum()
        neg_reg_loss = ((Z - h_2_all) ** 2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss
        result['reg_loss'] = reg_loss

        # semi-supervised module
        if self.args.isSemi:
            # semi = self.logistic(self.Z).squeeze(0)
            semi = self.logistic(h_1_all.mean(1)).flatten() # TODO(jiahang): experience shows that sum will be better than mean
            result['semi'] = semi

        return result

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
