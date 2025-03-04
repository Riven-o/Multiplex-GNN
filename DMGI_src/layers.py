import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, drop_prob, isBias=False):
        super(GCN, self).__init__()

        self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)

        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        elif act == 'rrelu':
            self.act = nn.RReLU()
        elif act == 'selu':
            self.act = nn.SELU()
        elif act == 'celu':
            self.act = nn.CELU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'identity':
            self.act = nn.Identity()

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.drop_prob = drop_prob
        self.isBias = isBias

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq = F.dropout(seq, self.drop_prob, training=self.training)
        seq = self.fc_1(seq)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            seq = torch.bmm(adj, seq)

        if self.isBias:
            seq += self.bias_1

        return self.act(seq)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k_bilinear = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1) # c: summary vector, h_pl: positive, h_mi: negative
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k_bilinear(h_pl, c_x), 2) # sc_1 = 1 x nb_nodes
        sc_2 = torch.squeeze(self.f_k_bilinear(h_mi, c_x), 2) # sc_2 = 1 x nb_nodes

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args
        self.A = nn.ModuleList([nn.Linear(args.hid_units, 1) for _ in range(args.nb_modal)])
        self.weight_init()

    def weight_init(self):
        for i in range(self.args.nb_modal):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, feat_pos, feat_neg, summary):
        feat_pos, feat_pos_attn = self.attn_feature(feat_pos)
        feat_neg, feat_neg_attn = self.attn_feature(feat_neg)
        summary, summary_attn = self.attn_summary(summary)

        return feat_pos, feat_neg, summary


    def attn_feature(self, features):
        features_attn = []
        for i in range(self.args.nb_modal):
            features_attn.append((self.A[i](features[i].squeeze())))
        features_attn = F.softmax(torch.cat(features_attn, 1), -1)
        features = torch.cat(features,1).squeeze(0)
        features_attn_reshaped = features_attn.transpose(1, 0).contiguous().view(-1, 1)
        features = features * features_attn_reshaped.expand_as(features)
        features = features.view(self.args.nb_modal, self.args.nb_nodes, self.args.hid_units).sum(0).unsqueeze(0)

        return features, features_attn

    def attn_summary(self, features):
        features_attn = []
        for i in range(self.args.nb_modal):
            features_attn.append((self.A[i](features[i].squeeze())))
        features_attn = F.softmax(torch.cat(features_attn), dim=-1).unsqueeze(1)
        features = torch.cat(features, 0)
        features_attn_expanded = features_attn.expand_as(features)
        features = (features * features_attn_expanded).sum(0).unsqueeze(0)

        return features, features_attn