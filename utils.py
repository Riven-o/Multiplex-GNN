import os
import pickle
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import random
import os

from torchmetrics import Accuracy, AUROC, AveragePrecision, MeanSquaredError
from sklearn.model_selection import train_test_split, KFold
from torch_geometric.data import Batch
from torch_geometric.transforms import SIGN
from torch_scatter import scatter
import torch.nn as nn

from torch_geometric.data import Batch, Data
from time import time

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
SINGLE_MODALITY_MODELS = ['GCN', 'SAGE', 'SGC', 'GAT', 'GIN', 'Transformer']
FUSE_SINGLE_MODALITY_MODELS = \
    [name + '_fuse_embed' for name in SINGLE_MODALITY_MODELS] + \
    [name + '_fuse_graph' for name in SINGLE_MODALITY_MODELS] + \
    [name + '_fuse_pred' for name in SINGLE_MODALITY_MODELS]
FUSE_SINGLE_MODALITY_MODELS_NOSIA = \
    [name + '_fuse_embed_nosia' for name in SINGLE_MODALITY_MODELS] + \
    [name + '_fuse_graph_nosia' for name in SINGLE_MODALITY_MODELS] + \
    [name + '_fuse_pred_nosia' for name in SINGLE_MODALITY_MODELS]

def model_infer(model, model_name, **kwargs):
    """
    adjs: adj matrices
    idx: split index 
    raw_Xs: original Xs
    data_lisr: list type of pyg data
    device
    """
    if model_name == 'MHGCN':
        adjs, raw_Xs = kwargs['adjs'], kwargs['raw_Xs']
        logits = model(adjs, raw_Xs)
        
    elif model_name in ['NeuroPath', 'Mew'] + SINGLE_MODALITY_MODELS + \
            FUSE_SINGLE_MODALITY_MODELS + \
            FUSE_SINGLE_MODALITY_MODELS_NOSIA:
        data = kwargs['data']
        logits = model(data)

    elif model_name in ['MewCustom', 'MewFuseGraph', 'MHGCNFuseGraph']:
        adjs, raw_Xs, no_sc_idx, no_fc_idx = \
            kwargs['adjs'], kwargs['raw_Xs'], kwargs['no_sc_idx'], kwargs['no_fc_idx']
        logits = model(adjs, raw_Xs, no_sc_idx, no_fc_idx)

    return logits.squeeze()

def load_dataset(label_type='classification', eval_type='split', split_args: dict = None, 
                 cross_args: dict = None, reload = False, 
                 file_option = "", seed=0, version=1, online_split = True,
                 target_label = 'nih_totalcogcomp_ageadjusted'):
    """

    label_type: if classification, all labels(int) are converted into its index. If regression, use original values.
        note: even if it's a regression task in nature, if labels are int, sometimes classification loss function,
        such as cross-entropy loss, has better performance.
    eval_type: if eval_type == split, then train-valid-test split style evaluation. elif eval_type == cross, then n_fold
                cross evalidation
    """
    # TODO train-valid-test split, and cross-validation
    # assert label_type in ['classification', 'regression']
    assert label_type == 'regression'
    # assert eval_type in ['split', 'cross']
    assert eval_type == 'split'
    assert file_option in ['', '_miss_graph', '_miss_graph_miss_label']
    if eval_type == 'split':
        assert split_args is not None

    file_path = f'./dataset/processed_data_{label_type}_{eval_type}{file_option}'
    if online_split:
        file_path += '_onlineSplit'
    else:
        file_path += '_onlineSplitFalse'
    file_path += '.pkl'
    if os.path.exists(file_path) and not reload:
        print(f"read processed data from {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        adjs = data['adjs']
        raw_Xs = data['raw_Xs']
        labels = data['labels']
        mu_lbls = data['mu_lbls']
        std_lbls = data['std_lbls']
        if '_miss_graph' in file_option:
            no_sc_idx = data['no_sc_idx']
            no_fc_idx = data['no_fc_idx']
        if '_miss_label' in file_option:
            no_lbl_idx = data['no_lbl_idx']
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(dir_path, 'dataset', 'valid_test_split', target_label)
        path = os.path.join(dir_path, f'v{version}.pkl')

        with open(path, 'rb') as f:
            splits = pickle.load(f)[seed]
        
        valid_names, test_names = splits['valid_names'], splits['test_names']
        data_path = './dataset'
        path = os.path.join(data_path, 'FC_Fisher_Z_transformed.pkl')
        with open(path, 'rb') as f:
            data_FC = pickle.load(f)

        path = os.path.join(data_path, 'SC.pkl')
        with open(path, 'rb') as f:
            data_SC = pickle.load(f)

        path = os.path.join(data_path, 'T1.pkl')
        with open(path, 'rb') as f:
            data_raw_X = pickle.load(f)

        path = os.path.join(data_path, 'demo.pkl')
        with open(path, 'rb') as f:
            data_labels = pickle.load(f)
        
        # NOTE note that only data_SC has full keys, is it a semi-supervised task?
        ## we only take graph which have labels and both SC, FC modalities.
        all_keys = np.unique(list(data_SC.keys()) + list(data_FC.keys()) + list(data_labels.keys()))
        data_size = len(all_keys)
        adjs = torch.zeros((data_size, 2, 200, 200))
        labels = torch.zeros(data_size)
        raw_Xs = torch.zeros((data_size, 200, 9))
        no_sc_idx, no_fc_idx, no_lbl_idx = \
            torch.zeros(data_size, dtype=torch.bool), \
            torch.zeros(data_size, dtype=torch.bool), \
            torch.zeros(data_size, dtype=torch.bool)
        mask = []
        train_idx, valid_idx, test_idx = \
            torch.zeros(data_size, dtype=torch.bool), \
            torch.zeros(data_size, dtype=torch.bool), \
            torch.zeros(data_size, dtype=torch.bool)
        
        for i, name in tqdm(enumerate(all_keys)):
            # valid and test set are pre-specified
            if name in valid_names:
                valid_idx[i] = True
                labels[i] = float(data_labels[name][target_label])
                
            elif name in test_names:
                test_idx[i] = True
                labels[i] = float(data_labels[name][target_label])
                
            else: # train set is built here to incorporate more special graphs (missing layers / labels)
                if name not in data_raw_X.keys():
                    continue
                if name not in data_labels.keys() or target_label not in data_labels[name].keys():
                    if '_miss_label' in file_option:
                        labels[i] = 0.
                        no_lbl_idx[i] = True
                    else:
                        continue
                else:
                    labels[i] = float(data_labels[name][target_label])
                
                if name not in data_SC.keys():
                    if '_miss_graph' in file_option:
                        adjs[i, 0] = torch.zeros(200, 200)
                        no_sc_idx[i] = True
                    else:
                        continue
                else:
                    adjs[i, 0] = torch.tensor(data_SC[name])

                if name not in data_FC.keys():
                    if '_miss_graph' in file_option:
                        adjs[i, 1] = torch.zeros(200, 200)
                        no_fc_idx[i] = True
                    else:
                        continue
                else:
                    adjs[i, 1] = torch.tensor(data_FC[name])
                train_idx[i] = True

            raw_X = data_raw_X[name].drop(columns=['StructName'])
            raw_X = raw_X.to_numpy().astype(float)
            raw_Xs[i] = torch.tensor(raw_X)

            mask.append(i)

        adjs = adjs[mask]
        labels = labels[mask]
        raw_Xs = raw_Xs[mask]
        no_sc_idx = no_sc_idx[mask]
        no_fc_idx = no_fc_idx[mask]
        no_lbl_idx = no_lbl_idx[mask]
        train_idx, valid_idx, test_idx = \
            train_idx[mask], valid_idx[mask], test_idx[mask]

        mu_lbls, std_lbls = None, None
        if label_type == 'classification':
            labels_class = torch.zeros_like(labels, dtype=torch.long)
            for i, label in enumerate(labels.unique()):
                labels_class[labels == label] = i
            labels = labels_class

        else:
            if '_miss_label' not in file_option:
                mu_lbls, std_lbls = labels.mean(), labels.std()
                labels = (labels - mu_lbls) / std_lbls
                
        
        batchnorm = nn.BatchNorm1d(raw_Xs.shape[-1], affine=False)
        layernorm = nn.LayerNorm([adjs.shape[-2], adjs.shape[-1]], elementwise_affine=False)

        original_feat_shape = raw_Xs.shape
        raw_Xs = batchnorm(
            raw_Xs.reshape(-1, raw_Xs.shape[-1])
        ).reshape(original_feat_shape)

        original_adjs_shape = adjs.shape
        adjs = layernorm(
            adjs.reshape(-1, adjs.shape[-2], adjs.shape[-1])
        ).reshape(original_adjs_shape)

        data = {
            'adjs': adjs,
            'raw_Xs': raw_Xs,
            'labels': labels,
            'mu_lbls': mu_lbls,
            'std_lbls': std_lbls,
        }

        if '_miss_graph' in file_option:
            data['no_sc_idx'] = no_sc_idx
            data['no_fc_idx'] = no_fc_idx
        if '_miss_label' in file_option:
            data['no_lbl_idx'] = no_lbl_idx

        assert (adjs == torch.transpose(adjs, 2, 3)).all().item(), "adj matrices are not symmetric"

    if online_split:
        if eval_type == 'split':
            train_size, valid_size, test_size = \
                split_args['train_size'], split_args['valid_size'], split_args['test_size']
            idx = np.arange(len(labels))
            train_valid_idx, test_idx = \
                train_test_split(idx, test_size=test_size)
            train_idx, valid_idx = \
                train_test_split(train_valid_idx, 
                                test_size=valid_size / (train_size + valid_size))
        elif eval_type == 'cross':
            kfold = KFold(n_splits=5, shuffle=True)
            splits = list(kfold.split(X=idx))

    if not online_split:
        assert valid_idx.sum() == test_idx.sum() == 75

    if file_option == "_miss_graph":
        if not online_split:
            assert no_sc_idx[valid_idx].sum() == no_sc_idx[test_idx].sum() \
                == no_fc_idx[valid_idx].sum() == no_fc_idx[test_idx].sum() == 0.
        return adjs, raw_Xs, labels, splits, mu_lbls, std_lbls, no_sc_idx, no_fc_idx
    elif file_option == "_miss_graph_miss_label":
        if not online_split:
            assert no_sc_idx[valid_idx].sum() == no_sc_idx[test_idx].sum() \
                == no_fc_idx[valid_idx].sum() == no_fc_idx[test_idx].sum() \
                == no_lbl_idx[valid_idx].sum() == no_lbl_idx[test_idx].sum() == 0.
        return adjs, raw_Xs, labels, splits, mu_lbls, std_lbls, no_sc_idx, no_fc_idx, no_lbl_idx
    else:
        return adjs, raw_Xs, labels, splits, mu_lbls, std_lbls

def load_dataset1(label_type='classification', eval_type='split', split_args: dict = None,
                 file_option = "", seed=0, version=1, online_split = True,
                 target_label = 'nih_totalcogcomp_ageadjusted'):
    """

    similar to load_dataset, but in this func we don't save at / load from any local dataset
    since it's trick to implement valid and test masks in these cases.
    """
    # TODO train-valid-test split, and cross-validation
    # assert label_type in ['classification', 'regression']
    assert label_type == 'regression'
    # assert eval_type in ['split', 'cross']
    assert eval_type == 'split'
    assert file_option in ['', '_miss_graph', '_miss_graph_miss_label']
    if eval_type == 'split':
        assert split_args is not None

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, 'dataset', '1', 'old_valid_test_split', target_label)
    path = os.path.join(dir_path, f'v{version}.pkl')

    with open(path, 'rb') as f:
        splits = pickle.load(f)[seed]
    
    valid_names, test_names = splits['valid_names'], splits['test_names']
    data_path = './dataset/1'
    path = os.path.join(data_path, 'FC_Fisher_Z_transformed.pkl')
    with open(path, 'rb') as f:
        data_FC = pickle.load(f)

    path = os.path.join(data_path, 'SC.pkl')
    with open(path, 'rb') as f:
        data_SC = pickle.load(f)

    path = os.path.join(data_path, 'T1.pkl')
    with open(path, 'rb') as f:
        data_raw_X = pickle.load(f)

    path = os.path.join(data_path, 'demo.pkl')
    with open(path, 'rb') as f:
        data_labels = pickle.load(f)
    
    all_keys = np.unique(list(data_SC.keys()) + list(data_FC.keys()) + list(data_labels.keys()))
    data_size = len(all_keys)
    sample = data_raw_X[list(data_raw_X.keys())[0]]
    num_nodes = len(sample)
    num_features = len(sample.keys()) - 1
    adjs = torch.zeros((data_size, 2, num_nodes, num_nodes))
    labels = torch.zeros(data_size)
    raw_Xs = torch.zeros((data_size, num_nodes, num_features))
    no_sc_idx, no_fc_idx, no_lbl_idx = \
        torch.zeros(data_size, dtype=torch.bool), \
        torch.zeros(data_size, dtype=torch.bool), \
        torch.zeros(data_size, dtype=torch.bool)
    mask = []
    train_idx, valid_idx, test_idx = \
        torch.zeros(data_size, dtype=torch.bool), \
        torch.zeros(data_size, dtype=torch.bool), \
        torch.zeros(data_size, dtype=torch.bool)
    
    for i, name in tqdm(enumerate(all_keys)):
        # valid and test set are pre-specified
        if name in valid_names:
            valid_idx[i] = True
            labels[i] = float(data_labels[name][target_label])
            adjs[i, 0] = torch.tensor(data_SC[name])
            adjs[i, 1] = torch.tensor(data_FC[name])
            
        elif name in test_names:
            test_idx[i] = True
            labels[i] = float(data_labels[name][target_label])
            adjs[i, 0] = torch.tensor(data_SC[name])
            adjs[i, 1] = torch.tensor(data_FC[name])
            
        else: # train set is built here to incorporate more special graphs (missing layers / labels)
            if name not in data_raw_X.keys():
                continue
            if name not in data_labels.keys() or target_label not in data_labels[name].keys():
                if '_miss_label' in file_option:
                    labels[i] = torch.inf
                    no_lbl_idx[i] = True
                else:
                    continue
            else:
                labels[i] = float(data_labels[name][target_label])
            
            if name not in data_SC.keys():
                if '_miss_graph' in file_option:
                    adjs[i, 0] = torch.zeros(num_nodes, num_nodes)
                    no_sc_idx[i] = True
                else:
                    continue
            else:
                adjs[i, 0] = torch.tensor(data_SC[name])

            if name not in data_FC.keys():
                if '_miss_graph' in file_option:
                    adjs[i, 1] = torch.zeros(num_nodes, num_nodes)
                    no_fc_idx[i] = True
                else:
                    continue
            else:
                adjs[i, 1] = torch.tensor(data_FC[name])
            train_idx[i] = True

        raw_X = data_raw_X[name].drop(columns=['StructName'])
        raw_X = raw_X.to_numpy().astype(float)
        raw_Xs[i] = torch.tensor(raw_X)

        mask.append(i)

    adjs = adjs[mask]
    labels = labels[mask]
    raw_Xs = raw_Xs[mask]
    no_sc_idx = no_sc_idx[mask]
    no_fc_idx = no_fc_idx[mask]
    no_lbl_idx = no_lbl_idx[mask]
    train_idx, valid_idx, test_idx = \
        train_idx[mask], valid_idx[mask], test_idx[mask]

    mu_lbls, std_lbls = None, None
    if label_type == 'classification':
        labels_class = torch.zeros_like(labels, dtype=torch.long)
        for i, label in enumerate(labels.unique()):
            labels_class[labels == label] = i
        labels = labels_class

    else:
        if '_miss_label' in file_option:
            mu_lbls, std_lbls = labels[~no_lbl_idx].mean(), labels[~no_lbl_idx].std()
            labels[~no_lbl_idx] = (labels[~no_lbl_idx] - mu_lbls) / std_lbls
        else:
            mu_lbls, std_lbls = labels.mean(), labels.std()
            labels = (labels - mu_lbls) / std_lbls
            
    
    batchnorm = nn.BatchNorm1d(raw_Xs.shape[-1], affine=False)
    layernorm = nn.LayerNorm([adjs.shape[-2], adjs.shape[-1]], elementwise_affine=False)

    original_feat_shape = raw_Xs.shape
    raw_Xs = batchnorm(
        raw_Xs.reshape(-1, raw_Xs.shape[-1])
    ).reshape(original_feat_shape)

    original_adjs_shape = adjs.shape
    adjs = layernorm(
        adjs.reshape(-1, adjs.shape[-2], adjs.shape[-1])
    ).reshape(original_adjs_shape)

    assert (adjs == torch.transpose(adjs, 2, 3)).all().item(), "adj matrices are not symmetric"

    if online_split:
        train_idx, valid_idx, test_idx = \
            torch.zeros(data_size, dtype=torch.bool), \
            torch.zeros(data_size, dtype=torch.bool), \
            torch.zeros(data_size, dtype=torch.bool)
        if eval_type == 'split':
            train_size, valid_size, test_size = \
                split_args['train_size'], split_args['valid_size'], split_args['test_size']
            idx = np.arange(len(labels))
            _train_valid_idx, _test_idx = \
                train_test_split(idx, test_size=test_size)
            _train_idx, _valid_idx = \
                train_test_split(_train_valid_idx, 
                                test_size=valid_size / (train_size + valid_size))
            train_idx[_train_idx], valid_idx[_valid_idx], test_idx[_test_idx] = \
                True, True, True
        elif eval_type == 'cross':
            kfold = KFold(n_splits=5, shuffle=True)
            splits = list(kfold.split(X=idx))

    if not online_split:
        assert valid_idx.sum() == test_idx.sum() == 75

    splits = {
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'test_idx': test_idx,
    }

    if file_option == "_miss_graph":
        if not online_split:
            assert no_sc_idx[valid_idx].sum() == no_sc_idx[test_idx].sum() \
                == no_fc_idx[valid_idx].sum() == no_fc_idx[test_idx].sum() == 0.
        return adjs, raw_Xs, labels, splits, mu_lbls, std_lbls, no_sc_idx, no_fc_idx
    elif file_option == "_miss_graph_miss_label":
        if not online_split:
            assert no_sc_idx[valid_idx].sum() == no_sc_idx[test_idx].sum() \
                == no_fc_idx[valid_idx].sum() == no_fc_idx[test_idx].sum() \
                == no_lbl_idx[valid_idx].sum() == no_lbl_idx[test_idx].sum() == 0.
        return adjs, raw_Xs, labels, splits, mu_lbls, std_lbls, no_sc_idx, no_fc_idx, no_lbl_idx
    else:
        return adjs, raw_Xs, labels, splits, mu_lbls, std_lbls


def load_dataset2(label_type='classification', eval_type='split', split_args: dict = None,
                 file_option = "", seed=0, version=1, online_split = True,
                 target_label = 'nih_totalcogcomp_ageadjusted', miss='none', gen_miss='pad', dataset='1'):

    # assert label_type in ['classification', 'regression']
    assert label_type == 'regression'
    assert file_option in ['', '_miss_graph', '_miss_graph_miss_label']

    data_path = f'/home/mufan/mohan/gnn/dataset/{dataset}'
    path = os.path.join(data_path, 'valid_test_split', target_label, 'v1.pkl')

    with open(path, 'rb') as f:
        splits = pickle.load(f)[seed]

    train_names, valid_names, test_names = splits['train_names'], splits['valid_names'], splits['test_names']
    full_names, miss_names = splits['full_names'], splits['miss_names']
    miss_SC_knn, miss_FC_knn = splits['miss_SC_knn'], splits['miss_FC_knn']

    path = os.path.join(data_path, 'FC_Fisher_Z_transformed.pkl')
    with open(path, 'rb') as f:
        data_FC = pickle.load(f)

    path = os.path.join(data_path, 'SC.pkl')
    with open(path, 'rb') as f:
        data_SC = pickle.load(f)

    path = os.path.join(data_path, 'T1.pkl')
    with open(path, 'rb') as f:
        data_raw_X = pickle.load(f)

    path = os.path.join(data_path, 'demo.pkl')
    with open(path, 'rb') as f:
        data_labels = pickle.load(f)
    
    if miss != 'none' and gen_miss == 'drop':
        train_names = full_names
    all_keys = np.concatenate([train_names, valid_names, test_names])
    data_size = len(all_keys)
    sample = data_raw_X[list(data_raw_X.keys())[0]]
    num_nodes = len(sample)
    num_features = len(sample.keys()) - 1

    adjs = torch.zeros((data_size, 2, num_nodes, num_nodes))
    labels = torch.zeros(data_size)
    raw_Xs = torch.zeros((data_size, num_nodes, num_features))
    no_sc_idx, no_fc_idx, no_lbl_idx = \
        torch.zeros(data_size, dtype=torch.bool), \
        torch.zeros(data_size, dtype=torch.bool), \
        torch.zeros(data_size, dtype=torch.bool)
    train_idx, valid_idx, test_idx = \
        torch.zeros(data_size, dtype=torch.bool), \
        torch.zeros(data_size, dtype=torch.bool), \
        torch.zeros(data_size, dtype=torch.bool)

    for i, name in tqdm(enumerate(all_keys)):
        # valid and test set are pre-specified
        if name in valid_names:
            valid_idx[i] = True
            adjs[i, 0] = torch.tensor(data_SC[name])
            adjs[i, 1] = torch.tensor(data_FC[name])
        elif name in test_names:
            test_idx[i] = True
            adjs[i, 0] = torch.tensor(data_SC[name])
            adjs[i, 1] = torch.tensor(data_FC[name])
        else:
            train_idx[i] = True
            if name in miss_names:
                if miss == 'fc':
                    no_fc_idx[i] = True
                    adjs[i, 0] = torch.tensor(data_SC[name])
                    if gen_miss == 'pad':
                        adjs[i, 1] = torch.zeros(num_nodes, num_nodes)
                    elif gen_miss == 'knn':
                        neighbor_names = miss_FC_knn[name]
                        neighbor_fc = [data_FC[neighbor_name] for neighbor_name in neighbor_names]
                        neighbor_fc = np.stack(neighbor_fc, axis=0)
                        adjs[i, 1] = torch.tensor(neighbor_fc.mean(axis=0))
                    elif gen_miss == 'drop':
                        adjs[i, 1] = torch.tensor(data_FC[name])
                    else:
                        raise NotImplementedError
                elif miss == 'sc':
                    no_sc_idx[i] = True
                    adjs[i, 1] = torch.tensor(data_FC[name])
                    if gen_miss == 'pad':
                        adjs[i, 0] = torch.zeros(num_nodes, num_nodes)
                    elif gen_miss == 'knn':
                        neighbor_names = miss_SC_knn[name]
                        neighbor_sc = [data_SC[neighbor_name] for neighbor_name in neighbor_names]
                        neighbor_sc = np.stack(neighbor_sc, axis=0)
                        adjs[i, 0] = torch.tensor(neighbor_sc.mean(axis=0))
                    elif gen_miss == 'drop':
                        adjs[i, 0] = torch.tensor(data_SC[name])
                    else:
                        raise NotImplementedError
                else:
                    adjs[i, 0] = torch.tensor(data_SC[name])
                    adjs[i, 1] = torch.tensor(data_FC[name])
            else:
                adjs[i, 0] = torch.tensor(data_SC[name])
                adjs[i, 1] = torch.tensor(data_FC[name])

        raw_X = data_raw_X[name].drop(columns=['StructName'])
        raw_X = raw_X.to_numpy().astype(float)
        raw_Xs[i] = torch.tensor(raw_X)
        labels[i] = float(data_labels[name][target_label])

    mu_lbls, std_lbls = None, None
    if label_type == 'classification':
        labels_class = torch.zeros_like(labels, dtype=torch.long)
        for i, label in enumerate(labels.unique()):
            labels_class[labels == label] = i
        labels = labels_class
    else:
        mu_lbls, std_lbls = labels.mean(), labels.std()
        labels = (labels - mu_lbls) / std_lbls
            
    batchnorm = nn.BatchNorm1d(raw_Xs.shape[-1], affine=False)
    layernorm = nn.LayerNorm([adjs.shape[-2], adjs.shape[-1]], elementwise_affine=False)

    original_feat_shape = raw_Xs.shape
    raw_Xs = batchnorm(
        raw_Xs.reshape(-1, raw_Xs.shape[-1])
    ).reshape(original_feat_shape)

    original_adjs_shape = adjs.shape
    adjs = layernorm(
        adjs.reshape(-1, adjs.shape[-2], adjs.shape[-1])
    ).reshape(original_adjs_shape)

    assert (adjs == torch.transpose(adjs, 2, 3)).all().item(), "adj matrices are not symmetric"

    splits = {
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'test_idx': test_idx,
    }

    return adjs, raw_Xs, labels, splits, mu_lbls, std_lbls



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Evaluator:
    def __init__(self, mu_lbls, std_lbls, label_type, num_classes, device):
        assert label_type in ['classification', 'regression']
        self.mu_lbls, self.std_lbls = mu_lbls, std_lbls
        if label_type == 'classification':
            assert num_classes is not None
            self.acc, self.auroc, self.auprc = \
                Accuracy(task="multiclass", num_classes=num_classes).to(device), \
                AUROC(task="multiclass", num_classes=num_classes).to(device), \
                AveragePrecision(task="multiclass", num_classes=num_classes).to(device)
        else:
            self.mse = MeanSquaredError().to(device)
        self.label_type = label_type

    def evaluate(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.label_type == 'classification':
            return self.acc(logits, labels), self.auroc(logits, labels), self.auprc(logits, labels)
        else:
            labels = labels * self.std_lbls + self.mu_lbls
            logits = logits * self.std_lbls + self.mu_lbls
            return self.mse(logits.squeeze(), labels).sqrt()

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def step_score(self, score, model, save=True):  # test score
        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_model(model)
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if save:
                self.save_model(model)
            self.best_score = np.max((score, self.best_score))
            self.counter = 0
            self.early_stop = False

    def save_model(self, model):
        model.eval()
        self.best_model = deepcopy(model.state_dict())

    def load_model(self, model):
        model.load_state_dict(self.best_model)

def evaluate(g, feat, labels, mask, model: torch.nn.Module):
    model.eval()
    with torch.no_grad():
        logits = model(g, feat)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def adj_weight2bin(adjs, ratio_sc, ratio_fc, single_modal=False):
    if single_modal:
        assert ratio_sc == ratio_fc # only one ratio is needed
        topk = int(adjs.shape[-2] * adjs.shape[-1] * ratio_sc)
        original_shape = adjs.shape
        adjs = adjs.flatten(-2)
        idx = torch.topk(adjs, topk, dim=-1)[1]
        adjs = scatter(torch.ones_like(idx), idx).int()
        return adjs.reshape(original_shape)

    if adjs.shape[1] == 2:
        topk_0 = int(adjs.shape[-2] * adjs.shape[-1] * ratio_sc)
        topk_1 = int(adjs.shape[-2] * adjs.shape[-1] * ratio_fc)
        original_shape = [adjs.shape[0], adjs.shape[2], adjs.shape[3]]
        adjs = adjs.flatten(-2)
        idx_0 = torch.topk(adjs[:, 0], topk_0, dim=-1)[1]
        idx_1 = torch.topk(adjs[:, 1], topk_1, dim=-1)[1]
        adjs_0 = scatter(torch.ones_like(idx_0), idx_0).int()
        adjs_1 = scatter(torch.ones_like(idx_1), idx_1).int()
        adjs_0 = adjs_0.reshape(original_shape)
        adjs_1 = adjs_1.reshape(original_shape)

        return adjs_0, adjs_1
    
def pyg_preprocess_sign(raw_Xs: torch.Tensor, adjs: torch.Tensor, k: int):
    data_list = torch.zeros((
        raw_Xs.shape[0], adjs.shape[1], k, raw_Xs.shape[1], raw_Xs.shape[2]
    )).to(raw_Xs.device) # num_graphs, num_graph_layers, k, num_nodes, num_feats

    for i in range(k):
        data_list[:, 0, i, :] = adjs[:, 0, :, :].matmul(raw_Xs)
        data_list[:, 1, i, :] = adjs[:, 1, :, :].matmul(raw_Xs)
        adjs = adjs.matmul(adjs)
    return data_list

def to_pyg_single(raw_Xs: torch.Tensor, labels: torch.Tensor, adjs: torch.Tensor, ratio_sc: float, ratio_fc: float, option: str):
    adjs_0, adjs_1 = adj_weight2bin(adjs, ratio_sc, ratio_fc)

    if option == 'sc':
        adjs_target = adjs_0
    elif option == 'fc':
        adjs_target = adjs_1
    
    data_list = []
    for i in tqdm(range(len(adjs_target))):
        data = {
            'x': raw_Xs[i],
            'y': labels[i],
            'edge_index': torch.stack(torch.nonzero(adjs_target[i], as_tuple=True)),
            'adj_sc': adjs_0[i].unsqueeze(0),
            'adj_fc': adjs_1[i].unsqueeze(0),
        }
        data = Data(**data)
        data_list.append(data)

    return data_list

def split_pyg(data_list: list, train_idx: list, valid_idx: list, test_idx: list):
    print("preprocessing pyg data list")
    time_st = time()
    train_data = [data_list[i] for i in train_idx]
    train_data = Batch.from_data_list(train_data).to(device)

    valid_data = [data_list[i] for i in valid_idx]
    valid_data = Batch.from_data_list(valid_data).to(device)

    test_data = [data_list[i] for i in test_idx]
    test_data = Batch.from_data_list(test_data).to(device)

    data = Batch.from_data_list(data_list).to(device)

    print(f"finish preprocessing: {time() - time_st:.2f}s")
    return train_data, valid_data, test_data, data

def to_pyg_fuse(raw_Xs: torch.Tensor, labels: torch.Tensor, adjs: torch.Tensor, 
                fuse_type: str, reduce_fuse: str, 
                ratio_sc: float, ratio_fc: float, ratio: float):
    if fuse_type == 'fuse_graph':
        if reduce_fuse == 'mean':
            adjs = adjs.mean(dim=1)
            adjs = adj_weight2bin(adjs, ratio, ratio, single_modal=True)
        elif reduce_fuse == 'sum':
            adjs = adjs.sum(dim=1)
            adjs = adj_weight2bin(adjs, ratio, ratio, single_modal=True)
        elif reduce_fuse == 'and':
            adjs_0, adjs_1 = adj_weight2bin(adjs, ratio_sc, ratio_fc)
            adjs = adjs_0 * adjs_1
        elif reduce_fuse == 'or':
            adjs_0, adjs_1 = adj_weight2bin(adjs, ratio_sc, ratio_fc)
            adjs = torch.logical_or(adjs_0, adjs_1).int()
        data_list = []
        for i in tqdm(range(len(adjs))):
            data = {
                'x': raw_Xs[i],
                'y': labels[i],
                'edge_index': torch.stack(torch.nonzero(adjs[i], as_tuple=True)),
            }
            data = Data(**data)
            data_list.append(data)

    elif fuse_type in ['fuse_embed', 'fuse_pred']:
        adjs_0, adjs_1 = adj_weight2bin(adjs, ratio_sc, ratio_fc)
        data_list = []
        for i in tqdm(range(len(adjs_0))):
            data = {
                'x': raw_Xs[i],
                'y': labels[i],
                'edge_index_sc': torch.stack(torch.nonzero(adjs_0[i], as_tuple=True)),
                'edge_index_fc': torch.stack(torch.nonzero(adjs_1[i], as_tuple=True)),
            }
            data = Data(**data)
            data_list.append(data)

    return data_list

def get_fuse_type(model_name: str):
    if 'fuse_embed' in model_name:
        fuse_type = 'fuse_embed'
    elif 'fuse_graph' in model_name:
        fuse_type = 'fuse_graph'
    if 'fuse_pred' in model_name:
        fuse_type = 'fuse_pred'
    return fuse_type

"""
adapted from dgl.knn_graph, but add some constraints
"""
from dgl import DGLError, remove_self_loop, remove_edges, EID
from dgl.base import dgl_warning
from dgl.transforms.functional import pairwise_squared_distance
from dgl.sampling import sample_neighbors
from dgl.transforms.functional import convert
import dgl.backend as dglF

def knn_graph(
    x, null_idx, k, algorithm="bruteforce-blas", dist="euclidean", exclude_self=False, null_filter=True
):
    """
    null_filter: if true, when constructing similarity matrix, similarity items between null graphs will
        be set to inf so that null graphs are not connected. this is to prevent noises of null graphs being
        propagated to other null graphs.
    """
    if exclude_self:
        # add 1 to k, for the self edge, since it will be removed
        k = k + 1

    # check invalid k
    if k <= 0:
        raise DGLError("Invalid k value. expect k > 0, got k = {}".format(k))

    # check empty point set
    x_size = tuple(dglF.shape(x))
    if x_size[0] == 0:
        raise DGLError("Find empty point set")

    d = dglF.ndim(x)
    x_seg = x_size[0] * [x_size[1]] if d == 3 else [x_size[0]]
    if algorithm == "bruteforce-blas":
        result = _knn_graph_blas(x, null_idx, k, dist=dist, null_filter=null_filter)

    if exclude_self:
        # remove_self_loop will update batch_num_edges as needed
        result = remove_self_loop(result)

        # If there were more than k(+1) coincident points, there may not have been self loops on
        # all nodes, in which case there would still be one too many out edges on some nodes.
        # However, if every node had a self edge, the common case, every node would still have the
        # same degree as each other, so we can check that condition easily.
        # The -1 is for the self edge removal.
        clamped_k = min(k, np.min(x_seg)) - 1
        if result.num_edges() != clamped_k * result.num_nodes():
            # edges on any nodes with too high degree should all be length zero,
            # so pick an arbitrary one to remove from each such node
            degrees = result.in_degrees()
            node_indices = dglF.nonzero_1d(degrees > clamped_k)
            edges_to_remove_graph = sample_neighbors(
                result, node_indices, 1, edge_dir="in"
            )
            edge_ids = edges_to_remove_graph.edata[EID]
            result = remove_edges(result, edge_ids)

    return result



def _knn_graph_blas(x, null_idx, k, dist="euclidean", null_filter=True):
    if dglF.ndim(x) == 2:
        x = dglF.unsqueeze(x, 0)
    n_samples, n_points, _ = dglF.shape(x)

    if k > n_points:
        dgl_warning(
            "'k' should be less than or equal to the number of points in 'x'"
            "expect k <= {0}, got k = {1}, use k = {0}".format(n_points, k)
        )
        k = n_points

    # if use cosine distance, normalize input points first
    # thus we can use euclidean distance to find knn equivalently.
    if dist == "cosine":
        l2_norm = lambda v: dglF.sqrt(dglF.sum(v * v, dim=2, keepdims=True))
        x = x / (l2_norm(x) + 1e-5)

    ctx = dglF.context(x)
    dist = pairwise_squared_distance(x)
    # Jiahang: revise such that null graphs not in neighbors
    if null_filter:
        null_idx_2d = (null_idx.unsqueeze(-1).float() @ null_idx.unsqueeze(0).float()).bool()
        dist[:, null_idx_2d] = torch.inf
        dist[:, null_idx, null_idx] = 0.

    k_indices = dglF.astype(dglF.argtopk(dist, k, 2, descending=False), dglF.int64)
    # index offset for each sample
    offset = dglF.arange(0, n_samples, ctx=ctx) * n_points
    offset = dglF.unsqueeze(offset, 1)
    src = dglF.reshape(k_indices, (n_samples, n_points * k))
    src = dglF.unsqueeze(src, 0) + offset
    dst = dglF.repeat(dglF.arange(0, n_points, ctx=ctx), k, dim=0)
    dst = dglF.unsqueeze(dst, 0) + offset
    return convert.graph((dglF.reshape(src, (-1,)), dglF.reshape(dst, (-1,))))
            
            
        
        


