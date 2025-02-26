import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import wandb
import numpy as np
from argparse import ArgumentParser

from MHGCN_src import MHGCN, MHGCNFuseGraph
from NeuroPath_src import DetourTransformer, Transformer, GAT, Vanilla
from custom_src import VanillaFuse, GATFuse, VanillaFuseNoSia, GATFuseNoSia, \
    SIGN_pred, MewCustom, MewFuseGraph, LabelProp
from utils import set_random_seed, load_dataset, model_infer, \
    Evaluator, EarlyStopping, SINGLE_MODALITY_MODELS, \
    FUSE_SINGLE_MODALITY_MODELS, FUSE_SINGLE_MODALITY_MODELS_NOSIA, \
    to_pyg_single, split_pyg, to_pyg_fuse, device, get_fuse_type, \
    pyg_preprocess_sign, load_dataset1, load_dataset2
from torch_geometric.data import Batch

def pipe(configs: dict):
    hid_dim = configs['hid_dim']
    nlayers = configs['nlayers']
    lr = configs['lr']
    wd = configs['wd']
    epochs  = configs['epochs']
    patience = configs['patience']
    if patience == -1:
        patience = torch.inf
    split_args = configs['split_args']
    use_wandb = configs['use_wandb']
    model_name = configs['model_name']
    reduce = configs['reduce']
    dropout = configs['dropout']

    file_option = configs.get('file_option', "")
    reload = configs.get('reload', False)
    label_type = configs.get('label_type', 'regression')
    eval_type = configs.get('eval_type', 'split')
    seed = configs.get('seed', 0)
    valid_test_version = configs.get('valid_test_version', 1)
    # online_split = configs.get('online_split', True) # deprecated: we need offline split!
    online_split = configs.get('online_split', False) # this is default now!
    label_prop_option = configs.get('label_prop', False)
    load_checkpoint = configs.get('load_checkpoint', False)
    save_checkpoint = configs.get('save_checkpoint', False)
    checkpoint_path = configs.get('checkpoint_path', None)
    dataset = configs.get('dataset', '1')
    target_label = configs.get('target_label', 'nih_totalcogcomp_ageadjusted')
    miss = configs.get('miss', 'none')
    gen_miss = configs.get('gen_miss', 'pad')
    # adjs, raw_Xs, labels, splits, mu_lbls, std_lbls, no_sc_idx, no_fc_idx = \
    # results = \
    #     load_dataset(split_args=split_args, label_type=label_type, 
    #                  eval_type=eval_type, reload=reload, 
    #                  file_option=file_option, seed=seed, 
    #                  version=valid_test_version, online_split=online_split)
    
    # results = \
    #     load_dataset1(split_args=split_args, label_type=label_type, 
    #                  eval_type=eval_type, file_option=file_option, seed=seed, 
    #                  version=valid_test_version, online_split=online_split, target_label=target_label)
    
    # temp_split_args = {'test_size': 0.2, 'train_size': 0.6, 'valid_size': 0.2}
    # dir_path = '/home/mufan/mohan/gnn/dataset/1/old_running_datasets'
    # if temp_split_args == split_args and label_type == 'regression' \
    #     and eval_type == 'split' and valid_test_version == 1 and not online_split:
    #     data_name = f'{file_option}_{seed}_{target_label}'
    #     data_path = os.path.join(dir_path, data_name)
    #     if os.path.exists(data_path):
    #         results = torch.load(data_path)
    #     else:
    #         results = \
    #             load_dataset1(split_args=split_args, label_type=label_type, 
    #                         eval_type=eval_type, file_option=file_option, seed=seed, 
    #                         version=valid_test_version, online_split=online_split, target_label=target_label)
    #         torch.save(results, data_path)
    # else:
    #     results = \
    #         load_dataset1(split_args=split_args, label_type=label_type, 
    #                     eval_type=eval_type, file_option=file_option, seed=seed, 
    #                     version=valid_test_version, online_split=online_split, target_label=target_label)
        
    dir_path = f'/home/mufan/mohan/gnn/dataset/{dataset}/running_datasets'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_name = f'{seed}_{miss}_{gen_miss}_{target_label}'
    data_path = os.path.join(dir_path, data_name)
    if os.path.exists(data_path):
        results = torch.load(data_path)
    else:
        results = \
            load_dataset2(split_args=split_args, label_type=label_type, 
                        eval_type=eval_type, file_option=file_option, seed=seed, 
                        version=valid_test_version, online_split=online_split,
                        target_label=target_label, miss=miss, gen_miss=gen_miss, dataset=dataset)
        torch.save(results, data_path)

        
    # raw_Xs: n_graphs * n_nodes * n_features
    # adjs: n_graphs * n_types * n_nodes * n_nodes
    # labels: n_graphs
    no_sc_idx, no_fc_idx = None, None
    if file_option == "":
        adjs, raw_Xs, labels, splits, mu_lbls, std_lbls = results
    elif file_option == "_miss_graph":
        adjs, raw_Xs, labels, splits, mu_lbls, std_lbls, no_sc_idx, no_fc_idx = results
        no_sc_idx = no_sc_idx.to(device)
        no_fc_idx = no_fc_idx.to(device)
    elif file_option == "_miss_graph_miss_label":
        adjs, raw_Xs, labels, splits, mu_lbls, std_lbls, no_sc_idx, no_fc_idx, no_lbl_idx = results
        no_sc_idx = no_sc_idx.to(device)
        no_fc_idx = no_fc_idx.to(device)
        no_lbl_idx = no_lbl_idx.to(device)
    train_idx, valid_idx, test_idx = splits['train_idx'], splits['valid_idx'], splits['test_idx']  
    train_idx, valid_idx, test_idx = \
        train_idx.to(device), valid_idx.to(device), test_idx.to(device)
    in_dim = raw_Xs.shape[-1]

    assert label_type in ['classification', 'regression']
    if label_type == 'classification':
        out_dim = (max(labels)+1).item()
    else:
        out_dim = 1

    data = None
    """
    loading model: output (num_graphs, num_classes)
    """
    if model_name == 'MHGCN':   # two adj channels fused by two learnable weights at graph level, then go through GCN as single adj with node feature
        model = MHGCN(nfeat=in_dim, nlayers=nlayers, nhid=hid_dim, out=out_dim, dropout=dropout)
        # nlayers: number of GCN layers
        adjs = adjs.to(device)
        raw_Xs = raw_Xs.to(device)

    elif model_name == 'Mew':   # two seperate n_hop FFN process each adj channel with node feature to get two node embeddings, then fuse with learnable weights, finally pool for graph pred
        adjs = adjs.to(device)
        raw_Xs = raw_Xs.to(device)
        # nlayers: number of hops, layer_i means i-hop adjacency matrix
        data = pyg_preprocess_sign(raw_Xs, adjs, nlayers)
        model = SIGN_pred(num_feat=in_dim, num_graph_tasks=out_dim, 
                          num_layer=nlayers, emb_dim=hid_dim, drop_ratio=dropout, 
                          graph_pooling=reduce,
                          attn_weight=configs['attn_weight'],
                          shared=configs['shared'])
        
    elif model_name == "MewCustom":
        adjs = adjs.to(device)
        raw_Xs = raw_Xs.to(device)
        
        k = configs.get('supp_k', 5)
        fuse_type = configs.get('fuse_type', 'normal') # normal has no fusion

        model = MewCustom(num_feat=in_dim, num_graph_tasks=out_dim, 
                          num_layer=nlayers, emb_dim=hid_dim, drop_ratio=dropout, 
                          graph_pooling=reduce,
                          attn_weight=configs['attn_weight'],
                          shared=configs['shared'], k=k, fuse_type=fuse_type)
    elif model_name == 'MewFuseGraph':
        adjs = adjs.to(device)
        raw_Xs = raw_Xs.to(device)

        k = configs.get('supp_k', 5)
        knn_on = configs.get('knn_on', "graph_embed")
        fuse_on = configs.get('fuse_on', "node_embed") # the best option
        fuse_method = configs.get('fuse_method', "mean")
        add_self_loop = configs.get('add_self_loop', False)
        null_filter = configs.get('null_filter', True)
        fusion_only_on_null = configs.get('fusion_only_on_null', False)

        model = MewFuseGraph(num_feat=in_dim, num_graph_tasks=out_dim, 
                            num_layer=nlayers, emb_dim=hid_dim, drop_ratio=dropout, 
                            graph_pooling=reduce,
                            attn_weight=configs['attn_weight'],
                            shared=configs['shared'], 
                            k=k, knn_on=knn_on, fuse_on=fuse_on, fuse_method=fuse_method, 
                            gnn_add_self_loop=add_self_loop, null_filter=null_filter,
                            fusion_only_on_null=fusion_only_on_null)
    elif model_name == 'MHGCNFuseGraph':
        adjs = adjs.to(device)
        raw_Xs = raw_Xs.to(device)

        k = configs.get('supp_k', 5)
        knn_on = configs.get('knn_on', "graph_embed")
        fuse_on = configs.get('fuse_on', "node_embed")
        fuse_method = configs.get('fuse_method', "mean")

        model = MHGCNFuseGraph(nfeat=in_dim, nlayers=nlayers, nhid=hid_dim, out=out_dim, 
                               dropout=dropout, k=k, fuse_method=fuse_method, 
                               knn_on=knn_on, fuse_on=fuse_on, 
                               shared=configs['shared'],
                               combine_type=configs['combine_type'])
            
    elif model_name in ['NeuroPath'] + SINGLE_MODALITY_MODELS:
        ratio_sc = configs.get('ratio_sc', 0.2)
        ratio_fc = configs.get('ratio_fc', 0.2)
        ratio = configs.get('ratio', 0.2)

        # data_list: list of each subject data, will output both sc and fc binary adj
        if model_name in SINGLE_MODALITY_MODELS:
            # use edge_index of target modality denoted by 'option'
            data_list = to_pyg_single(raw_Xs, labels, adjs, 
                                      ratio_sc=ratio, ratio_fc=ratio, option=configs['modality'])
        else: # NeuroPath
            # use adj_sc and adj_fc binary adj info, and edge_index is denoted by 'fc'
            data_list = to_pyg_single(raw_Xs, labels, adjs, 
                                      ratio_sc=ratio_sc, ratio_fc=ratio_fc, option='fc')
        data = Batch.from_data_list(data_list).to(device)

        if model_name == 'NeuroPath':
            model = DetourTransformer(num_nodes = raw_Xs.shape[1], in_dim = in_dim, 
                                      nclass = out_dim, hid_dim = hid_dim, 
                                      nlayers = nlayers, dropout=dropout)
        elif model_name in ['GCN', 'SAGE', 'SGC', 'GIN']:
            model = Vanilla(model_name=model_name, in_dim=in_dim, hid_dim=hid_dim, 
                            nlayers=nlayers, dropout=dropout, reduce=reduce, nclass=out_dim)
        elif model_name == 'GAT':
            model = GAT(in_dim=in_dim, hid_dim=hid_dim, nlayers=nlayers, 
                        dropout=dropout, reduce=reduce, nclass=out_dim)
        elif model_name == 'Transformer':
            model = Transformer(in_dim = in_dim, hid_dim = hid_dim, nclass = out_dim)
    
    elif model_name in FUSE_SINGLE_MODALITY_MODELS + FUSE_SINGLE_MODALITY_MODELS_NOSIA:
        ratio_sc = configs.get('ratio_sc', 0.2)
        ratio_fc = configs.get('ratio_fc', 0.2)
        ratio = configs.get('ratio', 0.2)
        reduce_fuse = configs.get('reduce_fuse', 'mean')

        fuse_type = get_fuse_type(model_name)
        data_list = to_pyg_fuse(raw_Xs, labels, adjs, 
                                fuse_type=fuse_type, reduce_fuse=reduce_fuse,
                                ratio_sc=ratio_sc, ratio_fc=ratio_fc, ratio=ratio)
        data = Batch.from_data_list(data_list).to(device)

        if 'GAT' in model_name:
            if model_name in FUSE_SINGLE_MODALITY_MODELS:
                model = GATFuse(model_name=model_name, in_dim=in_dim, hid_dim=hid_dim, 
                            nlayers=nlayers, dropout=dropout, nclass=out_dim, 
                            reduce_nodes=reduce, reduce_fuse=reduce_fuse)
            else:
                model = GATFuseNoSia(model_name=model_name, in_dim=in_dim, hid_dim=hid_dim, 
                            nlayers=nlayers, dropout=dropout, nclass=out_dim, 
                            reduce_nodes=reduce, reduce_fuse=reduce_fuse)
            
        else:
            if model_name in FUSE_SINGLE_MODALITY_MODELS:
                model = VanillaFuse(model_name=model_name, in_dim=in_dim, hid_dim=hid_dim, 
                            nlayers=nlayers, dropout=dropout, nclass=out_dim, 
                            reduce_nodes=reduce, reduce_fuse=reduce_fuse)
            else:
                model = VanillaFuseNoSia(model_name=model_name, in_dim=in_dim, hid_dim=hid_dim, 
                            nlayers=nlayers, dropout=dropout, nclass=out_dim, 
                            reduce_nodes=reduce, reduce_fuse=reduce_fuse)
    """
    loading model - end
    """

    """
    loading label propagation model
    """

    if label_prop_option:
        label_prop = LabelProp()
    """
    loading label propagation model - end
    """
    model = model.to(device)                            
    labels = labels.to(device)

    if label_type == 'classification':
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    evaluator = Evaluator(mu_lbls=mu_lbls, std_lbls=std_lbls, 
                          label_type=label_type, num_classes=out_dim, device=device)

    best_train_rmse = torch.inf
    best_val_rmse = torch.inf
    best_test_rmse = torch.inf
    cnt = 0
    ori_labels = labels.clone()
    if '_miss_label' in file_option and not label_prop_option:
        train_idx = train_idx * ~no_lbl_idx # only labeled data being computed loss against labels
    ori_train_idx = train_idx.clone()

    if load_checkpoint: # Evaluation
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        logits = model_infer(model, model_name, adjs=adjs,
                            raw_Xs=raw_Xs, data=data, 
                            no_sc_idx=no_sc_idx, no_fc_idx=no_fc_idx)
        if label_type == 'classification':
            valid_acc, valid_auroc, valid_auprc = evaluator.evaluate(logits[valid_idx], labels[valid_idx])
            test_acc, test_auroc, test_auprc = evaluator.evaluate(logits[test_idx], labels[test_idx])
            print(f"Valid Acc {valid_acc:.4f} | Valid Auroc {valid_auroc:.4f} | Valid Auprc: {valid_auprc:.4f}")
            print(f"Test Acc {test_acc:.4f} | Test Auroc {test_auroc:.4f} | Test Auprc: {test_auprc:.4f}")
            
            if use_wandb:
                wandb.log({
                    'valid_acc': valid_acc,
                    'valid_auroc': valid_auroc,
                    'valid_auprc': valid_auprc,

                    'test_acc': test_acc,
                    'test_auroc': test_auroc,
                    'test_auprc': test_auprc,
                })
        else:
            valid_rmse = evaluator.evaluate(logits[valid_idx], labels[valid_idx])
            test_rmse = evaluator.evaluate(logits[test_idx], labels[test_idx])

            print(f"Valid RMSE {valid_rmse:.4f} | Test RMSE {test_rmse:.4f}")

            if use_wandb:
                wandb.log({
                    'valid_rmse': valid_rmse,
                    'test_rmse': test_rmse,
                })

    else: # Training
        for epoch in range(epochs):
            if epoch == 1000 and model_name == 'MHGCN':
                for g in optimizer.param_groups:
                    g['lr'] *= 1e-1
            logits = model_infer(model, model_name, adjs=adjs,
                                raw_Xs=raw_Xs, data=data, 
                                no_sc_idx=no_sc_idx, no_fc_idx=no_fc_idx)
            if label_prop_option:
                labels = ori_labels.clone()
                train_idx = ori_train_idx.clone()
                labels, train_idx = label_prop(labels, no_lbl_idx, model.knn_sc, model.knn_fc, train_idx)
            loss = loss_fn(logits[train_idx], labels[train_idx])

            if label_type == 'classification':
                train_acc, train_auroc, train_auprc = evaluator.evaluate(logits[train_idx], labels[train_idx])
                print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc:.4f} | Train Auroc {train_auroc:.4f} | "
                        f"Train Auprc: {train_auprc:.4f}")
                if use_wandb:
                    wandb.log({
                        'train_acc': train_acc,
                        'train_auroc': train_auroc,
                        'train_auprc': train_auprc,
                    })
            else:
                train_rmse = evaluator.evaluate(logits[train_idx], labels[train_idx])
                print(f"Epoch {epoch:05d} | Loss (calib RMSE) {loss.item():.4f} | Train RMSE {train_rmse:.4f}")
                if use_wandb:
                    wandb.log({
                        'train_rmse': train_rmse,
                        'train_loss': loss.item(),
                    })

            if epoch % 5 == 0:
                if label_type == 'classification':
                    valid_acc, valid_auroc, valid_auprc = evaluator.evaluate(logits[valid_idx], labels[valid_idx])
                    test_acc, test_auroc, test_auprc = evaluator.evaluate(logits[test_idx], labels[test_idx])
                    print(f"Valid Acc {valid_acc:.4f} | Valid Auroc {valid_auroc:.4f} | Valid Auprc: {valid_auprc:.4f}")
                    print(f"Test Acc {test_acc:.4f} | Test Auroc {test_auroc:.4f} | Test Auprc: {test_auprc:.4f}")
                    
                    if use_wandb:
                        wandb.log({
                            'valid_acc': valid_acc,
                            'valid_auroc': valid_auroc,
                            'valid_auprc': valid_auprc,

                            'test_acc': test_acc,
                            'test_auroc': test_auroc,
                            'test_auprc': test_auprc,
                        })
                else:
                    valid_rmse = evaluator.evaluate(logits[valid_idx], labels[valid_idx])
                    test_rmse = evaluator.evaluate(logits[test_idx], labels[test_idx])

                    print(f"Valid RMSE {valid_rmse:.4f} | Test RMSE {test_rmse:.4f}")

                    if valid_rmse < best_val_rmse:
                        best_train_rmse = train_rmse
                        best_val_rmse = valid_rmse
                        best_test_rmse = test_rmse
                        cnt = 0

                        if save_checkpoint:
                            par = os.path.dirname(checkpoint_path)
                            if not os.path.exists(par):
                                os.makedirs(par)
                            torch.save(model.state_dict(), checkpoint_path)
                    else:
                        cnt += 1

                    if use_wandb:
                        wandb.log({
                            'valid_rmse': valid_rmse,
                            'test_rmse': test_rmse,
                        })

                    if cnt >= patience:
                        break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if use_wandb:
            wandb.log({
                'best_train_rmse': best_train_rmse,
                'best_val_rmse': best_val_rmse,
                'best_test_rmse': best_test_rmse,
            })
        return best_train_rmse.item(), best_val_rmse.item(), best_test_rmse.item()

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--file_option', type=str, choices=['', '_miss_graph'], default="")
    parser.add_argument('--online_split', type=str, choices=['True', 'False'], default=False)

    args = parser.parse_args()

    log_idx = 1
    model_name = 'MHGCNFuseGraph'
    seed=0
    set_random_seed(seed)
    searchSpace = {
                # "hid_dim": 64,
                "hid_dim": 2,
                "lr": 1e-2,
                "epochs": 2000,
                "patience": 10,
                "wd": 1e-2,
                "nlayers": 2,
                # "nlayers": 1,
                "split_args": {
                    'train_size': 0.6,
                    'valid_size': 0.2,
                    'test_size': 0.2,
                },
                "dropout": 0.5,
                "modality": 'sc',
                "ratio_sc": 0.2,
                "ratio_fc": 0.2,
                "ratio": 0.2,
                "reduce": "mean",
                "reduce_fuse": "concat",
                "use_wandb": False,
                "model_name": model_name,
                "label_type": "regression",
                "attn_weight": True, 
                "shared": False,
                # "reload": True,
                # "file_option": "",
                # "file_option": "_miss_graph",
                "file_option": "_miss_graph_miss_label",
                # "file_option": args.file_option,
                "supp_k": 2,
                # "fuse_type": "unit_miss",
                "knn_on": "graph_embed",
                "fuse_on": "node_embed",
                "fuse_method": "GCN",
                "label_prop": False,
                "fusion_only_on_null": True,
                "combine_type": "elementwise",
                # "add_self_loop": True,
                # "null_filter": False
                # "online_split": False
                # "online_split": True if args.online_split == 'True' else False
            }
    if searchSpace['use_wandb']:
        run = wandb.init(
            # Set the project where this run will be logged
            project="multiplex gnn",
            # Track hyperparameters and run metadata
            config=searchSpace
        )
    best_train_rmse, best_val_rmse, best_test_rmse = pipe(searchSpace)
    print(f"BEST RMSE: train - {best_train_rmse:.4f} | valid - {best_val_rmse:.4f} | test - {best_test_rmse:.4f}")

    # with open(f'./logs/log_{log_idx}.txt', 'a') as f:
    #     f.write(f"{searchSpace['model_name']}: ")
    #     f.write(f'best_train_rmse: {np.mean(train):.4f}±{np.std(train):.4f} | '
    #             f'best_val_rmse: {np.mean(valid):.4f}±{np.std(valid):.4f} | '
    #             f'best_test_rmse: {np.mean(test):.4f}±{np.std(test):.4f}\n')