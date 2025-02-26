import datetime, os, torch
import numpy as np
from sklearn.model_selection import train_test_split

import CIVAE_src.ci_ivae_main as CI_iVAE

import torch
import torch.nn as nn
import wandb

from MHGCN_src import MHGCN
from utils import set_random_seed, load_dataset, Evaluator, EarlyStopping, device
from sklearn import svm

def pipe(configs):
    x_concat_adj = configs['x_concat_adj']
    epochs  = configs['epochs']
    split_args = configs['split_args']
    label_type = 'regression'
    target_label = configs.get('target_label', 'nih_totalcogcomp_ageadjusted')

    assert label_type in ['classification', 'regression']
    if label_type == 'classification':
        out_dim = (max(labels)+1).item()
    else:
        out_dim = 1

    adjs, raw_Xs, labels, splits = load_dataset(split_args=split_args, label_type=label_type, target_label=target_label)
    train_idx, valid_idx, test_idx = splits['train_idx'], splits['valid_idx'], splits['test_idx']    

    # raw_Xs = raw_Xs.reshape(raw_Xs.shape[0] * raw_Xs.shape[1], -1)
    raw_Xs = raw_Xs.reshape(-1, raw_Xs.shape[1] * raw_Xs.shape[2])
    
    labels = torch.unsqueeze(labels, 1)

    if x_concat_adj:
        raw_Xs = torch.concatenate([raw_Xs, adjs.flatten(1)], dim=-1)
    in_dim = raw_Xs.shape[-1]
    

    evaluator = Evaluator(label_type=label_type, num_classes=out_dim, device=device)
    
    # build CI-iVAE networks
    dim_x, dim_u = in_dim, 1
    ci_ivae = CI_iVAE.model(dim_x=dim_x, dim_u=dim_u)

    # train CI-iVAE networks. Results will be saved at the result_path
    now = datetime.datetime.now()
    result_path = './results/ci_ivae-time=%d-%d-%d-%d-%d' % (now.month, now.day, now.hour, now.minute, now.second)
    CI_iVAE.fit(model=ci_ivae, x_train=raw_Xs[train_idx], u_train=labels[train_idx],
                x_val=raw_Xs[valid_idx], u_val=labels[valid_idx], 
                num_epoch=epochs, result_path=result_path, num_worker=2,
                )

    # extract features with trained CI-iVAE networks
    z_train = CI_iVAE.extract_feature(result_path=result_path, x=raw_Xs[train_idx])
    z_valid = CI_iVAE.extract_feature(result_path=result_path, x=raw_Xs[valid_idx])
    z_test = CI_iVAE.extract_feature(result_path=result_path, x=raw_Xs[test_idx])
    z_train = z_train.detach().cpu().numpy()
    z_valid = z_valid.detach().cpu().numpy()
    z_test = z_test.detach().cpu().numpy()

    # evaluation with svr
    svr = svm.SVR()
    svr.fit(z_train, labels[train_idx].flatten().numpy())

    pred_train = svr.predict(z_train)
    pred_valid = svr.predict(z_valid)
    pred_test = svr.predict(z_test)

    pred_train, pred_valid, pred_test = \
        torch.tensor(pred_train), \
        torch.tensor(pred_valid), \
        torch.tensor(pred_test)
    evaluator = Evaluator(label_type=label_type, num_classes=out_dim, device='cpu')
    train_rmse = evaluator.evaluate(pred_train, labels[train_idx].flatten())
    valid_rmse = evaluator.evaluate(pred_valid, labels[valid_idx].flatten())
    test_rmse = evaluator.evaluate(pred_test, labels[test_idx].flatten())

    print(f"train rmse {train_rmse:.4f} | valid rmse {valid_rmse:.4f} | test rmse {test_rmse:.4f}")

    return train_rmse, valid_rmse, test_rmse
    


if __name__ == '__main__':
    log_idx = 1
    train, valid, test = [], [], []
    for seed in range(5):
        set_random_seed(seed)
        searchSpace = {
                    "epochs": 100,
                    "split_args": {
                        'train_size': 0.6,
                        'valid_size': 0.2,
                        'test_size': 0.2,
                    },
                    "use_wandb": False,
                    "x_concat_adj": False
                }
        # run = wandb.init(
        #     # Set the project where this run will be logged
        #     project="multiplex gnn",
        #     # Track hyperparameters and run metadata
        #     config=searchSpace
        # )
        best_train_rmse, best_val_rmse, best_test_rmse = pipe(searchSpace)
        train.append(best_train_rmse)
        valid.append(best_val_rmse)
        test.append(best_test_rmse)

    with open(f'./logs/log_{log_idx}.txt', 'a') as f:
            f.write(f"CIVAE: ")
            f.write(f'best_train_rmse: {np.mean(train):.4f}±{np.std(train):.4f} | '
                    f'best_val_rmse: {np.mean(valid):.4f}±{np.std(valid):.4f} | '
                    f'best_test_rmse: {np.mean(test):.4f}±{np.std(test):.4f}\n')