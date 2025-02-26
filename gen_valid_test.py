from argparse import ArgumentParser
import pickle
import os
from tqdm import tqdm
import numpy as np
from utils import set_random_seed

from sklearn.model_selection import train_test_split

"""
第一个数据集中有label记录的共528个样本，其中没有待预测的标签有146个，剩下的382个中5个同时缺失SC和X，5个只缺失SC，只有372个有效样本
第二个数据集'109830', '236130', '614439'没有label, '168139'的HCPemotion_268_LR存在行常数
"""

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument("--version", type=int, default=1)

#     args = parser.parse_args()

def gen_valid_test(target_label):
    split_args = {
        'train_size': 0.6,
        'valid_size': 0.2,
        'test_size': 0.2
    }

    data_path = '/home/mufan/mohan/gnn/dataset/2'
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

    names = []

    for i, name in tqdm(enumerate(data_labels.keys())):
        if name not in data_SC.keys() or name not in data_FC.keys() or name not in data_raw_X.keys():
            print(name)
            continue
        if target_label not in data_labels[name].keys():
            print(name)
            continue

        names.append(name)

    names = np.array(names)
    train_size, valid_size, test_size = \
        split_args['train_size'], split_args['valid_size'], split_args['test_size']
    idx = np.arange(len(names))
    splits_seeds = []
    for seed in range(10):
        set_random_seed(seed)
        train_valid_idx, test_idx = \
            train_test_split(idx, test_size=test_size)
        train_idx, valid_idx = \
            train_test_split(train_valid_idx, 
                            test_size=valid_size / (train_size + valid_size))
        full_idx, miss_idx = train_test_split(train_idx, test_size=0.5)
        
        train_names, valid_names, test_names = names[train_idx], names[valid_idx], names[test_idx]
        full_names, miss_names = names[full_idx], names[miss_idx]
        
        SC_train = np.array([data_SC[name] for name in full_names])
        SC_train = SC_train.reshape(SC_train.shape[0], -1)
        FC_train = np.array([data_FC[name] for name in full_names])
        FC_train = FC_train.reshape(FC_train.shape[0], -1)
        miss_SC_knn = {}
        miss_FC_knn = {}
        k=5
        for i, name in enumerate(miss_names):
            SC = data_SC[name].reshape(-1)
            FC = data_FC[name].reshape(-1)
            SC_dist = np.linalg.norm(SC_train - SC, axis=1)
            FC_dist = np.linalg.norm(FC_train - FC, axis=1)
            SC_knn = full_names[np.argsort(SC_dist)[:k]]
            FC_knn = full_names[np.argsort(FC_dist)[:k]]
            miss_SC_knn[name] = FC_knn
            miss_FC_knn[name] = SC_knn
        
        splits = {
            'seed': seed,
            'train_names': train_names,
            'valid_names': valid_names,
            'test_names': test_names,
            'full_names': full_names,
            'miss_names': miss_names,
            'miss_SC_knn': miss_SC_knn,
            'miss_FC_knn': miss_FC_knn
        }
        splits_seeds.append(splits)

    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # dir_path = os.path.join(dir_path, 'dataset', 'valid_test_split', target_label)
    # path = os.path.join(dir_path, f'v1.pkl')
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    
    path = os.path.join(data_path, 'valid_test_split', target_label, 'v1.pkl')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'wb') as f:
        pickle.dump(splits_seeds, f)

# target_labels = ['nih_totalcogcomp_ageadjusted', 'nih_fluidcogcomp_ageadjusted', 'nih_crycogcomp_ageadjusted']
target_labels = ['PMAT24_A_CR', 'PMAT24_A_RTCR', 'PMAT24_A_SI', 'PicSeq_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj']
for target_label in target_labels:
    gen_valid_test(target_label)