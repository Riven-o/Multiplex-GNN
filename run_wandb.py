import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['WANDB_API_KEY'] = '3f9ee2a779655a1a6b91a47982ed685b55f4d6ee'
import wandb
import yaml
from argparse import ArgumentParser

from utils import set_random_seed, SINGLE_MODALITY_MODELS, FUSE_SINGLE_MODALITY_MODELS, FUSE_SINGLE_MODALITY_MODELS_NOSIA

project_name = 'graph_test'

def modify_project_name(new_project_name):
    global project_name
    project_name = new_project_name

def main(seed=0, config=None):
    set_random_seed(seed)
    if config is not None:
        wandb.init(project=project_name, config=config)
    else:
        wandb.init(project=project_name)
    config = wandb.config
    if config.model_name in \
        ['MHGCN', 'NeuroPath', 'Mew', 'MewCustom', 'MewFuseGraph', 'MHGCNFuseGraph'] \
            + SINGLE_MODALITY_MODELS \
            + FUSE_SINGLE_MODALITY_MODELS \
            + FUSE_SINGLE_MODALITY_MODELS_NOSIA:
        from GNN import pipe
    elif config.model_name in ['CIVAE']:
        from CIVAE import pipe
    elif config.model_name in ['DMGI']:
        from DMGI import pipe
    # pipe(config)
    best_train_rmse, best_valid_rmse, best_test_rmse = pipe(config)
    import json
    import os
    path = config.results_path
    if os.path.exists(path):
        with open(path, 'r') as f:
            results = json.load(f)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results = {}

    target_label = config.target_label
    model_name = config.model_name
    miss = config.miss
    gen_miss = config.gen_miss

    if target_label not in results:
        results[target_label] = {}
    if model_name not in results[target_label]:
        results[target_label][model_name] = {}
    if miss not in results[target_label][model_name]:
        results[target_label][model_name][miss] = {}
    if gen_miss not in results[target_label][model_name][miss]:
        results[target_label][model_name][miss][gen_miss] = {}
    seed = str(seed)
    if seed not in results[target_label][model_name][miss][gen_miss]:
        results[target_label][model_name][miss][gen_miss][seed] = {}
    results[target_label][model_name][miss][gen_miss][seed]['best_train_rmse'] = best_train_rmse
    results[target_label][model_name][miss][gen_miss][seed]['best_valid_rmse'] = best_valid_rmse
    results[target_label][model_name][miss][gen_miss][seed]['best_test_rmse'] = best_test_rmse

    with open(path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--wandb', type=str, choices=['sweep', 'normal', 'repeat'], default='normal')
    parser.add_argument('--config', type=str, default='./configs/NeuroPath_best.yaml')
    parser.add_argument('--project_name', type=str, default='graph_test')
    parser.add_argument('--save_checkpoint', action='store_true', default=False)
    parser.add_argument('--load_checkpoint', action='store_true', default=False)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='2') # 1, 2
    parser.add_argument('--target_label', type=str, default='PMAT24_A_CR')
    parser.add_argument('--miss', type=str, default='none')   # none, fc, sc
    parser.add_argument('--gen_miss', type=str, default='pad') # knn, pad, drop
    parser.add_argument('--results_path', type=str, default='results.json')
    
    
    # target_label = 'nih_totalcogcomp_ageadjusted'
    # target_label = 'nih_fluidcogcomp_ageadjusted'
    # target_label = 'nih_crycogcomp_ageadjusted'
    
    # target_label = 'PMAT24_A_CR'
    # target_label = 'PMAT24_A_RTCR'
    # target_label = 'PMAT24_A_SI'
    # target_label = 'PicSeq_AgeAdj'
    # target_label = 'PicVocab_AgeAdj'
    # target_label = 'ProcSpeed_AgeAdj'
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    modify_project_name(args.project_name)
    config['seed'] = args.seed
    config['save_checkpoint'] = args.save_checkpoint
    config['load_checkpoint'] = args.load_checkpoint
    config['checkpoint_path'] = args.checkpoint_path
    config['dataset'] = args.dataset
    config['target_label'] = args.target_label
    config['miss'] = args.miss
    config['gen_miss'] = args.gen_miss
    config['results_path'] = args.results_path

    if args.wandb == 'sweep':
        sweep_id = wandb.sweep(sweep=config, project=project_name)
        print(f"sweep id: {sweep_id}")
        wandb.agent(sweep_id, function=main)
    elif args.wandb == 'normal':
        main(args.seed, config=config)
    elif args.wandb == 'repeat':
        for seed in range(5):
            main(seed, config=config)
