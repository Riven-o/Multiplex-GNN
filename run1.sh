#!/bin/bash

# python GNN.py --online_split=False &
# python GNN.py --file_option=_miss_graph --online_split=False &

# python run_wandb.py --wandb normal --config configs/GCN_best.yaml --project_name local_multiplex_test --seed 0 \
#     --save_checkpoint --checkpoint_path checkpoints/GCN/seed=0.pkl

# python run_wandb.py --wandb normal --config configs/GCN_best.yaml --project_name local_multiplex_test --seed 0 \
#     --load_checkpoint --checkpoint_path checkpoints/GCN/seed=0.pkl

device_st=0
device_end=7
device=${device_st}
# model=GCN
# models=( GCN SAGE SGC GAT GCN_fuse_embed SAGE_fuse_embed SGC_fuse_embed GAT_fuse_embed MHGCN Mew NeuroPath )
models=( MewFuseGraph_fuse_method_GAT_missLabel_labelPropFalse MewFuseGraph_fuse_method_SAGE MewFuseGraph_fuse_method_mean)
seeds=( {0..9..1} )
for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
        CUDA_VISIBLE_DEVICES=${device} python run_wandb.py --wandb normal --config configs/${model}_best.yaml --project_name multiplex-reproduce-3 --seed $seed --save_checkpoint --checkpoint_path checkpoints/${model}/seed=${seed}.pkl &
        device=$(( device + 1 ))
        if [ ${device} -eq $(( device_end + 1 )) ]; then
            device=${device_st}
            wait
        fi
    done
done