model=GCN
device=7
seed=0
# target_label = 'nih_totalcogcomp_ageadjusted'
# target_label = 'nih_fluidcogcomp_ageadjusted'
target_label = 'nih_crycogcomp_ageadjusted'
CUDA_VISIBLE_DEVICES=${device} python run_wandb.py --wandb normal --config configs/${model}_best.yaml --project_name ${target_label} --seed $seed --save_checkpoint --checkpoint_path checkpoints/${model}/${seed}/${target_label}.pkl --target_label ${target_label} 

下面是我启动训练脚本的一个bash命令，现在我想写一个python脚本自动在多个设备上启动不同训练：
CUDA_VISIBLE_DEVICES=${device} python run_wandb.py --wandb normal --config configs/${model}_best.yaml --project_name ${target_label} --seed $seed --save_checkpoint --checkpoint_path checkpoints/${model}/${seed}/${target_label}.pkl --target_label ${target_label} 
1. 首先遍历目录"/home/mufan/mohan/gnn/configs"下的所有yaml文件，文件名以"_best"结尾的文件中删除"_best"后缀，得到模型名称model
2. target_label的取值可能为: 'nih_fluidcogcomp_ageadjusted', 'nih_crycogcomp_ageadjusted'
3. 对于每个model的两种target_label，分别在seed=0,1,2,3,4,5,6,7,8,9上重复训练
4. 指定一个device_list，比如：[2,3,5,6]，分别在这些设备上启动训练，每个设备上同时只能有一个训练任务，当某个设备上训练完成后，再在该设备上启动下一个训练任务
5. 注意python脚本应具有异常终止后重新运行时能继续未完成的训练任务的功能，即训练过程中如果中断python文件，再次运行时应继续完成尚未完成的训练任务，已经完成的训练任务不需要再次运行