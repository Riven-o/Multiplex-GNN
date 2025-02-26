import os
import subprocess
import time
import queue
import threading
# 配置路径和设备列表
config_dir = "/home/mufan/mohan/gnn/configs"
device_list = [0,1,2,4,5,6,7]
dataset = 2
# target_labels = ['nih_totalcogcomp_ageadjusted', 'nih_fluidcogcomp_ageadjusted', 'nih_crycogcomp_ageadjusted']
target_labels = ['PMAT24_A_CR', 'PMAT24_A_RTCR', 'PMAT24_A_SI', 'PicSeq_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj']


seeds = list(range(10))  # seed=0,1,2,...,9
completed_tasks_file = "completed_tasks.txt"  # 用来存储已完成的任务

miss_list = [['none', 'pad'], ['none', 'drop'],
             ['fc', 'pad'], ['fc', 'knn'],
             ['sc', 'pad'], ['sc', 'knn']]

# 获取已完成任务的记录
def load_completed_tasks():
    if os.path.exists(completed_tasks_file):
        with open(completed_tasks_file, 'r') as f:
            completed_tasks = set(line.strip() for line in f)
    else:
        completed_tasks = set()
    return completed_tasks

# 保存已完成任务的记录
def save_completed_task(task_name):
    with open(completed_tasks_file, 'a') as f:
        f.write(task_name + "\n")

# 获取所有配置文件
def get_config_files():
    # config_files = []
    # for filename in os.listdir(config_dir):
    #     if filename.endswith("_best.yaml"):
    #         model_name = filename.replace("_best.yaml", "")
    #         config_files.append(model_name)
    config_files = os.listdir('/home/mufan/mohan/gnn/configs')
    config_files = [config_file.replace("_best.yaml", "") for config_file in config_files]
    return config_files

# 加载已完成的任务
completed_tasks = load_completed_tasks()

# 获取所有模型配置
benchmark_names = get_config_files()
miss_names = ['Mew', 'MHGCN', 'NeuroPath']

# 创建一个队列用于管理任务
task_queue = queue.Queue()

# 遍历每个模型和目标标签，获取剩余的任务
for target_label in target_labels:
    for model in benchmark_names:
        for seed in seeds:
            task_name = f"dataset_{dataset}_{model}_{target_label}_seed{seed}"
            if task_name not in completed_tasks:
                command = "CUDA_VISIBLE_DEVICES={0} " + f"python run_wandb.py --wandb normal --config configs/{model}_best.yaml --project_name {'benchmark_' + target_label} --seed {seed} --save_checkpoint --checkpoint_path checkpoints/benchmark/dataset_{dataset}/{target_label}/{model}/{seed}.pkl --target_label {target_label} --dataset {dataset} --results_path results/benchmark_dataset_{dataset}.json"
                task_queue.put(command)

for target_label in target_labels:
    for model in miss_names:
        for miss in miss_list:
            for seed in seeds:
                task_name = f"dataset_{dataset}_{model}_{target_label}_seed{seed}_{miss[0]}_{miss[1]}"
                if task_name not in completed_tasks:
                    command = "CUDA_VISIBLE_DEVICES={0} " + f"python run_wandb.py --wandb normal --config configs/{model}_best.yaml --project_name {'miss_' + target_label} --seed {seed} --save_checkpoint --checkpoint_path checkpoints/miss/dataset_{dataset}/{target_label}/{model}/{miss[0]}/{miss[1]}/{seed}.pkl --target_label {target_label} --miss {miss[0]} --gen_miss {miss[1]} --dataset {dataset} --results_path results/miss_dataset_{dataset}.json"
                    task_queue.put(command)


# 用于调度和运行任务的函数
def run_tasks(device_id):
    while not task_queue.empty():
        command = task_queue.get()
        command = command.format(device_id)
        print(f"Running command: {command}")
        
        # 启动任务并等待完成
        process = subprocess.Popen(command, shell=True)
        process.communicate()  # 等待任务完成
        
        print(f"Finished command: {command}")
        task_queue.task_done()  # 标记任务为已完成
        
# 启动多个进程来运行任务
threads = []
for device_id in device_list:
    thread = threading.Thread(target=run_tasks, args=(device_id,))
    thread.start()
    threads.append(thread)
    
# 等待所有任务完成
for thread in threads:
    thread.join()