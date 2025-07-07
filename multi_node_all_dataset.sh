#!/bin/bash
# filepath: /staff/wangtiantong/MedSAM2/simple_multi_gpu_train.sh

# 设置环境变量
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=2,3  

# 记录时间戳
timestamp=$(date +"%Y%m%d-%H%M")
echo "开始训练时间: $timestamp"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# 配置参数
config=configs/train_all_dataset.yaml
output_path=./exp_log/all_dataset_$timestamp
num_gpus=2

# 使用torchrun替代torch.distributed.launch
# torchrun默认使用环境变量而不是命令行参数传递rank信息
python training/train.py \
        -c $config \
        --output-path $output_path \
        --use-cluster 0 \
        --num-gpus $num_gpus \
        --num-nodes 1

echo "训练完成,改过了trainer,所有数据集的,初始条件帧数量为3,修改了data_loader那里，第二次修改了sampler"
echo "结束时间: $(date +"%Y%m%d-%H%M")"