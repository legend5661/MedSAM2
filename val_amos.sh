#!/bin/bash
#SBATCH -t 7-00:0:0
#SBATCH -J medsam2-tr-tiny-single
#SBATCH --mem=80G
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -o out_single_gpu_tiny.out

export PATH=/usr/local/cuda/bin:$PATH
timestamp=$(date +"%Y%m%d-%H%M")

# Print GPU information
echo "Running on single GPU"

config=configs/val_amos.yaml
output_path=./exp_log/val_amos_$timestamp

# Run training on single GPU
python training/train.py \
        -c $config \
        --output-path $output_path \
        --use-cluster 0 \
        --num-gpus 1 \
        --num-nodes 1

echo "val completed. Output saved to $output_path"