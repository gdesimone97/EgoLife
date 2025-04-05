#!/bin/bash

#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

srun --gres=gpu:1 python3 infer.py