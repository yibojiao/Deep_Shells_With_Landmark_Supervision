#!/bin/bash
#SBATCH --account=ai4d-core-01 -J deepshell
#SBATCH --partition=TrixieMain
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
hostname
/usr/bin/nvidia-smi

srun python main.py

