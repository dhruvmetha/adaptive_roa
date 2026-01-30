#!/bin/bash
#SBATCH --partition=unlimited
#SBATCH --job-name=q3d_1000
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_q3d_1000_%j.out
#SBATCH --error=slurm_q3d_1000_%j.err
#SBATCH --time=08:00:00

source /common/users/rm1838/miniforge3/bin/activate adaptive_roa
cd /common/users/rm1838/adaptive_classification/adaptive_roa
python src/classification/train_quadrotor3d.py trainer.max_epochs=1000 name=q3d_1000ep_batch


