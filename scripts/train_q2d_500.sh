#!/bin/bash
#SBATCH --job-name=q2d_500ep
#SBATCH --output=logs/q2d_500ep_%j.out
#SBATCH --error=logs/q2d_500ep_%j.err
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G

source /common/users/rm1838/miniforge3/bin/activate adaptive_roa
cd /common/users/rm1838/adaptive_classification/adaptive_roa
python src/classification/train_quadrotor2d.py trainer.max_epochs=500 name=q2d_500ep


