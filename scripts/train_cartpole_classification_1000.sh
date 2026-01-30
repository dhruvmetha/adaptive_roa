#!/bin/bash
#SBATCH --job-name=cartpole_cls_1000
#SBATCH --output=outputs/cartpole_cls_1000_%j.out
#SBATCH --error=outputs/cartpole_cls_1000_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Activate environment
source /common/users/rm1838/miniforge3/bin/activate adaptive_roa

# Change to project directory
cd /common/users/rm1838/adaptive_classification/adaptive_roa

# Create output directory if it doesn't exist
mkdir -p outputs

# Run training for 1000 epochs
python src/classification/train_cartpole.py \
    trainer.max_epochs=1000 \
    scheduler.T_max=1000 \
    trainer.devices=[0] \
    name=cartpole_classification_1000ep

echo "Training complete!"

