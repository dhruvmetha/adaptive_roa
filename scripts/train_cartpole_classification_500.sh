#!/bin/bash
#SBATCH --job-name=cartpole_cls_500
#SBATCH --output=outputs/cartpole_cls_500_%j.out
#SBATCH --error=outputs/cartpole_cls_500_%j.err
#SBATCH --time=4:00:00
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

# Run training for 500 epochs
python src/classification/train_cartpole.py \
    trainer.max_epochs=500 \
    trainer.devices=[0] \
    name=cartpole_classification_500ep

echo "Training complete!"

