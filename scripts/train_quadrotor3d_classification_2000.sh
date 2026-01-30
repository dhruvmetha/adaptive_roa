#!/bin/bash
# Train Quadrotor 3D classifier for 2000 epochs
# Uses first 1000 trajectories from shuffled_indices_0.txt with sample balancing

cd /common/users/rm1838/adaptive_classification/adaptive_roa

python src/classification/train_quadrotor3d.py \
    name=quadrotor3d_cls_2000ep \
    trainer.max_epochs=2000 \
    data.max_samples=1000 \
    data.balance_samples=true \
    scheduler.T_max=2000 \
    trainer.callbacks.1.patience=200
