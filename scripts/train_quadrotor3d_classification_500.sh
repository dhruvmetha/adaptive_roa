#!/bin/bash
# Train Quadrotor 3D classifier for 500 epochs
# Uses first 1000 trajectories from shuffled_indices_0.txt with sample balancing

cd /common/users/rm1838/adaptive_classification/adaptive_roa

python src/classification/train_quadrotor3d.py \
    name=quadrotor3d_cls_500ep \
    trainer.max_epochs=500 \
    data.max_samples=1000 \
    data.balance_samples=true \
    scheduler.T_max=500
