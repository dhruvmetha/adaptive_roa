#!/bin/bash
# Mountain Car Experiment on GPU 7
# Parameters: init=1000, epochs=1, warm_start=false, opt=delta, 
#             incr_data=50, d2_ratio=0.5, train_epochs=500, lr=0.0005, batch=1024

# Initialize conda from miniforge
eval "$(/common/users/rm1838/miniforge3/bin/conda shell.bash hook)"
conda activate adaptive_roa

cd /common/users/rm1838/adaptive_cartpole

export CUDA_VISIBLE_DEVICES=7

python adaptive_roa/adaptive/run_adaptive_mountain_car.py \
    initial_train_size=1000 \
    n_epochs=1 \
    warm_start=false \
    conformal.optimize_mode=delta \
    adaptive_data_max=50 \
    d2_ratio=0.5 \
    trainer.max_epochs=500 \
    optimizer.lr=0.0005 \
    batch_size=1024

echo "Experiment completed at $(date)"
