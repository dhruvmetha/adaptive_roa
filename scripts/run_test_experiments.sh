#!/bin/bash
# Test runs for CartPole and Mountain Car
# Uses GPUs 4, 5, 6 (last 4 GPUs as requested)

# Set up conda
source /koko/system/anaconda/etc/profile.d/conda.sh
conda activate /common/users/rm1838/miniforge3/envs/adaptive_roa

cd /common/users/rm1838/adaptive_cartpole
export PYTHONPATH=/common/users/rm1838/adaptive_cartpole:$PYTHONPATH

# Run 1: CartPole - batch_size=1024 on GPU 4
echo "Starting CartPole run on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python src/adaptive/run_adaptive_cartpole.py \
    initial_train_size=1000 \
    n_epochs=1 \
    warm_start=false \
    conformal.optimize_mode=delta \
    adaptive_data_max=50 \
    d2_ratio=0.5 \
    trainer.max_epochs=500 \
    optimizer.lr=0.001 \
    batch_size=1024 \
    > logs/cartpole_bs1024.log 2>&1 &
PID1=$!
echo "CartPole PID: $PID1"

# Run 2: Mountain Car - batch_size=512 on GPU 5
echo "Starting Mountain Car (bs=512) run on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python src/adaptive/run_adaptive_mountain_car.py \
    initial_train_size=1000 \
    n_epochs=1 \
    warm_start=false \
    conformal.optimize_mode=delta \
    adaptive_data_max=50 \
    d2_ratio=0.5 \
    trainer.max_epochs=500 \
    optimizer.lr=0.001 \
    batch_size=512 \
    > logs/mountain_car_bs512.log 2>&1 &
PID2=$!
echo "Mountain Car (bs=512) PID: $PID2"

# Run 3: Mountain Car - batch_size=1024 on GPU 6
echo "Starting Mountain Car (bs=1024) run on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python src/adaptive/run_adaptive_mountain_car.py \
    initial_train_size=1000 \
    n_epochs=1 \
    warm_start=false \
    conformal.optimize_mode=delta \
    adaptive_data_max=50 \
    d2_ratio=0.5 \
    trainer.max_epochs=500 \
    optimizer.lr=0.001 \
    batch_size=1024 \
    > logs/mountain_car_bs1024.log 2>&1 &
PID3=$!
echo "Mountain Car (bs=1024) PID: $PID3"

echo ""
echo "All 3 runs started!"
echo "PIDs: CartPole=$PID1, MC_bs512=$PID2, MC_bs1024=$PID3"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/cartpole_bs1024.log"
echo "  tail -f logs/mountain_car_bs512.log"
echo "  tail -f logs/mountain_car_bs1024.log"
