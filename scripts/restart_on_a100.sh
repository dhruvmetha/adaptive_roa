#!/bin/bash
# Restart 7 experiments on A100 GPUs

BASE_DIR="/cache/home/rm1838/adaptive_main_roa"
PYTHON="/home/rm1838/miniconda3/envs/adaptive_roa/bin/python"
# Override EXP_DIR to use /cache path (avoid /common/home which doesn't exist)
export EXP_DIR="/home/rm1838/adaptive_main_roa/outputs"
export DATA_DIR="/home/rm1838"
COMMON_ARGS="trainer.max_epochs=1000 conformal.alpha_sampling=0.1 conformal.use_conformal=true"

echo "Submitting 7 experiments on A100s (--constraint=ampere)..."

# Run 1: Non-adaptive (d2_ratio=0), 18 iterations
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_r1 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=50 n_epochs=18 \
    sampling_mode=conformal d2_ratio=0.0 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run1_a100.log 2>&1 &
echo "Run 1 (non-adaptive, d2=0.0) submitted"

sleep 2

# Run 2: Conformal, d2=0.5, 9 iterations
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_r2 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=conformal d2_ratio=0.5 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run2_a100.log 2>&1 &
echo "Run 2 (conformal, d2=0.5) submitted"

sleep 2

# Run 3: Conformal, d2=0.75, 9 iterations
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_r3 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=conformal d2_ratio=0.75 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run3_a100.log 2>&1 &
echo "Run 3 (conformal, d2=0.75) submitted"

sleep 2

# Run 4: Direct, d2=0.5, 9 iterations
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_r4 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=direct d2_ratio=0.5 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run4_a100.log 2>&1 &
echo "Run 4 (direct, d2=0.5) submitted"

sleep 2

# Run 5: Direct, d2=0.75, 9 iterations
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_r5 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=direct d2_ratio=0.75 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run5_a100.log 2>&1 &
echo "Run 5 (direct, d2=0.75) submitted"

sleep 2

# Run 6: Ranked, d2=0.5, 9 iterations
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_r6 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=ranked d2_ratio=0.5 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run6_a100.log 2>&1 &
echo "Run 6 (ranked, d2=0.5) submitted"

sleep 2

# Run 7: Ranked, d2=0.75, 9 iterations
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_r7 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=ranked d2_ratio=0.75 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run7_a100.log 2>&1 &
echo "Run 7 (ranked, d2=0.75) submitted"

echo ""
echo "All 7 jobs submitted with --constraint=ampere"
echo "Check status with: squeue -u rm1838"
