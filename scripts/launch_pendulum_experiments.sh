#!/bin/bash
# Launch 7 Pendulum Adaptive Sampling Experiments
# Each run gets its own GPU via separate srun commands
#
# Terminology:
#   - n_epochs = number of adaptive sampling ITERATIONS
#   - trainer.max_epochs = flow matcher training epochs PER iteration (1000)

# Base directory
BASE_DIR="/cache/home/rm1838/adaptive_main_roa"
CONDA_ENV="/home/rm1838/miniconda3/envs/adaptive_roa"
# Override EXP_DIR and DATA_DIR to use correct paths
export EXP_DIR="/home/rm1838/adaptive_main_roa/outputs"
export DATA_DIR="/home/rm1838"
PYTHON="${CONDA_ENV}/bin/python"

# Set PYTHONPATH to ensure correct package is imported
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Output base directory (override .env paths)
OUTPUT_BASE="/cache/home/rm1838/adaptive_main_roa/outputs"

# Common settings (1000 training epochs per iteration)
COMMON_ARGS="trainer.max_epochs=1000 conformal.alpha_sampling=0.1 conformal.use_conformal=true"

# Create logs directory
mkdir -p "${BASE_DIR}/experiment_logs"

echo "=============================================="
echo "Launching 7 Pendulum Experiments"
echo "=============================================="
echo "Each iteration trains flow matcher for 1000 epochs"
echo ""

# Run 1: Non-adaptive (d2_ratio=0), 18 iterations (100 + 18*50 = 1000 total)
echo "[Run 1] Non-adaptive: initial=100, per_iter=50, iterations=18, d2_ratio=0 (1000 total)"
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_nonadapt --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=50 n_epochs=18 \
    sampling_mode=conformal d2_ratio=0.0 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run1_nonadaptive.log 2>&1 &

sleep 5

# Run 2: Conformal, d2=0.5, 9 iterations
echo "[Run 2] Conformal: initial=100, per_iter=100, iterations=9, d2_ratio=0.5"
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_conf_05 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=conformal d2_ratio=0.5 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run2_conformal_d2_0.5.log 2>&1 &

sleep 5

# Run 3: Conformal, d2=0.75, 9 iterations
echo "[Run 3] Conformal: initial=100, per_iter=100, iterations=9, d2_ratio=0.75"
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_conf_075 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=conformal d2_ratio=0.75 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run3_conformal_d2_0.75.log 2>&1 &

sleep 5

# Run 4: Direct, d2=0.5, 9 iterations
echo "[Run 4] Direct: initial=100, per_iter=100, iterations=9, d2_ratio=0.5"
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_dir_05 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=direct d2_ratio=0.5 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run4_direct_d2_0.5.log 2>&1 &

sleep 5

# Run 5: Direct, d2=0.75, 9 iterations
echo "[Run 5] Direct: initial=100, per_iter=100, iterations=9, d2_ratio=0.75"
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_dir_075 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=direct d2_ratio=0.75 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run5_direct_d2_0.75.log 2>&1 &

sleep 5

# Run 6: Ranked, d2=0.5, 9 iterations
echo "[Run 6] Ranked: initial=100, per_iter=100, iterations=9, d2_ratio=0.5"
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_rank_05 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=ranked d2_ratio=0.5 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run6_ranked_d2_0.5.log 2>&1 &

sleep 5

# Run 7: Ranked, d2=0.75, 9 iterations
echo "[Run 7] Ranked: initial=100, per_iter=100, iterations=9, d2_ratio=0.75"
srun -p gpu --gres=gpu:1 --constraint=ampere --mem=32G -t 48:00:00 -c 8 -J pend_rank_075 --export=ALL,PYTHONPATH=${BASE_DIR} \
    ${PYTHON} ${BASE_DIR}/scripts/run_adaptive_pendulum.py \
    initial_train_size=100 samples_per_epoch=100 n_epochs=9 \
    sampling_mode=ranked d2_ratio=0.75 \
    ${COMMON_ARGS} \
    > ${BASE_DIR}/experiment_logs/run7_ranked_d2_0.75.log 2>&1 &

echo ""
echo "=============================================="
echo "All 7 jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u rm1838"
echo "Logs in: ${BASE_DIR}/experiment_logs/"
echo ""
echo "To check a log: tail -f ${BASE_DIR}/experiment_logs/run1_nonadaptive.log"
