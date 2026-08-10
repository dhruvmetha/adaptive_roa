#!/bin/bash
# Scalar-outcome FM ablation arm. One stage per submission, ascending noise.
#
#   sbatch scripts/sbatch_outcome_fm.sh deterministic
#   sbatch scripts/sbatch_outcome_fm.sh low
#   sbatch scripts/sbatch_outcome_fm.sh med|high|xhigh
#
# Spec: docs/superpowers/specs/2026-08-10-outcome-fm-ablation-design.md
#SBATCH --job-name=of_stage
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=48:00:00
#SBATCH --mem=40G
set -e
cd /common/home/st1122/Projects/adaptive_roa

LEVEL="${1:?usage: sbatch scripts/sbatch_outcome_fm.sh <deterministic|low|med|high|xhigh> [seed]}"
SEED="${2:-42}"

# Seed replicates are what make a verdict non-provisional. D1 selection is
# deterministic (dataset_builder takes a sorted prefix of a fixed on-disk
# shuffle, no RNG), so changing the seed varies ONLY training -- weight init and
# the x0 draws -- on byte-identical data. That is precisely the run-to-run floor
# the attribution needs as a denominator, and it is the same construction the
# stoch_compare campaign used for its fm/clf seed families.

# ~/.local numpy shadows the env's and silently hangs multiprocessing jobs at 0% GPU.
export PYTHONNOUSERSITE=1

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Stage: $LEVEL"

# The deterministic pendulum is a DIFFERENT system config from the stochastic one,
# not a noise_level of it. Stage 1 is a correctness check only: true p is in {0,1}
# there, so calibration is degenerate and it answers nothing about the ablation.
if [ "$LEVEL" = "deterministic" ]; then
    SYSTEM_ARGS=(system=pendulum)
    RUN_NAME="fm_outcome_deterministic"
else
    SYSTEM_ARGS=(system=pendulum_stoch "noise_level=$LEVEL")
    RUN_NAME="fm_outcome_$LEVEL"
fi
# Seed 42 is the base run and keeps the bare name, matching the stoch_compare
# convention (fm_med_dir00 / fm_med_dir00_s43 / _s44) that seed_runs() expects.
[ "$SEED" != "42" ] && RUN_NAME="${RUN_NAME}_s${SEED}"

echo "Run: $RUN_NAME  (seed $SEED)"

./env/bin/python scripts/run_adaptive.py \
    +experiment=fm_outcome_baseline \
    "${SYSTEM_ARGS[@]}" \
    seed="$SEED" \
    exp_dir=/common/users/shared/pracsys/adaptive_roa_experiments/outcome_fm \
    name="$RUN_NAME"

echo "Done: $(date)"
