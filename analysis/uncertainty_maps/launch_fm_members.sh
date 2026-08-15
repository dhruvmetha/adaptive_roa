#!/bin/bash
# Recompute per-member p(success) over the eval grid for every flow-matching arm
# whose per-epoch checkpoints survived: fm_high and fm_xhigh (5 arms each).
#
# One job per (level, arm) -- ~19 epochs x 5 members x K sweeps. At the measured
# 0.35 s per full-grid sweep that is roughly an hour per job, and splitting this
# way keeps any single job small enough to schedule on the shared partition
# instead of holding one card for half a day.
#
# K=100 matches the evaluation's num_mc_samples_eval. Acquisition itself ran at
# K=20, where each member's MC noise inflates raw between-member variance by
# mean[p(1-p)]/(K-1) and BALD by ~(1/2K)(1-1/M) ~ 0.02 nats; at K=100 that floor
# is 5x lower, so these maps show member disagreement rather than sampling noise.
set -euo pipefail

BASE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps
CODE=$BASE/code
OUT=$BASE/members
SCRIPTS=$BASE/jobs
EXPROOT=/common/users/shared/pracsys/adaptive_roa_experiments/ensemble_epistemic
REPO=/common/home/st1122/Projects/adaptive_roa
PY=$REPO/env/bin/python
LOGS=$REPO/slurm_logs
K=${K:-100}

mkdir -p "$OUT" "$LOGS" "$SCRIPTS"

for LEVEL in high xhigh; do
  for ARM in dir00 total aleat epi_var epi_bald; do
    RUN="fm_${LEVEL}_${ARM}"
    if [ ! -d "$EXPROOT/$RUN" ]; then
      echo "skip $RUN (absent)"
      continue
    fi
    S="$SCRIPTS/${RUN}.sbatch"
    {
      echo '#!/bin/bash'
      echo "#SBATCH --job-name=um_${LEVEL}_${ARM}"
      echo '#SBATCH --partition=unlimited'
      echo '#SBATCH --gres=gpu:1'
      echo '#SBATCH --mem=40G'
      echo '#SBATCH --time=12:00:00'
      echo "#SBATCH --output=${LOGS}/um_${RUN}_%j.out"
      echo "#SBATCH --error=${LOGS}/um_${RUN}_%j.err"
      echo 'export PYTHONNOUSERSITE=1'
      echo "export PYTHONPATH=${CODE}"
      echo "cd ${CODE}"
      echo "srun ${PY} analysis/uncertainty_maps/compute_members.py --run ${RUN} --out ${OUT} --device cuda --k ${K} --batch 16384"
    } > "$S"
    chmod +x "$S"
    sbatch "$S"
  done
done
