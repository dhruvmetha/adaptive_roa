#!/usr/bin/env bash
# FLAGSHIP.md finalists for q3d-lqr, at most 2 GPU processes on cuda:1, then resume STAGE2.
set -u
cd "$(dirname "$0")" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PYTHONPATH=$(cd ../../../.. && pwd):$(cd ../.. && pwd)
PY=${PY:-/common/users/dm1487/envs/arcmg/bin/python}
run() { nice -n 5 "$PY" flagship.py "$@"; }
echo "[$(date '+%F %T')] flagship start" >> flagship.log
run screen viz_theta_vs_q:r2 viz_phi_vs_p:r2 viz_theta_vs_q:r3 >> flagship.log 2>&1 &
run screen viz_theta_vs_q:r1 viz_phi_vs_p:r1 viz_phi_vs_p:r3 >> flagship.log 2>&1 &
wait
echo "[$(date '+%F %T')] screening done" >> flagship.log
run full viz_theta_vs_q >> flagship.log 2>&1 &
run full viz_phi_vs_p >> flagship.log 2>&1 &
wait
run index >> flagship.log 2>&1
echo "[$(date '+%F %T')] FLAGSHIP DONE; resuming STAGE2 run_stage2.py" >> flagship.log
nice -n 5 "$PY" run_stage2.py >> run.log 2>&1
echo "[$(date '+%F %T')] run_stage2.py exited $?" >> flagship.log
