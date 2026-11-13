#!/usr/bin/env bash
# ALL_SEEDS.md for q3d-lqr: 2 workers on cuda:1 now, 2 on cuda:4 once the q3d-ppo agent's
# processes have left it; then the final indexes and the STAGE2 median-seed sheet.
set -u
cd "$(dirname "$0")" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PYTHONPATH=$(cd ../../../.. && pwd):$(cd ../.. && pwd)
PY=${PY:-/common/users/dm1487/envs/arcmg/bin/python}
GPU4_UUID=GPU-6ebe6d39-37fd-e48c-3a97-c32af971c880
LOG=all_seeds.log
T0=$(date +%s)
echo "[$(date '+%F %T')] all_seeds start" >> $LOG
worker() { nice -n 5 "$PY" all_seeds.py work "$1" >> $LOG 2>&1; }
n_done() { ls viz_*__r[123].npz 2>/dev/null | grep -vc tmp; }
q3dppo_on_gpu4() {
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | tr -d ' ' |
  while IFS=, read -r pid uuid; do
    [ "$uuid" = "$GPU4_UUID" ] && ps -o cmd= -p "$pid" 2>/dev/null | grep -q "q3d-ppo" && echo "$pid"
  done
}
worker cuda:1 &
sleep 20
worker cuda:1 &
(
  while :; do
    [ "$(n_done)" -ge 18 ] && { echo "[$(date '+%F %T')] cuda:4 waiter: all jobs done, not launching" >> $LOG; exit 0; }
    [ -z "$(q3dppo_on_gpu4)" ] && break
    sleep 60
  done
  echo "[$(date '+%F %T')] cuda:4 free of q3d-ppo processes; launching 2 workers there" >> $LOG
  worker cuda:4 &
  sleep 20
  worker cuda:4 &
  wait
) &
wait
nice -n 5 "$PY" all_seeds.py final >> $LOG 2>&1
echo "[$(date '+%F %T')] ALL DONE: $(n_done)/18 preds, wall $(( $(date +%s) - T0 ))s" >> $LOG
