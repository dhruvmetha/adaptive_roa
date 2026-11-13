#!/usr/bin/env bash
# Keep at most 2 cp-lqr all_seeds.py workers per card (cuda:0, cuda:2). While jobs are still held by
# the placeholder pid, start fresh workers with --adopt <pid> as slots free up. Fresh processes import
# the fixed infer.py (Amarel path remap), so they can run r3. No new worker on cuda:2 while the cp-ppo
# agent has processes there. Exits when no claim file names the placeholder any more.
set -u
HOLD=${1:?placeholder pid}
SL=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/common/users/dm1487/envs/arcmg/bin/python}
P=$SL/pred/cp-lqr
cd "$SL" || exit 1
declare -A BUS=( [cuda:0]=00000000:01:00.0 [cuda:2]=00000000:41:00.0 )
n=0
held() { grep -lx "$HOLD" "$P"/.claim_* 2>/dev/null | wc -l; }
ours() { pgrep -fc "^$PY pred/cp-lqr/all_seeds.py --device $1( |$)"; }
cpppo_on() {
  local bus=$1 b pid
  while read -r b pid; do
    [ "$b" = "$bus," ] || continue
    tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null | grep -q "pred/cp-ppo/" && return 0
  done < <(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader)
  return 1
}
while [ "$(held)" -gt 0 ]; do
  for dev in cuda:0 cuda:2; do
    if [ "$dev" = cuda:2 ] && cpppo_on "${BUS[$dev]}"; then continue; fi
    while [ "$(ours "$dev")" -lt 2 ] && [ "$(held)" -gt 0 ]; do
      n=$((n + 1)); tag="R${n}_${dev#cuda:}"
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=$(cd "$SL/../.." && pwd):$SL:$P \
        nohup nice -n 5 "$PY" pred/cp-lqr/all_seeds.py --device "$dev" --adopt "$HOLD" \
        > "$P/all_seeds_$tag.log" 2>&1 &
      echo "$(date '+%F %T') launched $tag pid $! on $dev (held left $(held))"
      sleep 20      # let it adopt before counting again
    done
  done
  sleep 30
done
echo "$(date '+%F %T') no held claims left; supervisor exits"
