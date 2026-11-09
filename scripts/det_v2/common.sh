# Shared setup for every det_v2 script. Source it; do not run it.
#
# Loads the per-user settings from scripts/det_v2/user.env (or $DET_V2_USER_ENV), fills
# the campaign defaults, sets umask 002 so the shared root stays group-writable, and
# defines the ledger, manifest and liveness helpers. user.env.example documents every key.

DET_V2_SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DET_V2_USER_ENV=${DET_V2_USER_ENV:-$DET_V2_SCRIPTS/user.env}

die() { echo "$*" >&2; exit 1; }

[ -f "$DET_V2_USER_ENV" ] || die "no $DET_V2_USER_ENV: copy scripts/det_v2/user.env.example to it and fill it in"
set -a
# shellcheck disable=SC1090
source "$DET_V2_USER_ENV"
set +a

: "${NET_ID:?user.env must set NET_ID}"
: "${PYTHON:?user.env must set PYTHON}"
: "${SNAPSHOT_DIR:?user.env must set SNAPSHOT_DIR}"
EXP_DIR=${EXP_DIR:-/common/users/shared/pracsys/adaptive_roa_experiments}
DATA_DIR=${DATA_DIR:-/common/users/shared/pracsys/genMoPlan/data_trajectories}
SHARED_DATA_BASE=${SHARED_DATA_BASE:-$DATA_DIR}
USER_BASE=${USER_BASE:-/common/users/shared/pracsys/genMoPlan/global_dynamics_experiments}
# Campaign: det_v2 by default; the scripts/det_v3/ wrappers set det_v3. It picks the shared
# root (and with it the ledger, logs and hold list), the manifest and the Amarel log dir, so
# two campaigns never share a ledger even though they reuse run names.
DET_CAMPAIGN=${DET_CAMPAIGN:-det_v2}
case $DET_CAMPAIGN in det_v[0-9]*) ;; *) die "DET_CAMPAIGN must be det_v2, det_v3, ..., got '$DET_CAMPAIGN'" ;; esac
# The run configs put every run under ${exp_dir}/<campaign>, so the root is derived, not set.
DET_V2_ROOT=$EXP_DIR/$DET_CAMPAIGN
LEDGER=${LEDGER:-$DET_V2_ROOT/launched.tsv}
LOG_DIR=${LOG_DIR:-$DET_V2_ROOT/logs}
HOLD_HOSTS=${HOLD_HOSTS:-$DET_V2_ROOT/hold_hosts.txt}
# det_v2 keeps its unprefixed tmux sessions; later campaigns tag theirs (run names repeat).
SESSION_TAG=$([ "$DET_CAMPAIGN" = det_v2 ] || echo "${DET_CAMPAIGN#det_}_")
SHARED_GROUP=${SHARED_GROUP:-login-bekris}
TMP_ROOT=${TMP_ROOT:-/tmp}
DET_V2_REPO=${DET_V2_REPO:-$(cd "$DET_V2_SCRIPTS/../.." && pwd)}
MANIFEST=${MANIFEST:-$(dirname "$DET_V2_SCRIPTS")/$DET_CAMPAIGN/manifest.tsv}
# Amarel logs: det_v2 keeps the user.env dir; later campaigns get a subdir, so log pulls stay apart.
[ "$DET_CAMPAIGN" = det_v2 ] || [ -z "${AMAREL_LOG_DIR:-}" ] || AMAREL_LOG_DIR=$AMAREL_LOG_DIR/$DET_CAMPAIGN
ILAB_SUBMIT_HOST=${ILAB_SUBMIT_HOST:-ilab1.cs.rutgers.edu}
ILAB_PARTITION=${ILAB_PARTITION:-unlimited}
ILAB_EXCLUDE=${ILAB_EXCLUDE:-ilab3,ilab4}
ILAB_CPU_THREADS=${ILAB_CPU_THREADS:-8}
# iLab resumes use short walls (tracker: queue age earns no priority there, and a 6-7 h wall
# fits the backfill window); a run continues from its last banked epoch at each wall.
ILAB_RESUME_WALL=${ILAB_RESUME_WALL:-06:00:00}
DIRECT_HOST_SUFFIX=${DIRECT_HOST_SUFFIX-.cs.rutgers.edu}
DIRECT_TASKSET=${DIRECT_TASKSET:-}
DIRECT_TMP_ROOT=${DIRECT_TMP_ROOT:-}
DIRECT_JOBS_TSV=${DIRECT_JOBS_TSV:-}
SHARED_CARDS=${SHARED_CARDS:-}
GPU_FREE=${GPU_FREE:-$HOME/.claude/skills/compute/scripts/gpu-free.sh}
SSH_CLEAR_AGENT=${SSH_CLEAR_AGENT:-0}
AMAREL_HOST=${AMAREL_HOST:-amarel.rutgers.edu}
AMAREL_USER=${AMAREL_USER:-$NET_ID}
AMAREL_ACCOUNT=${AMAREL_ACCOUNT:-general}
AMAREL_PARTITION=${AMAREL_PARTITION:-gpu}
AMAREL_EXCLUDE=${AMAREL_EXCLUDE:-gpu029,gpu046}
# Amarel's maintenance reservation (scontrol show reservation maintenance-202611): no Amarel job runs
# at or after the start and before the end, so an 'outage' row stamped inside it frees is_live.
AMAREL_OUTAGE_START=${AMAREL_OUTAGE_START:-2026-11-10T08:00:00}
AMAREL_OUTAGE_END=${AMAREL_OUTAGE_END:-2026-11-11T23:59:00}
# gpu029 throws ECC faults and kills jobs in under a minute; cgpu has no /scratch, so jobs die
# in 0 s with an empty log. Refuse a user.env that would send work there.
case ",$AMAREL_EXCLUDE," in *,gpu029,*) ;; *) die "AMAREL_EXCLUDE must keep gpu029 (ECC faults): $AMAREL_EXCLUDE" ;; esac
case ",$AMAREL_PARTITION," in *,cgpu,*) die "AMAREL_PARTITION must not include cgpu (no /scratch there): $AMAREL_PARTITION" ;; esac
AMAREL_CPUS=${AMAREL_CPUS:-4}
DET_V2_TMUX_WAIT=${DET_V2_TMUX_WAIT:-3}
export EXP_DIR DATA_DIR SHARED_DATA_BASE USER_BASE DET_V2_ROOT LEDGER LOG_DIR HOLD_HOSTS SHARED_GROUP \
  TMP_ROOT DET_V2_REPO MANIFEST ILAB_SUBMIT_HOST ILAB_PARTITION ILAB_EXCLUDE ILAB_CPU_THREADS ILAB_RESUME_WALL \
  DIRECT_HOST_SUFFIX DIRECT_TASKSET DIRECT_TMP_ROOT DIRECT_JOBS_TSV SHARED_CARDS GPU_FREE SSH_CLEAR_AGENT \
  AMAREL_HOST AMAREL_USER AMAREL_ACCOUNT AMAREL_PARTITION AMAREL_EXCLUDE AMAREL_CPUS DET_V2_TMUX_WAIT
if [ "$SSH_CLEAR_AGENT" = 1 ]; then export SSH_AUTH_SOCK=; fi   # westeros: the agent hangs key auth
export DET_V2_LOADED=1
umask 002

# Ledger columns, in order. Append-only: the last row for a name is its current state.
#   action: claim (launch in progress) | launch | resume | abandon (submit failed) | cancel (moved off)
#           | outage (Amarel job the last snapshots showed RUNNING or ended TIMEOUT, PREEMPTED or
#             NODE_FAIL, written inside the reservation; log column squeue:<state> or sacct:<state>)
LEDGER_HEADER=$'name\tuser\twhere\tjob\thostgpu\ttimestamp\tlog\tcommit\trun_dir\taction'

need_amarel() {
  local k
  for k in AMAREL_CLONE AMAREL_PYTHON AMAREL_SNAPSHOT_DIR AMAREL_EXP_DIR AMAREL_DATA_DIR AMAREL_LOG_DIR; do
    [ -n "${!k:-}" ] || die "user.env must set $k for Amarel"
  done
}

rsh() { ssh -n -o BatchMode=yes "$@"; }               # one remote command, stdin closed
direct_fqdn() { echo "$1$DIRECT_HOST_SUFFIX"; }
on_host() {                                           # on_host <short host> <command>: local or ssh
  if [ "$(hostname -s)" = "$1" ]; then bash -c "$2"; else rsh "$(direct_fqdn "$1")" "$2"; fi
}
taskset_for() {                                       # "host=0-9 other=0-3" -> taskset prefix
  local pair
  for pair in $DIRECT_TASKSET; do
    [ "${pair%%=*}" = "$1" ] && { echo "taskset -c ${pair#*=}"; return; }
  done
  echo ""
}
tmp_root_for() {                                      # "arrakis=/nfs/tmp" -> that host's TMP_ROOT
  local pair root=$TMP_ROOT
  for pair in $DIRECT_TMP_ROOT; do
    local host=${pair%%=*} dir=${pair#*=}
    if [[ "$pair" != *=* || -z "$host" || "$dir" != /* ]]; then
      echo "DIRECT_TMP_ROOT entry '$pair' is not host=/absolute/path" >&2; return 1
    fi
    [ "$host" = "$1" ] && root=$dir
  done
  echo "$root"
}
stamp() { date +%Y-%m-%dT%H:%M:%S; }

# --- manifest -------------------------------------------------------------------------
manifest_get() {                                      # manifest_get <name> <column>
  awk -F'\t' -v n="$1" -v c="$2" 'NR==1{for(i=1;i<=NF;i++) if($i==c) k=i; next} $2==n{print $k; exit}' "$MANIFEST"
}

# --- ledger ---------------------------------------------------------------------------
ledger_last() {                                       # latest ledger row for a name, or nothing
  [ -f "$LEDGER" ] || return 0
  awk -F'\t' -v n="$1" 'NR>1 && $1==n' "$LEDGER" | tail -1
}
lf() { cut -f"$2" <<< "$1"; }                         # field of a ledger row: lf "$ROW" 3
L_USER=2 L_WHERE=3 L_JOB=4 L_HOSTGPU=5 L_LOG=7 L_COMMIT=8 L_RUNDIR=9 L_ACTION=10

# The ledger is written from several hosts and both accounts over NFS. fd 9 is only the
# lock; every row is written through a fresh open taken after the lock is held, so the
# append lands at the server's end of file (close-to-open) rather than a client's cached
# size. The lock is taken with short non-blocking retries: a blocking flock on NFS backs
# off to tens of seconds per retry, so a host racing another host's loop starved past a
# 120 s wait (2026-11-06: 4 of 25 arrakis appends died against westeros).
ledger_lock() {                                       # ledger_lock: flock fd 9 within LEDGER_LOCK_WAIT s
  local deadline=$(( SECONDS + ${LEDGER_LOCK_WAIT:-900} ))
  until flock -n 9; do
    [ "$SECONDS" -lt "$deadline" ] || return 1
    sleep "0.$(( RANDOM % 4 + 1 ))"
  done
}

ledger_append() {                                     # ledger_append <10 fields>; serialized by flock
  [ $# -eq 10 ] || die "ledger_append takes 10 fields, got $#"
  mkdir -p "$(dirname "$LEDGER")"
  local line; line=$(IFS=$'\t'; echo "$*")
  (
    ledger_lock || die "could not lock $LEDGER"
    { [ -s "$LEDGER" ] || printf '%s\n' "$LEDGER_HEADER"; printf '%s\n' "$line"; } >> "$LEDGER"
  ) 9>>"$LEDGER"
}

# ledger_claim <name> <where> <hostgpu> <commit> <run_dir>: append a claim row unless the
# name already has a live claim. Check and append hold one lock, so two accounts launching
# the same name cannot both pass. DET_V2_RELAUNCH=1 claims over an earlier row.
ledger_claim() {
  mkdir -p "$(dirname "$LEDGER")"
  (
    ledger_lock || exit 2
    last=$(awk -F'\t' -v n="$1" 'NR>1 && $1==n' "$LEDGER" | tail -1)
    act=$(cut -f10 <<< "$last")
    if [ -n "$last" ] && [ "$act" != abandon ] && [ "${DET_V2_RELAUNCH:-0}" != 1 ] \
       && ! { [ "${DET_V2_RERUN_PLACED:-0}" = 1 ] && [ "$act" != claim ]; }; then
      echo "$1 is already in the ledger ($(cut -f2-4 <<< "$last" | tr '\t' ' '), $(cut -f10 <<< "$last")); resume it, or set DET_V2_RELAUNCH=1 if it has nothing banked" >&2
      exit 3
    fi
    { [ -s "$LEDGER" ] || printf '%s\n' "$LEDGER_HEADER"
      printf '%s\t%s\t%s\t-\t%s\t%s\t-\t%s\t%s\tclaim\n' "$1" "$NET_ID" "$2" "$3" "$(stamp)" "$4" "$5"; } >> "$LEDGER"
  ) 9>>"$LEDGER"
}

# --- run dirs -------------------------------------------------------------------------
banked() {                                            # banked <run dir>: epochs with artifacts_v2.json
  local f n=0
  for f in "$1"/epoch_*/artifacts_v2.json; do [ -e "$f" ] && n=$((n + 1)); done
  echo "$n"
}
amarel_dir() { echo "${1/#$EXP_DIR/$AMAREL_EXP_DIR}"; }   # iLab run dir -> its Amarel scratch twin
under_root() { [[ "$1" == "$DET_V2_ROOT"/* && "$1" != *[[:space:]]* ]]; }
run_dir_of() {                                        # run_dir_of <name> <code dir>: the config's output_dir
  PYTHONPATH="$2" PYTHONNOUSERSITE=1 "$PYTHON" "$DET_V2_SCRIPTS/run_dir.py" "$1" --code "$2"
}

# --- liveness -------------------------------------------------------------------------
# is_live <ledger row>: 0 when the row's job or process still exists, 1 when it does not,
# 2 when the host could not be asked. Any account can see any job in squeue and any pid in
# ps, so this covers the other account's runs too. A job squeue still lists in a final
# state (CANCELLED, TIMEOUT, ...) is not live, nor is a job a later cancel row names, an Amarel
# job an outage row names, or one its final pull marked: none needs its cluster asked (it may be down).
is_live() {
  local row=$1 where job hostgpu probe state
  where=$(lf "$row" $L_WHERE); job=$(lf "$row" $L_JOB); hostgpu=$(lf "$row" $L_HOSTGPU)
  [ -n "$job" ] && [ "$job" != - ] || return 1
  ! cancelled_in_ledger "$row" || return 1
  ! outage_in_ledger "$row" || return 1
  ! amarel_final_ok "$row" || return 1
  case "$where" in
    ilab|ilabcpu) probe=$(rsh "$ILAB_SUBMIT_HOST" "squeue -h -j $job -o %T 2>/dev/null; echo DET_V2_OK" 2>/dev/null) || true ;;
    amarel)       probe=$(rsh "$AMAREL_HOST" "squeue -h -j $job -o %T 2>/dev/null; echo DET_V2_OK" 2>/dev/null) || true ;;
    direct)       probe=$(on_host "${hostgpu%%:*}" "ps -p $job -o pid= 2>/dev/null; echo DET_V2_OK" 2>/dev/null) || true ;;
    *) return 1 ;;
  esac
  grep -q DET_V2_OK <<< "$probe" || return 2
  state=$(grep -v DET_V2_OK <<< "$probe" | head -1 | tr -d ' ')
  case "$state" in
    ""|COMPLETED|CANCELLED*|FAILED|TIMEOUT|PREEMPTED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE) return 1 ;;
    *) return 0 ;;
  esac
}

# cancelled_in_ledger <ledger row>: the ledger holds a cancel row for the same name, world and
# Slurm job id. rebalance.cancel_row writes one only after sacct shows the job CANCELLED (move and
# the outage cancel both check), so the job cannot run again. Direct pids are always probed.
cancelled_in_ledger() {
  local where job
  where=$(lf "$1" $L_WHERE); job=$(lf "$1" $L_JOB)
  case "$where" in ilab|ilabcpu|amarel) ;; *) return 1 ;; esac
  [ -f "$LEDGER" ] || return 1
  awk -F'\t' -v n="$(lf "$1" 1)" -v w="$where" -v j="$job" \
    'NR>1 && $1==n && $3==w && $4==j && $10=="cancel" {f=1} END {exit !f}' "$LEDGER"
}

# outage_in_ledger <ledger row>: the ledger holds an 'outage' row for the same name and Amarel job id,
# stamped at or after AMAREL_OUTAGE_START and before AMAREL_OUTAGE_END. amarel_outage_rows.sh writes
# one only inside that window, for a job the last snapshots showed RUNNING or ended (TIMEOUT,
# PREEMPTED, NODE_FAIL); no Amarel job runs then, so the job is not live. A row stamped outside
# the window is ignored.
outage_in_ledger() {
  [ "$(lf "$1" $L_WHERE)" = amarel ] && [ -f "$LEDGER" ] || return 1
  awk -F'\t' -v n="$(lf "$1" 1)" -v j="$(lf "$1" $L_JOB)" -v s="$AMAREL_OUTAGE_START" -v e="$AMAREL_OUTAGE_END" \
    'NR>1 && $1==n && $3=="amarel" && $4==j && $10=="outage" && $6>=s && $6<e {f=1} END {exit !f}' "$LEDGER"
}

# amarel_final_ok <ledger row>: the row is an Amarel placement whose run dir holds a .amarel_final
# marker for the row's own job. final_pull_amarel.sh writes it only once squeue no longer lists
# the job and the pull returned 0, so the job cannot run again and the shared dir holds every
# banked epoch: neither a liveness check nor a pull needs Amarel (it may be down), for any account.
amarel_final_ok() {
  local rd job
  [ "$(lf "$1" $L_WHERE)" = amarel ] || return 1
  rd=$(lf "$1" $L_RUNDIR); job=$(lf "$1" $L_JOB)
  [ -n "$job" ] && [ "$job" != - ] && [ -f "$rd/.amarel_final" ] \
    && [ "$(awk 'NR==1 {print $1}' "$rd/.amarel_final")" = "$job" ]
}

# refuse_if_live <name> <ledger row>: die when the row's job runs, or when its host cannot be asked.
refuse_if_live() {
  local rc=0
  is_live "$2" || rc=$?
  case $rc in
    0) die "$1 still runs: $(lf "$2" $L_WHERE) job $(lf "$2" $L_JOB) by $(lf "$2" $L_USER)" ;;
    2) die "$1: could not reach $(lf "$2" $L_WHERE) to check job $(lf "$2" $L_JOB); not starting a second copy" ;;
  esac
}

# --- job submission -------------------------------------------------------------------
# job_exports <code> <commit> <name> <python> <threads>: the --export list every job gets.
# job.sh turns these into the runtime environment (umask, caches, threads, TMPDIR).
job_exports() {
  echo "ALL,DET_V2_CODE=$1,ADAPTIVE_ROA_COMMIT=$2,DET_V2_NAME=$3,DET_V2_PYTHON=$4,OMP_NUM_THREADS=$5,DET_V2_NET_ID=$NET_ID,DET_V2_TMP_ROOT=$TMP_ROOT"
}

# start_direct <name> <host> <gpu> <code> <commit> <log> <script> <args...>: write the
# command file and start it in a detached tmux session; prints the pane pid.
start_direct() {
  local name=$1 host=$2 gpu=$3 code=$4 commit=$5 log=$6; shift 6
  local sess="gpu_$SESSION_TAG$name" cmdf ts troot
  troot=$(tmp_root_for "$host") || die "not starting $name on $host: fix DIRECT_TMP_ROOT in $DET_V2_USER_ENV"
  ts=$(date +%Y%m%d_%H%M%S)
  mkdir -p "$LOG_DIR/cmd"
  cmdf="$LOG_DIR/cmd/${ts}_${name}.sh"
  if on_host "$host" "tmux has-session -t '=$sess'" >/dev/null 2>&1; then
    die "tmux session $sess already exists on $host"
  fi
  {
    echo '#!/bin/bash'
    echo "# det_v2 $name on $host gpu$gpu, written $(stamp) by $NET_ID"
    echo 'umask 002'
    echo "export DET_V2_CODE=$code ADAPTIVE_ROA_COMMIT=$commit DET_V2_NAME=$name DET_V2_PYTHON=$PYTHON"
    echo "export DET_V2_NET_ID=$NET_ID DET_V2_TMP_ROOT=$troot OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$gpu"
    echo "exec $(taskset_for "$host") bash $code/scripts/det_v2/job.sh $* >> $log 2>&1"
  } > "$cmdf"
  on_host "$host" "tmux new-session -d -s '$sess' 'bash $cmdf'" >/dev/null 2>&1 || die "tmux launch failed for $name on $host"
  sleep "$DET_V2_TMUX_WAIT"
  local pid
  pid=$(on_host "$host" "tmux list-panes -s -t '=$sess' -F '#{pane_pid}'" 2>/dev/null | grep -E '^[0-9]+$' | head -1)
  [ -n "$pid" ] || die "no tmux pane for $name on $host (did it exit at once? see $log)"
  if [ -n "$DIRECT_JOBS_TSV" ]; then
    printf '%s\t%s\tgpu%s\t%s\t%s\t%s\t%s\n' "$host" "$pid" "$gpu" "$name" "$log" "$ts" "$*" >> "$DIRECT_JOBS_TSV"
  fi
  echo "$pid"
}

# amarel_snapshot <full commit>: build the commit's snapshot on Amarel from the Amarel
# clone (fetching it from origin) with the Amarel paths in its .env; prints the path.
amarel_snapshot() {
  need_amarel
  rsh "$AMAREL_HOST" "cd $AMAREL_CLONE && git fetch -q origin && git show $1:scripts/det_v2/snapshot.sh | DET_V2_LOADED=1 DET_V2_REPO=$AMAREL_CLONE SNAPSHOT_DIR=$AMAREL_SNAPSHOT_DIR NET_ID=$NET_ID DATA_DIR=$AMAREL_DATA_DIR SHARED_DATA_BASE=$AMAREL_DATA_DIR EXP_DIR=$AMAREL_EXP_DIR USER_BASE=$USER_BASE bash -s $1" | tail -1
}

on_origin() {                                         # on_origin <full commit>
  (cd "$DET_V2_REPO" && { git branch -r --contains "$1" 2>/dev/null | grep -q . || git ls-remote origin 2>/dev/null | grep -q "^$1"; })
}

# amarel_constraint <name>: the node feature a run needs on Amarel's untyped gpu partition.
# Placement rule: q3d and q2d FM runs need L40S/A100/Ada cards, which are the
# adalovelace nodes there (ampere mixes A100s with slower sm_86 cards). DET_V2_CONSTRAINT
# overrides it; empty means any node.
amarel_constraint() {
  if [ -n "${DET_V2_CONSTRAINT+x}" ]; then echo "$DET_V2_CONSTRAINT"; return; fi
  case "$1" in [mx][0-9]*_q3d*_fm_*|[mx][0-9]*_q2d*_fm_*) echo adalovelace ;; *) echo "" ;; esac
}

# amarel_submit <name> <code> <commit> <mem> <wall> <script> <args...>: prints the job id.
amarel_submit() {
  local name=$1 code=$2 commit=$3 mem=$4 wall=$5; shift 5
  local feat; feat=$(amarel_constraint "$name")
  rsh "$AMAREL_HOST" "umask 002 && mkdir -p $AMAREL_LOG_DIR && cd $code && sbatch --parsable -J $name --account=$AMAREL_ACCOUNT --partition=$AMAREL_PARTITION --gres=gpu:1 ${feat:+--constraint=$feat }--cpus-per-task=$AMAREL_CPUS --exclude=$AMAREL_EXCLUDE --mem=$mem --time=$wall --output=$AMAREL_LOG_DIR/%x_%j.out --error=$AMAREL_LOG_DIR/%x_%j.err --export=$(job_exports "$code" "$commit" "$name" "$AMAREL_PYTHON" "$AMAREL_CPUS") scripts/det_v2/job.sh $* </dev/null" 2>/dev/null | grep -E '^[0-9]+$' | tail -1
}

# ilab_submit <name> <code> <commit> <mem> <wall> <gres|cpu> <script> <args...>: prints the job id.
ilab_submit() {
  local name=$1 code=$2 commit=$3 mem=$4 wall=$5 gres=$6; shift 6
  local res threads=1
  if [ "$gres" = cpu ]; then res=""; threads=$ILAB_CPU_THREADS; else res="--gres=gpu:$gres:1"; fi
  rsh "$ILAB_SUBMIT_HOST" "umask 002 && mkdir -p $LOG_DIR && cd $code && sbatch --parsable -J $name --partition=$ILAB_PARTITION --nodes=1 $res --mem=$mem --time=$wall --exclude=${DET_V2_EXCLUDE:-$ILAB_EXCLUDE} --output=$LOG_DIR/%x_%j.out --error=$LOG_DIR/%x_%j.err --export=$(job_exports "$code" "$commit" "$name" "$PYTHON" "$threads") scripts/det_v2/job.sh $* </dev/null" 2>/dev/null | grep -E '^[0-9]+$' | tail -1
}

# amarel_pull <name> <run dir> <account that ran it>: final pull of an Amarel-run cell into
# its shared run dir before it runs anywhere else. Only that account can read the scratch dir.
amarel_pull() {
  [ "$3" = "$NET_ID" ] || die "$1 ran on $3's Amarel scratch; only that account can pull it"
  need_amarel
  local live; live=$(rsh "$AMAREL_HOST" "squeue -u $AMAREL_USER -h -n $1 -o %i" 2>/dev/null)
  [ -z "$live" ] || die "$1 still has Amarel job $live; cancel it first"
  local rc=0
  mkdir -p "$2"
  rsync -rlptDO --chmod=Dg+rwxs,Fg+rw --delete --exclude .amarel_mirror "$AMAREL_HOST:$(amarel_dir "$2")/" "$2/" </dev/null || rc=$?
  [ $rc = 0 ] || die "final pull of $1 from Amarel failed (rsync $rc)"
  rm -f "$2/.amarel_mirror"
}

# --- resume ---------------------------------------------------------------------------
ledger_last_placed() {                                # latest launch/resume row: where the run really is
  [ -f "$LEDGER" ] || return 0
  awk -F'\t' -v n="$1" 'NR>1 && $1==n && ($10=="launch" || $10=="resume")' "$LEDGER" | tail -1
}

# resume_prepare <name>: read the run's ledger rows and refuse what must not resume.
# Sets LAST (latest row), PLACED (latest launch/resume row), PREV and PREV_USER (where and
# by whom it last ran), RUN_DIR and COMMIT.
resume_prepare() {
  LAST=$(ledger_last "$1")
  [ -n "$LAST" ] || die "$1 has no ledger row; launch it with launch.sh"
  [ "$(manifest_get "$1" script)" != scripts/run_partx_faithful.py ] \
    || die "$1 is a Part-X row: rerun launch.sh $1 ilabcpu, it continues from partx_state.pkl"
  if [ "$(lf "$LAST" $L_ACTION)" = claim ] && [ "${DET_V2_RELAUNCH:-0}" != 1 ]; then
    die "$1 has a claim in progress ($(lf "$LAST" $L_USER) at $(cut -f6 <<< "$LAST")); wait, or DET_V2_RELAUNCH=1 if it died"
  fi
  PLACED=$(ledger_last_placed "$1")
  [ -n "$PLACED" ] || die "$1 was never placed; launch it with launch.sh"
  refuse_if_live "$1" "$PLACED"
  PREV=$(lf "$PLACED" $L_WHERE); PREV_USER=$(lf "$PLACED" $L_USER)
  RUN_DIR=$(lf "$PLACED" $L_RUNDIR); COMMIT=$(lf "$PLACED" $L_COMMIT)
  [ "$RUN_DIR" != - ] && [ "$COMMIT" != - ] || die "$1: the ledger row has no run dir or commit"
  under_root "$RUN_DIR" || die "$1: run dir $RUN_DIR is not under $DET_V2_ROOT"
}

# ledger_claim_resume <name> <expected last row> <where> <hostgpu> <commit> <run_dir>:
# append a claim row only if nobody appended for the name since the caller read it.
ledger_claim_resume() {
  (
    ledger_lock || exit 2
    now=$(awk -F'\t' -v n="$1" 'NR>1 && $1==n' "$LEDGER" | tail -1)
    if [ "$now" != "$2" ]; then
      echo "$1 changed in the ledger meanwhile ($(cut -f2-4,10 <<< "$now" | tr '\t' ' ')); check it before resuming" >&2
      exit 3
    fi
    printf '%s\t%s\t%s\t-\t%s\t%s\t-\t%s\t%s\tclaim\n' "$1" "$NET_ID" "$3" "$4" "$(stamp)" "$5" "$6" >> "$LEDGER"
  ) 9>>"$LEDGER"
}

full_commit() {                                       # full_commit <short>: fetches it if missing
  (cd "$DET_V2_REPO" && { git rev-parse -q --verify "$1^{commit}" 2>/dev/null || { git fetch -q origin && git rev-parse "$1^{commit}"; }; })
}

# leave_amarel <name>: a run that last ran on Amarel gets its final pull before it runs
# here, unless final_pull_amarel.sh already took it for that job (amarel_final_ok), or an
# outage row names the job (outage_in_ledger): then the run continues from the shared copy,
# and epochs only Amarel holds get redone (driver 2026-11-09: waiting for Amarel costs about
# 40 h of the row before the deadline). A mirror marker left by a push that never ran is
# dropped: the shared dir is current.
leave_amarel() {
  if [ "$PREV" = amarel ] && amarel_final_ok "$PLACED"; then
    echo "$1: Amarel job $(lf "$PLACED" $L_JOB) already had its final pull ($(cut -d' ' -f2- "$RUN_DIR/.amarel_final")); not asking Amarel" >&2
    rm -f "$RUN_DIR/.amarel_mirror"
  elif [ "$PREV" = amarel ] && outage_in_ledger "$PLACED"; then
    echo "$1: Amarel job $(lf "$PLACED" $L_JOB) has an outage row: resumed from iLab copy; Amarel may hold later epochs" >&2
    rm -f "$RUN_DIR/.amarel_mirror"
  elif [ "$PREV" = amarel ]; then
    amarel_pull "$1" "$RUN_DIR" "$PREV_USER"
  elif [ -f "$RUN_DIR/.amarel_mirror" ]; then
    echo "$1 last ran on $PREV; dropping the stale .amarel_mirror marker" >&2
    rm -f "$RUN_DIR/.amarel_mirror"
  fi
}
