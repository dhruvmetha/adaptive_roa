"""A throwaway det_v2 world for the tests: no network, no SLURM, no GPU.

The `world` fixture (conftest.py) builds a tiny git repo holding copies of scripts/det_v2 plus stub runners and a
two-run config tree, a bare "origin" and an "Amarel" clone of it, and a bin/ of stubs on
PATH: ssh runs the remote command here, sbatch runs the job script at once with its
--export list, tmux runs the session command at once, rsync drops "host:" prefixes,
squeue / sacct / scancel answer from files. The job's python is a wrapper that dumps
its environment, umask and arguments when asked to run a runner, and runs the real
python otherwise (run_dir.py, preflight.py), so every launch and resume path executes
end to end and leaves a dump to assert on.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "scripts" / "det_v2"
GPU_RUN = "m01_q3dlqr_fm_qa"
SMALL_RUN = "m02_cplqr_mlpclf_qa"
PARTX_RUN = "m03_cplqr_partx"
MANIFEST = (
    "id\tname\trank\towner\tconfig\tscript\tmem\test_hours\n"
    f"M01\t{GPU_RUN}\t1\tSumanth\t+experiment=det_v2/runs/{GPU_RUN}\tscripts/run_adaptive.py\t16G\t8.5\n"
    f"M02\t{SMALL_RUN}\t2\tDhruv\t+experiment=det_v2/runs/{SMALL_RUN}\tscripts/run_adaptive.py\t8G\t0.4\n"
    f"M03\t{PARTX_RUN}\t3\tDhruv\t+experiment=det_v2/runs/{PARTX_RUN}\tscripts/run_partx_faithful.py\t16G\t0.1\n")

STUBS = {
    "ssh": r'''
while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -*) shift;; *) break;; esac; done
host=$1; shift
printf '%s\t%s\n' "$host" "$*" >> "$CALLS/ssh"
exec bash -c "$*"
''',
    "sbatch": r'''
exports=""; out=""; name=""
while [ $# -gt 0 ]; do
  case "$1" in
    --export=*) exports=${1#--export=} ;;
    --output=*) out=${1#--output=} ;;
    -J) name=$2; shift ;;
    -*) ;;
    *) break ;;
  esac
  shift
done
script=$1; shift
jid=$(( $(cat "$CALLS/jid" 2>/dev/null || echo 1000) + 1 )); echo $jid > "$CALLS/jid"
printf '%s\t%s\t%s\t%s\n' "$name" "$jid" "$PWD" "$exports" >> "$CALLS/sbatch"
out=${out//%x/$name}; out=${out//%j/$jid}
vars=(); IFS=, read -ra kvs <<< "$exports"
for kv in "${kvs[@]}"; do [ "$kv" = ALL ] || vars+=("$kv"); done
SLURM_JOB_ID=$jid env "${vars[@]}" bash "$script" "$@" > "${out:-/dev/null}" 2>&1 || true
echo $jid
''',
    "squeue": r'''
printf '%s\n' "$*" >> "$CALLS/squeue"
[ -f "$CALLS/squeue_out" ] && cat "$CALLS/squeue_out"
exit 0
''',
    "sacct": r'''
printf '%s\n' "$*" >> "$CALLS/sacct"
[ -f "$CALLS/sacct_out" ] && cat "$CALLS/sacct_out"
exit 0
''',
    "scancel": r'''printf '%s\n' "$*" >> "$CALLS/scancel"''',
    "tmux": r'''
printf '%s\n' "$*" >> "$CALLS/tmux"
case "$1" in
  has-session) exit 1 ;;
  new-session) shift; while [ $# -gt 1 ]; do case "$1" in -d) shift ;; -s) shift 2 ;; *) break ;; esac; done
               bash -c "$1"; exit 0 ;;
  list-panes) echo 999999999 ;;
esac
''',
    "rsync": r'''
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -e) shift ;;
    *:*) args+=("${1#*:}") ;;
    *) args+=("$1") ;;
  esac
  shift
done
printf '%s\n' "${args[*]}" >> "$CALLS/rsync"
exec /usr/bin/rsync "${args[@]}"
''',
}

FAKE_PYTHON = r'''#!/bin/bash
case "${1:-}" in
  scripts/run_adaptive.py|scripts/resume_adaptive.py|scripts/run_partx_faithful.py)
    f="$DUMP_DIR/$(date +%s%N).dump"
    { echo "ARGS=$*"; echo "UMASK=$(umask)"; echo "PWD=$PWD"; env; } > "$f"
    exit 0 ;;
esac
exec REAL_PYTHON "$@"
'''

DEFAULT_YAML = """defaults:
  - _self_
exp_dir: ${exp_dir:}
data_dir: ${data_dir:}
output_dir: ${exp_dir}/det_v2/unset
seed: 0
data_source:
  split_dir: ${data_dir}/toy/splits
  trajectories_dir: ${data_source.split_dir}/train.npz
  test_set_file: ${data_source.split_dir}/test_set.txt
"""
RUN_YAML = """# @package _global_
seed: 7
output_dir: ${{exp_dir}}/det_v2/{setting}/{name}_s${{seed}}
"""
RUNNER = '@hydra.main(config_path="../configs/adaptive_v2", config_name="{cfg}", version_base=None)\n'


def _exe(p: Path, body: str) -> None:
    p.write_text(body if body.startswith("#!") else "#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def git(cwd, *args) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


class World:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.calls = tmp / "calls"
        self.dumps = tmp / "dumps"
        self.bin = tmp / "bin"
        self.exp = tmp / "exp"
        self.data = tmp / "data"
        self.amarel = tmp / "amarel"
        for d in (self.calls, self.dumps, self.bin, self.exp / "det_v2", self.amarel / "exp", self.amarel / "logs"):
            d.mkdir(parents=True)
        self._repo()
        for name, body in STUBS.items():
            _exe(self.bin / name, body)
        _exe(self.bin / "fake_python", FAKE_PYTHON.replace("REAL_PYTHON", sys.executable))
        for root in (self.data, self.amarel / "data"):
            (root / "toy" / "splits").mkdir(parents=True)
            for f in ("train.npz", "test_set.txt"):
                (root / "toy" / "splits" / f).write_text("x\n")
        self.user_env = tmp / "user.env"
        self.user_env.write_text("\n".join([
            "NET_ID=tester", f"PYTHON={self.bin / 'fake_python'}", f"SNAPSHOT_DIR={tmp / 'snap'}",
            f"DATA_DIR={self.data}", f"EXP_DIR={self.exp}", "ILAB_SUBMIT_HOST=ilabsubmit",
            "DIRECT_HOST_SUFFIX=", f"GPU_FREE={self.bin / 'gpu-free'}",
            "AMAREL_HOST=amarelhost", f"AMAREL_CLONE={self.amarel / 'clone'}",
            f"AMAREL_PYTHON={self.bin / 'fake_python'}", f"AMAREL_SNAPSHOT_DIR={self.amarel / 'snap'}",
            f"AMAREL_EXP_DIR={self.amarel / 'exp'}", f"AMAREL_DATA_DIR={self.amarel / 'data'}",
            f"AMAREL_LOG_DIR={self.amarel / 'logs'}", f"TMP_ROOT={tmp / 'tmproot'}", ""]))
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "CALLS": str(self.calls),
                    "DUMP_DIR": str(self.dumps), "DET_V2_USER_ENV": str(self.user_env),
                    "DET_V2_TMUX_WAIT": "0", "PYTHONNOUSERSITE": "1"}
        self.env.pop("DET_V2_LOADED", None)
        self.env.pop("SSH_AUTH_SOCK", None)

    def _repo(self):
        r = self.repo
        (r / "scripts" / "det_v2").mkdir(parents=True)
        for f in SRC.iterdir():
            if f.is_file() and f.name != "user.env" and f.suffix in (".sh", ".py", ".example"):
                shutil.copy2(f, r / "scripts" / "det_v2" / f.name)
        (r / "scripts" / "det_v2" / "manifest.tsv").write_text(MANIFEST)
        (r / "scripts" / "run_adaptive.py").write_text(RUNNER.format(cfg="default"))
        (r / "scripts" / "resume_adaptive.py").write_text("# resume stub\n")
        (r / "scripts" / "run_partx_faithful.py").write_text(RUNNER.format(cfg="partx_faithful"))
        cfg = r / "configs" / "adaptive_v2"
        (cfg / "experiment" / "det_v2" / "runs").mkdir(parents=True)
        (cfg / "default.yaml").write_text(DEFAULT_YAML)
        (cfg / "partx_faithful.yaml").write_text("defaults:\n  - default\n  - _self_\npartx_faithful: {}\n")
        for name, setting in ((GPU_RUN, "q3d-lqr"), (SMALL_RUN, "cp-lqr"), (PARTX_RUN, "cp-lqr")):
            (cfg / "experiment" / "det_v2" / "runs" / f"{name}.yaml").write_text(RUN_YAML.format(setting=setting, name=name))
        (r / ".gitignore").write_text("scripts/det_v2/user.env\n__pycache__/\n")
        git(r, "init", "-q", "-b", "main")
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", "init")
        origin = self.tmp / "origin.git"
        git(self.tmp, "clone", "-q", "--bare", str(r), str(origin))
        git(r, "remote", "add", "origin", str(origin))
        git(r, "fetch", "-q", "origin")
        git(self.tmp, "clone", "-q", str(origin), str(self.amarel / "clone"))

    def run(self, script: str, *args: str, check: bool = True, extra: dict | None = None) -> subprocess.CompletedProcess:
        r = subprocess.run(["bash", str(self.repo / "scripts" / "det_v2" / script), *args],
                           env={**self.env, **(extra or {})}, capture_output=True, text=True, timeout=300)
        if check and r.returncode != 0:
            raise AssertionError(f"{script} {args} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
        return r

    def head(self) -> str:
        return git(self.repo, "rev-parse", "--short=10", "HEAD")

    def run_dir(self, name: str) -> Path:
        setting = "q3d-lqr" if name == GPU_RUN else "cp-lqr"
        return self.exp / "det_v2" / setting / f"{name}_s7"

    def ledger(self) -> list[list[str]]:
        p = self.exp / "det_v2" / "launched.tsv"
        return [l.split("\t") for l in p.read_text().splitlines()] if p.exists() else []

    def dumps_list(self) -> list[dict]:
        out = []
        for f in sorted(self.dumps.glob("*.dump")):
            d = {}
            for line in f.read_text().splitlines():
                k, _, v = line.partition("=")
                d.setdefault(k, v)
            out.append(d)
        return out

    def bank(self, run_dir: Path, epochs: int) -> None:
        for e in range(epochs):
            (run_dir / f"epoch_{e:03d}").mkdir(parents=True, exist_ok=True)
            (run_dir / f"epoch_{e:03d}" / "artifacts_v2.json").write_text("{}")

    def ledger_row(self, name, where, job, hostgpu, run_dir, action="launch", user="tester", commit=None):
        p = self.exp / "det_v2" / "launched.tsv"
        if not p.exists():
            p.write_text("name\tuser\twhere\tjob\thostgpu\ttimestamp\tlog\tcommit\trun_dir\taction\n")
        with open(p, "a") as f:
            f.write("\t".join([name, user, where, str(job), hostgpu, "2026-11-06T00:00:00", "-",
                               commit or self.head(), str(run_dir), action]) + "\n")
