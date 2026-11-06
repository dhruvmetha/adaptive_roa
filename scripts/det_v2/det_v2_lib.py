"""Shared helpers for the det_v2 Python tools (run_dir, preflight, status, rebalance).

Settings come from common.sh through bash, so the Python tools and the shell scripts
read user.env the same way. Hydra composition mirrors scripts/run_adaptive.py:
config_path configs/adaptive_v2, the script's own config_name, and the resolvers it
registers, with their values taken from user.env instead of a checkout's .env.
"""
from __future__ import annotations

import csv
import fcntl
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER_COLUMNS = ["name", "user", "where", "job", "hostgpu", "timestamp", "log", "commit", "run_dir", "action"]
N_EPOCHS = 11            # epoch_000 .. epoch_010


def load_user_env() -> dict[str, str]:
    """The environment common.sh exports (user.env plus campaign defaults)."""
    r = subprocess.run(["bash", "-c", 'source "$1" >/dev/null && env -0', "_", str(HERE / "common.sh")],
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr.decode().strip() or "common.sh failed")
    return dict(kv.split("=", 1) for kv in r.stdout.decode().split("\0") if "=" in kv)


def read_manifest(path: str | Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def manifest_row(env: dict, name: str) -> dict:
    for r in read_manifest(env["MANIFEST"]):
        if r["name"] == name:
            return r
    raise SystemExit(f"no manifest row for {name}")


# --- ledger -----------------------------------------------------------------------------

def read_ledger(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for i, line in enumerate(p.read_text().splitlines()):
        f = line.split("\t")
        if i == 0 and f[0] == "name":
            continue
        if len(f) >= len(LEDGER_COLUMNS):
            rows.append(dict(zip(LEDGER_COLUMNS, f)))
    return rows


def latest_rows(path: str | Path) -> dict[str, dict]:
    """The last ledger row per name: its current placement."""
    out: dict[str, dict] = {}
    for r in read_ledger(path):
        out[r["name"]] = r
    return out


def append_ledger(path: str | Path, row: dict) -> None:
    """Append one row under the same flock the shell scripts take."""
    os.umask(0o002)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if f.tell() == 0 and p.stat().st_size == 0:
                f.write("\t".join(LEDGER_COLUMNS) + "\n")
            f.write("\t".join(str(row.get(c, "-")) for c in LEDGER_COLUMNS) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --- run dirs ---------------------------------------------------------------------------

def epoch_count(run_dir: str | Path, n_epochs: int = N_EPOCHS) -> int:
    """Epochs 000..n-1 whose artifacts_v2.json exists, counted from disk."""
    d = Path(run_dir)
    return sum((d / f"epoch_{e:03d}" / "artifacts_v2.json").exists() for e in range(n_epochs))


def is_finished(run_dir: str | Path, n_epochs: int = N_EPOCHS) -> bool:
    return epoch_count(run_dir, n_epochs) == n_epochs


def find_run_dir(env: dict, name: str) -> Path | None:
    """<DET_V2_ROOT>/<setting>/<name>_s<seed> when exactly one such dir exists."""
    hits = [p for p in Path(env["DET_V2_ROOT"]).glob(f"*/{name}_s*")
            if p.is_dir() and re.fullmatch(re.escape(name) + r"_s\d+", p.name)]
    return hits[0] if len(hits) == 1 else None


# --- Hydra composition ------------------------------------------------------------------

def config_name_for(code: str | Path, script: str) -> str:
    """The config_name the runner's @hydra.main uses (default for run_adaptive.py)."""
    p = Path(code) / script
    m = re.search(r'config_name="([^"]+)"', p.read_text()) if p.exists() else None
    return m.group(1) if m else "default"


def register_resolvers(env: dict) -> None:
    from omegaconf import OmegaConf
    values = {"net_id": env["NET_ID"], "exp_dir": env["EXP_DIR"], "data_dir": env["DATA_DIR"],
              "shared_data_base": env["SHARED_DATA_BASE"]}
    for key, val in values.items():
        if OmegaConf.has_resolver(key):
            OmegaConf.clear_resolver(key)
        OmegaConf.register_new_resolver(key, (lambda v: (lambda default="": v))(val))
    if OmegaConf.has_resolver("env"):
        OmegaConf.clear_resolver("env")
    OmegaConf.register_new_resolver("env", lambda key, default="": os.environ.get(key, env.get(key, default)))


def compose(env: dict, code: str | Path, overrides: list[str], config_name: str = "default"):
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    register_resolvers(env)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(Path(code) / "configs" / "adaptive_v2")):
        return hydra_compose(config_name=config_name, overrides=overrides)


def compose_row(env: dict, row: dict, code: str | Path | None = None):
    code = code or env["DET_V2_REPO"]
    return compose(env, code, row["config"].split(), config_name_for(code, row["script"]))


def run_kind(name: str) -> tuple[str, str]:
    """(setting token, model token) from a det_v2 name: m01_q3dlqr_mlpclf_qa -> (q3dlqr, mlpclf)."""
    m = re.match(r"[a-z]\d+_([a-z0-9]+)_([a-z]+)", name)
    return (m.group(1), m.group(2)) if m else ("", "")
