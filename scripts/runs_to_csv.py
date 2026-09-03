"""Render runs.jsonl as a CSV alongside it, plus a live epoch-depth column.

`runs.jsonl` is the append-only source of truth -- one record per launch, never
edited in place (a resume is a NEW record, not a status update on the old one).
This script is a projection of it, so the CSV is regenerated, never hand-edited:

    python scripts/runs_to_csv.py

Two columns are NOT in the jsonl and are computed fresh from disk each time:

  epochs_on_disk  artifacts_v2.json count under output_dir. This is the only
                  honest measure of depth -- epoch_* directories are created
                  before an epoch finishes, so counting them overstates progress.
                  Blank when the count could not be taken; read `dir_state` to
                  learn why.
  dir_state       why a count is or isn't available, because "no number" has
                  three very different meanings and conflating them is how a
                  superseded run gets mistaken for a live one:
                    local   -- counted here, the number is real
                    remote  -- output_dir is on another filesystem (Amarel
                               /scratch), so it cannot be counted from this host.
                               NOT evidence of zero progress.
                    missing -- path is local and absent: the run never wrote
                               anything, e.g. cancelled before it started.

Sorted newest-launch-first so the active campaign is at the top.
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# One log per campaign, NOT one global log. quad2D and quad3D are separate
# systems with different state spaces, different pools and (since 2026-08-22)
# different budgets -- 24 epochs vs 18 -- so a shared log invites comparing rows
# that are not comparable and makes "how deep is this level" a filtering problem
# rather than a wc -l. Each jsonl gets a sibling CSV of the same stem.
LOGS = [
    ROOT / "docs/experiments/ensemble_epistemic/runs.jsonl",
    Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor2d/runs_quad2d.jsonl"),
    Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor3d/runs_quad3d.jsonl"),
]

FIELDS = [
    "run_id", "launched_at", "system", "level", "arm", "predictor", "score_mode",
    "d2_ratio", "seed", "n_members", "k_acq", "cluster", "job_id", "status",
    "epochs_on_disk", "dir_state", "code_hash", "output_dir", "notes",
]

# Roots that do not exist on the host running this script. An output_dir under one
# of these is unmeasurable from here, which is NOT the same as empty.
REMOTE_ROOTS = ("/scratch/",)


def depth(output_dir):
    """(epochs, dir_state) -- count artifacts_v2.json, never epoch_* dirs."""
    if not output_dir:
        return "", "missing"
    if output_dir.startswith(REMOTE_ROOTS):
        return "", "remote"
    p = Path(output_dir)
    if not p.is_dir():
        return "", "missing"
    return sum(1 for _ in p.glob("epoch_*/artifacts_v2.json")), "local"


def render(src_path):
    """Project one jsonl to its sibling CSV. Returns (n_records, dir_state counts)."""
    dst_path = src_path.with_suffix(".csv")
    records = []
    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n, ds = depth(r.get("output_dir"))
            r["epochs_on_disk"], r["dir_state"] = n, ds
            records.append(r)

    records.sort(key=lambda r: r.get("launched_at", ""), reverse=True)

    with open(dst_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return dst_path, records


for src_path in LOGS:
    if not src_path.exists():
        print(f"skip {src_path.relative_to(ROOT)} (absent)")
        continue
    dst_path, records = render(src_path)
    counts = Counter(r["dir_state"] for r in records)
    print(f"wrote {dst_path.relative_to(ROOT)}")
    print(f"  {len(records)} runs | " + " | ".join(f"{k}={v}" for k, v in sorted(counts.items())))
