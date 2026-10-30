"""What each acquisition round bought, from the pick log (tracker H09; GRAPE's sampling table).

Per round: how many picks came from the uniform share and from the acquisition
score, the success share of each, where along their trajectories the acquired
picks sit (row / (length - 1); 0 is a start), and how many picks landed on a
trajectory already in training (each still cost a rollout, D22a).

Rounds are cut at the budgets the epochs' artifacts record, since a direct round
can fall short of samples_per_epoch (D25); without artifacts, every round is
samples_per_epoch picks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def _rounds(run: Path, n0: int, step: int, n_log: int) -> list[tuple[int, int]]:
    budgets = [int(json.loads(a.read_text())["train_trajectories"])
               for a in sorted(run.glob("epoch_*/artifacts_v2.json"))]
    if not budgets or budgets[0] != n0:
        return [(s, min(s + step, n_log)) for s in range(n0, n_log, step)]
    cuts = [b for b in budgets if b <= n_log]
    if cuts[-1] < n_log:
        cuts.append(n_log)          # picks made after the last stored epoch
    return list(zip(cuts[:-1], cuts[1:]))


def acquisition_profile(run_dir) -> list[dict]:
    run = Path(run_dir)
    cfg = OmegaConf.load(run / "resolved_config.yaml")
    state = json.loads((run / "dataset_builder_state.json").read_text())
    if state.get("mode") != "suffix":
        raise ValueError(f"{run}: pick log needs candidate_mode=suffix")
    ds = cfg.data_source
    with np.load(ds.trajectories_dir) as z:
        offsets = z["offsets"].astype(np.int64)
    ids = np.loadtxt(ds.shuffled_indices_file, dtype=np.int64)
    labels = np.loadtxt(ds.shuffled_labels_file, dtype=np.int64)
    lengths = np.diff(offsets)[ids]
    n0, step = int(cfg.initial_train_size), int(cfg.samples_per_epoch)
    log = state["pick_log"]
    seen = {t for t, _r, src in log[:n0]}
    rows = []
    for e, (start, end) in enumerate(_rounds(run, n0, step, len(log))):
        batch = log[start:end]
        uni = [(t, r) for t, r, src in batch if src == "uniform"]
        acq = [(t, r) for t, r, src in batch if src == "acquisition"]
        frac = [r / max(lengths[t] - 1, 1) for t, r in acq]
        rows.append({
            "run": run.name, "round": e, "budget_after": end,
            "n_uniform": len(uni), "n_acquired": len(acq),
            "success_share_uniform": float(np.mean([labels[t] == 1 for t, _ in uni])) if uni else float("nan"),
            "success_share_acquired": float(np.mean([labels[t] == 1 for t, _ in acq])) if acq else float("nan"),
            "acquired_at_start": float(np.mean([r == 0 for _, r in acq])) if acq else float("nan"),
            "acquired_row_fraction_mean": float(np.mean(frac)) if frac else float("nan"),
            "picks_on_seen_trajectories": int(sum(t in seen for t, _r, _s in batch)),
        })
        seen |= {t for t, _r, _s in batch}
    return rows
