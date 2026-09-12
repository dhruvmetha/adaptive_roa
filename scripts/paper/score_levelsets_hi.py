#!/usr/bin/env python
"""Re-project already-evaluated adaptive runs onto the HIGH level grid.

betas = 0.50, 0.55, ..., 0.95 (ten levels), versus the 0.05..0.95 grid the
monitoring CSVs use. No model inference: each epoch's full_roa_per_point.npz
(start_states, p_success, p_invalid) is joined to the continuous ground truth
eval_success_prob.npz exactly as score_epoch_full in scripts/stoch_prob_metrics.py
does, and the level-set table, its M/K-aware oracle and the area summary are
reused from that module unchanged.

Outputs, one pair per system group, under docs/stochastic/paper/levelsets/:
  <group>_levelsets_b50.csv        long: one row per (arm, epoch, beta)
  <group>_levelset_areas_b50.csv   wide: one row per (arm, epoch)

Epoch selection (--epochs):
  common   per campaign, the final epoch every present arm has reached
           (min over arms of max present epoch); scored for every arm.
  all      every present epoch of every present arm.
  3,5,7    an explicit list.
An epoch is "present" only when BOTH artifacts_v2.json and
full_roa_per_point.npz exist in it.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


spm = _load("stoch_prob_metrics", SCRIPTS / "stoch_prob_metrics.py")
ssi = _load("score_stoch_incremental", SCRIPTS / "score_stoch_incremental.py")

BETAS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
OUT_DIR = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic/paper_draft/levelsets")

GROUPS = {
    "pendulum_lqr": ["pend_low", "pend_med", "pend_high"],
    "cartpole_ppo": ["cprl_base", "cprl_low", "cprl_med", "cprl_high"],
    # `loud` added 2026-09-11 (main-comp session's runs; one uniform seed only).
    # `q2d_csbase` (no noise, deterministic) added 2026-09-11: its live FM runs
    # are the `_fast` dirs; score them with --arms ... --arm-alias <arm>_fast=<arm>.
    "quad2d_rl": ["q2d_csbase", "q2d_cs", "q2d_csloud"],
    "quad3d_ppo": ["q3dppo_cs000", "q3dppo_cs012", "q3dppo_cs020", "q3dppo_cs040"],
    # Same system, same ground truth, 800k trainable pool (user decision 2026-09-10:
    # quad3d is read from the 800k campaign). Separate group so the two pools
    # never share a file.
    "quad3d_ppo800k": ["q3d800k_cs000", "q3d800k_cs012", "q3d800k_cs020", "q3d800k_cs040",
                       # ambient-noise variants (main-comp session, 2026-09-11; one seed, early)
                       "q3damb_f012a03", "q3damb_f020a04", "q3damb_f040a04"],
}
CAMPAIGN_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}

# BALD arms are the top-N (greedy) runs (decision 2026-09-10); greedy_diverse is out of the paper
PAPER_ARMS = ["dir00_s42", "epi_bald_greedy", "epi_var_greedy",   # single-seed control (2026-09-11)
              "clf_epi_bald_greedy", "bnn_mfvi_a1_greedy", "partx_fix", "clf_dir00", "bnn_a1_dir00"]

LEAD = ["predictor", "level", "arm", "epoch", "train_trajectories"]
TABLE_KEYS = ["tp", "tn", "fp", "fn", "n_pos_true", "n_neg_true"] + list(spm.LEVEL_STATS)
LONG_FIELDS = LEAD + ["beta"] + TABLE_KEYS + [f"{k}_oracle" for k in TABLE_KEYS]
WIDE_FIELDS = LEAD + ["K", "M", "n_levels", "n_levels_defined"]
for _s in spm.LEVEL_STATS:
    WIDE_FIELDS += [f"auc_{_s}", f"auc_{_s}_oracle"]
WIDE_FIELDS.append("worst_overclaim")


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def run_dir_of(campaign: str, arm: str) -> Path:
    c = ssi.CAMPAIGNS[campaign]
    return (c.get("exp") or ssi.EXP) / f"{c['prefix']}_{arm}"


def present_epochs(run_dir: Path) -> set[int]:
    out = set()
    for d in run_dir.glob("epoch_*"):
        if (d / "artifacts_v2.json").exists() and (d / "full_roa_per_point.npz").exists():
            try:
                out.add(int(d.name.split("_", 1)[1]))
            except ValueError:
                pass
    return out


def plan_jobs(campaign: str, mode: str, arms: list[str] | None = None) -> tuple[list[tuple], dict]:
    """Return (jobs, info). info records the chosen epoch(s) and arm presence.
    `arms` overrides PAPER_ARMS (e.g. a subset to backfill); with mode "common"
    the common epoch is taken over THOSE arms."""
    c = ssi.CAMPAIGNS[campaign]
    gt_root = c["root"] / c["level"]
    present: dict[str, set[int]] = {}
    for arm in (arms or PAPER_ARMS):
        rd = run_dir_of(campaign, arm)
        if not rd.is_dir():
            continue
        eps = present_epochs(rd)
        if eps:
            present[arm] = eps
    info = {"campaign": campaign, "arms": sorted(present), "missing": [], "epochs": None}
    if not present:
        return [], info
    if mode == "common":
        common = min(max(e) for e in present.values())
        wanted = {arm: {common} for arm in present}
        info["epochs"] = [common]
    elif mode == "all":
        wanted = {arm: set(eps) for arm, eps in present.items()}
        info["epochs"] = sorted(set().union(*present.values()))
    else:
        req = {int(x) for x in mode.split(",") if x.strip()}
        wanted = {arm: req for arm in present}
        info["epochs"] = sorted(req)
    jobs = []
    for arm, eps in wanted.items():
        for ep in sorted(eps):
            if ep not in present[arm]:
                info["missing"].append((arm, ep))
                continue
            jobs.append((campaign, arm, ep, str(run_dir_of(campaign, arm) / f"epoch_{ep:03d}"),
                         str(gt_root), c["key"]))
    return jobs, info


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
_GT_CACHE: dict[str, tuple] = {}


def _ground_truth(gt_root: str) -> tuple:
    gt = _GT_CACHE.get(gt_root)
    if gt is None:
        gt = spm.load_ground_truth(Path(gt_root))
        _GT_CACHE[gt_root] = gt
    return gt


def score_job(job: tuple) -> tuple[tuple, list[dict], dict, float]:
    campaign, arm, ep, epoch_dir, gt_root, key = job
    t0 = time.time()
    epoch_dir = Path(epoch_dir)
    gt_starts, gt_p, gt_s, gt_t = _ground_truth(gt_root)
    with np.load(epoch_dir / "full_roa_per_point.npz") as z:
        states = z["start_states"]
        p_hat = z["p_success"].astype(np.float64)
    idx = spm.match_to_truth(states, gt_starts)
    meta = {}
    try:
        meta = json.loads((epoch_dir / "artifacts_v2.json").read_text())
    except ValueError:
        meta = {}
    try:
        k = float(meta["eval_metrics"]["num_mc_samples"])
    except (KeyError, ValueError, TypeError):
        k = None
    p, t = gt_p[idx], gt_t[idx]
    m = float(np.mean(t))
    rows = spm.level_set_table(p_hat, p, betas=BETAS)
    oracle = spm.level_set_oracle(p, k, m, betas=BETAS, n_draws=8, seed=0)
    summary = spm.level_set_summary(rows, oracle)
    tt = meta.get("train_trajectories")
    lead = {"predictor": ssi.predictor_of(arm), "level": key, "arm": arm, "epoch": int(ep),
            "train_trajectories": int(tt) if tt is not None else ""}
    long_rows = []
    for r in spm.level_rows(rows, oracle):
        row = dict(lead)
        row["beta"] = f"{r['beta']:.2f}"
        row.update({kk: v for kk, v in r.items() if kk != "beta"})
        long_rows.append(row)
    wide = dict(lead)
    wide["K"] = "" if k is None else k
    wide["M"] = m
    wide.update(summary)
    return (campaign, arm, ep), long_rows, wide, time.time() - t0


# --------------------------------------------------------------------------
# csv io
# --------------------------------------------------------------------------
def _norm_key(r: dict, with_beta: bool) -> tuple:
    k = (str(r["level"]), str(r["arm"]), str(int(float(r["epoch"]))))
    if with_beta:
        k += (f"{float(r['beta']):.2f}",)
    return k


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def merge_rows(old: list[dict], new: list[dict], with_beta: bool) -> list[dict]:
    merged = {_norm_key(r, with_beta): r for r in old}
    for r in new:
        merged[_norm_key(r, with_beta)] = r
    return sorted(merged.values(),
                  key=lambda r: (str(r["level"]), str(r["predictor"]), str(r["arm"]),
                                 int(float(r["epoch"])),
                                 float(r["beta"]) if with_beta else 0.0))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Atomic replace so a concurrent reader sees the whole old or whole new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, path.parent.stat().st_mode | 0o070)
    except OSError:
        pass
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o660
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaigns", nargs="*", default=list(CAMPAIGN_GROUP),
                    help="subset of the in-scope campaigns (default: all)")
    ap.add_argument("--epochs", default="common", help="common | all | comma list")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"default {OUT_DIR}; must be given explicitly together with --arms")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="arm names to score instead of PAPER_ARMS; pair with --out-dir so a "
                         "different arm set never rewrites the paper files")
    ap.add_argument("--arm-alias", nargs="*", default=[], metavar="FROM=TO",
                    help="rename arm FROM (a run-dir name) to TO in the OUTPUT rows, long and "
                         "wide, so the level-set stores carry canonical names (e.g. the "
                         "q2d_csbase `_fast` runs: dir00_s42_fast=dir00_s42)")
    a = ap.parse_args()
    if a.arms and a.out_dir is None:
        sys.exit("--arms changes the common epoch; name the --out-dir explicitly")
    a.out_dir = a.out_dir or OUT_DIR
    alias: dict[str, str] = {}
    for spec in a.arm_alias:
        if "=" not in spec:
            sys.exit(f"--arm-alias expects FROM=TO, got {spec!r}")
        src, dst = spec.split("=", 1)
        alias[src] = dst

    bad = [c for c in a.campaigns if c not in CAMPAIGN_GROUP]
    if bad:
        sys.exit(f"not in scope: {bad}; in scope: {list(CAMPAIGN_GROUP)}")

    t_start = time.time()
    jobs, infos = [], []
    for c in a.campaigns:
        j, info = plan_jobs(c, a.epochs, a.arms)
        infos.append(info)
        jobs.extend(j)
        print(f"[plan] {c:13s} epochs={info['epochs']} arms={len(info['arms'])} "
              f"{info['arms']}" + (f" MISSING={info['missing']}" if info["missing"] else ""),
              flush=True)
    if not jobs:
        print("nothing to score", flush=True)
        return

    long_by_group: dict[str, list[dict]] = {g: [] for g in GROUPS}
    wide_by_group: dict[str, list[dict]] = {g: [] for g in GROUPS}
    n_workers = max(1, min(a.workers, len(jobs)))
    failures = []
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        for res in pool.imap_unordered(_safe_score, jobs):
            if isinstance(res, Exception):
                failures.append(str(res))
                print(f"[FAIL] {res}", flush=True)
                continue
            (campaign, arm, ep), long_rows, wide, dt = res
            g = CAMPAIGN_GROUP[campaign]
            if arm in alias:
                for r in long_rows:
                    r["arm"] = alias[arm]
                wide["arm"] = alias[arm]
            long_by_group[g].extend(long_rows)
            wide_by_group[g].append(wide)
            print(f"[done] {campaign:13s} {arm:14s} epoch={ep:3d} K={wide['K']} "
                  f"M={wide['M']:.1f} tt={wide['train_trajectories']} "
                  f"auc_f05={wide['auc_f05']:.3f} ({dt:.1f}s)", flush=True)

    written = []
    for g in GROUPS:
        if not wide_by_group[g]:
            continue
        lp = a.out_dir / f"{g}_levelsets_b50.csv"
        wp = a.out_dir / f"{g}_levelset_areas_b50.csv"
        write_csv(lp, LONG_FIELDS, merge_rows(read_csv(lp), long_by_group[g], True))
        write_csv(wp, WIDE_FIELDS, merge_rows(read_csv(wp), wide_by_group[g], False))
        written += [lp, wp]
    for p in written:
        print(f"[wrote] {p}", flush=True)
    print(f"[time] {time.time() - t_start:.1f}s, {len(jobs)} jobs, "
          f"{len(failures)} failed, {n_workers} workers", flush=True)
    if failures:
        sys.exit(1)


def _safe_score(job: tuple):
    try:
        return score_job(job)
    except Exception as e:  # surface the job identity, keep the pool alive
        return RuntimeError(f"{job[0]} {job[1]} epoch={job[2]}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
