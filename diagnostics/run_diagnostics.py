#!/usr/bin/env python
"""Four-link diagnostic over the completed ensemble-acquisition runs. READ-ONLY.

Each of the four causal links below must hold for epistemic acquisition to
produce a gain. This script tests each and writes every computed number to
diagnostics.json keyed by level/predictor/arm/epoch.

  Link 1  the acquisition score differentiates the candidate pool
  Link 2  different scores acquire different states
  Link 3  acquired states shift the training distribution
  Link 4  ensemble disagreement stays above the measurement floor

Nothing here launches, resumes, or mutates a run.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/common/home/st1122/Projects/adaptive_roa/scripts")

EXP = Path("/common/users/shared/pracsys/adaptive_roa_experiments/ensemble_epistemic")
POOL = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr")
LEVELS = ["det", "low", "med", "high", "xhigh"]
PREDICTORS = ["clf", "fm"]
ARMS = ["dir00", "total", "epi_var", "epi_bald", "aleat"]
TREATMENT = ["total", "epi_var", "epi_bald", "aleat"]
K_ACQ = 20          # FM MC samples per member
N_MEMBERS = 5

OUT = Path(__file__).parent
D = {"meta": {}, "link1": {}, "link2": {}, "link3": {}, "link4": {}, "sources": {}}


# ---------------------------------------------------------------- inventory
def load_verdicts_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ev", "/common/home/st1122/Projects/adaptive_roa/scripts/ensemble_verdicts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def epochs_of(run_dir: Path):
    """Completed epochs only: an epoch counts when artifacts_v2.json exists."""
    out = {}
    for f in sorted(run_dir.glob("epoch_*/artifacts_v2.json")):
        try:
            out[int(f.parent.name.split("_")[1])] = json.load(open(f))
        except Exception:
            continue
    return out


def inventory():
    ev = load_verdicts_module()
    flagged = ev.contaminated_runs(EXP)
    included, excluded = [], []
    for pred, lvl, arm in itertools.product(PREDICTORS, LEVELS, ARMS):
        name = f"{pred}_{lvl}_{arm}"
        p = EXP / name
        if not p.is_dir():
            excluded.append({"run": name, "reason": "no run directory"})
            continue
        if name in flagged:
            excluded.append({"run": name,
                             "reason": f"contaminated_runs(): {flagged[name]} epoch-mtime inversion(s)"})
            continue
        n = len(list(p.glob("epoch_*/artifacts_v2.json")))
        if n == 0:
            excluded.append({"run": name, "reason": "no completed epochs"})
            continue
        included.append({"run": name, "completed_epochs": n, "path": str(p)})
    D["meta"]["included"] = included
    D["meta"]["excluded"] = excluded
    D["meta"]["contaminated_flagged"] = {k: int(v) for k, v in flagged.items()}
    D["meta"]["constants"] = {"n_members": N_MEMBERS, "k_acq_fm": K_ACQ,
                              "clf_members": "exact (K=None)"}
    return {r["run"] for r in included}


# ------------------------------------------------- Link 1: score over pool
def link1(live):
    """Pool-level score statistics.

    These come from acquisition.diagnostics, which the strategy writes over the
    FULL scored candidate pool (n_candidates_evaluated), so they are pool-level
    and NOT a held-out proxy. Quantiles and the exactly-zero fraction were never
    stored -- only min/max/mean and the selected-set mean -- so the degeneracy
    criterion is applied to what exists and the gap is declared.
    """
    for pred, lvl in itertools.product(PREDICTORS, LEVELS):
        for arm in TREATMENT:
            name = f"{pred}_{lvl}_{arm}"
            if name not in live:
                continue
            rows = {}
            for ep, art in epochs_of(EXP / name).items():
                if ep == 0:
                    continue
                dg = (art.get("acquisition") or {}).get("diagnostics") or {}
                if "score_mean" not in dg:
                    continue
                smin, smax = float(dg["score_min"]), float(dg["score_max"])
                smean = float(dg["score_mean"])
                rows[ep] = {
                    "score_min": smin, "score_max": smax, "score_mean": smean,
                    "score_mean_selected": float(dg.get("score_mean_selected", float("nan"))),
                    "epistemic_mean_pool": float(dg.get("epistemic_mean", float("nan"))),
                    "aleatoric_mean_pool": float(dg.get("aleatoric_mean", float("nan"))),
                    "epistemic_mean_selected": float(dg.get("epistemic_mean_selected", float("nan"))),
                    "aleatoric_mean_selected": float(dg.get("aleatoric_mean_selected", float("nan"))),
                    "n_candidates_evaluated": int(dg.get("n_candidates_evaluated", 0)),
                    "member_sample_size": dg.get("member_sample_size"),
                    "selection_rule": dg.get("selection_rule"),
                    # derived
                    "range": smax - smin,
                    "mean_over_max": (smean / smax) if smax > 0 else float("nan"),
                    "enrichment": (float(dg.get("score_mean_selected", np.nan)) / smean)
                                  if smean not in (0.0,) else float("nan"),
                    "cv_proxy_mean_over_range": (smean / (smax - smin)) if smax > smin else float("nan"),
                }
            if rows:
                D["link1"][name] = rows
    D["sources"]["link1"] = ("<run>/epoch_*/artifacts_v2.json :: acquisition.diagnostics "
                             "(pool-level, n_candidates_evaluated points)")


# --------------------------------------------- Link 2: acquired-set overlap
def acquired(run: str):
    """{epoch: set(d2 indices)} and the cumulative union. dir00 uses d1."""
    per, cum = {}, set()
    for ep, art in epochs_of(EXP / run).items():
        acq = art.get("acquisition") or {}
        idx = list(acq.get("d2_indices") or []) or list(acq.get("d1_indices") or [])
        if not idx:
            continue
        s = set(int(i) for i in idx)
        per[ep] = s
        cum |= s
    return per, cum


def jac(a, b):
    u = len(a | b)
    return (len(a & b) / u) if u else float("nan")


def link2(live):
    for pred, lvl in itertools.product(PREDICTORS, LEVELS):
        present = [a for a in ARMS if f"{pred}_{lvl}_{a}" in live]
        if len(present) < 3:
            continue
        per, cum = {}, {}
        for a in present:
            per[a], cum[a] = acquired(f"{pred}_{lvl}_{a}")
        key = f"{pred}_{lvl}"
        entry = {"n_cumulative": {a: len(cum[a]) for a in present},
                 "cumulative_jaccard": {}, "per_epoch_jaccard": {},
                 "vs_dir00_cumulative": {}}
        for x, y in itertools.combinations([a for a in present if a in TREATMENT], 2):
            entry["cumulative_jaccard"][f"{x}|{y}"] = jac(cum[x], cum[y])
        if "dir00" in present:
            for x in [a for a in present if a in TREATMENT]:
                entry["vs_dir00_cumulative"][x] = jac(cum[x], cum["dir00"])
        shared = sorted(set.intersection(*[set(per[a]) for a in present]) - {0}) if all(per[a] for a in present) else []
        for ep in shared:
            e = {}
            for x, y in itertools.combinations([a for a in present if a in TREATMENT], 2):
                e[f"{x}|{y}"] = jac(per[x][ep], per[y][ep])
            if "dir00" in present:
                for x in [a for a in present if a in TREATMENT]:
                    e[f"{x}|dir00"] = jac(per[x][ep], per["dir00"][ep])
            entry["per_epoch_jaccard"][ep] = e
        D["link2"][key] = entry
    D["sources"]["link2"] = ("<run>/epoch_*/artifacts_v2.json :: acquisition.d2_indices "
                             "(d1_indices for dir00, which acquires uniformly)")


# ------------------------------------- Link 3: realized distribution shift
_starts_cache, _grid_cache = {}, {}


def pool_starts(level: str):
    """Start states in POOL-INDEX order.

    Mirrors scripts/acquisition_diagnostics.py:pool_starts. Acquisition records
    pool indices k, NOT npz rows; the loader maps k through the shuffle with
    npz_row = rollout_ids[k]. Indexing the npz directly with k returns unrelated
    states that look exactly like random selection -- a convincing wrong answer.
    """
    if level in _starts_cache:
        return _starts_cache[level]
    with np.load(POOL / level / "train.npz") as z:
        starts = z["starts"].astype(np.float64)
    ids = np.loadtxt(POOL / level / "train_test_splits" / "shuffled_indices_0.txt",
                     dtype=np.int64, ndmin=1)
    _starts_cache[level] = starts[ids]
    return _starts_cache[level]


def pool_labels(level: str):
    f = POOL / level / "train_test_splits" / "shuffled_labels_0.txt"
    return np.loadtxt(f, dtype=np.int64, ndmin=1)


def grid_lookup(level: str):
    if level in _grid_cache:
        return _grid_cache[level]
    from scipy.spatial import cKDTree
    with np.load(POOL / level / "eval_success_prob.npz") as z:
        s = z["starts"].astype(np.float64)
        p = z["p_success"].astype(np.float64)
    tree = cKDTree(s)
    _grid_cache[level] = lambda q: p[tree.query(q, k=1)[1]]
    return _grid_cache[level]


def link3(live):
    stoch = [l for l in LEVELS if l != "det"]
    for lvl in stoch:
        if not (POOL / lvl / "train.npz").exists():
            D["link3"][lvl] = {"status": "UNAVAILABLE", "reason": f"no pool at {POOL/lvl}"}
            continue
        starts = pool_starts(lvl)
        labels = pool_labels(lvl)
        look = grid_lookup(lvl)
        for pred in PREDICTORS:
            for arm in ARMS:
                name = f"{pred}_{lvl}_{arm}"
                if name not in live:
                    continue
                per, _ = acquired(name)
                rows, cum_lab = {}, []
                for ep in sorted(per):
                    if ep == 0:
                        continue
                    idx = np.fromiter(per[ep], dtype=np.int64)
                    idx = idx[(idx >= 0) & (idx < len(starts))]
                    if idx.size == 0:
                        continue
                    p_true = look(starts[idx])
                    lab = labels[idx] if idx.max() < len(labels) else np.array([])
                    cum_lab.append(lab)
                    rows[ep] = {
                        "n_acquired": int(idx.size),
                        "batch_mean_true_p": float(np.mean(p_true)),
                        "batch_frac_ambiguous_0.2_0.8": float(np.mean((p_true > 0.2) & (p_true < 0.8))),
                        "batch_frac_decided": float(np.mean((p_true < 0.05) | (p_true > 0.95))),
                        "batch_mean_abs_p_minus_half": float(np.mean(np.abs(p_true - 0.5))),
                        "batch_label_marginal": float(np.mean(lab)) if lab.size else None,
                        "cumulative_label_marginal": float(np.mean(np.concatenate(cum_lab)))
                                                     if cum_lab else None,
                    }
                if rows:
                    D["link3"].setdefault(f"{pred}_{lvl}", {})[arm] = rows
    D["sources"]["link3"] = {
        "pool_states": str(POOL / "<level>/train.npz :: starts, permuted by "
                                  "train_test_splits/shuffled_indices_0.txt"),
        "pool_labels": str(POOL / "<level>/train_test_splits/shuffled_labels_0.txt"),
        "true_p": str(POOL / "<level>/eval_success_prob.npz :: nearest-grid lookup (158x315)"),
        "note": "det excluded: deterministic system, p degenerate in {0,1}; no lqr/det pool tree",
    }


# --------------------------- Link 4: diversity trajectory + lineage audit
def link4(live):
    """Ensemble disagreement vs the MC floor.

    Per-member probabilities on the held-out eval set were NEVER stored (only the
    ensemble mean p_success is in full_roa_per_point.npz), so the requested
    eval-set Var_m[p_m] is UNAVAILABLE without re-running inference. What IS
    stored, per epoch, is the pool-level epistemic and aleatoric means from the
    acquisition diagnostics -- computed from the same per-member probabilities at
    acquisition time. Those are used, and labelled POOL rather than eval-set.

    The MC floor for FM is mean_m[p(1-p)/(K-1)]. It is not stored, but the
    epi_var arm's score_min is exactly -(that floor) whenever some candidate has
    raw between-member variance 0, which gives an empirical read on it.
    """
    for pred, lvl in itertools.product(PREDICTORS, LEVELS):
        for arm in TREATMENT:
            name = f"{pred}_{lvl}_{arm}"
            if name not in live:
                continue
            rows = {}
            for ep, art in epochs_of(EXP / name).items():
                if ep == 0:
                    continue
                dg = (art.get("acquisition") or {}).get("diagnostics") or {}
                if "epistemic_mean" not in dg:
                    continue
                epi = float(dg["epistemic_mean"])
                r = {"epistemic_mean_pool": epi,
                     "aleatoric_mean_pool": float(dg.get("aleatoric_mean", np.nan)),
                     "score_min": float(dg.get("score_min", np.nan)),
                     "member_sample_size": dg.get("member_sample_size")}
                if arm == "epi_var":
                    # debiased variance; score_min <= 0 means the debias exceeded raw var
                    floor_emp = -float(dg.get("score_min", np.nan))
                    r["mc_floor_empirical_from_score_min"] = floor_emp
                    r["debiased_var_mean"] = epi
                    r["signal_over_floor"] = (epi / floor_emp) if floor_emp > 0 else float("nan")
                    r["raw_var_mean_implied"] = epi + floor_emp if floor_emp == floor_emp else float("nan")
                rows[ep] = r
            if rows:
                D["link4"].setdefault(f"{pred}_{lvl}", {})[arm] = rows

    D["link4"]["_eval_set_variance"] = {
        "status": "UNAVAILABLE",
        "reason": ("full_roa_per_point.npz stores only ensemble-mean p_success/p_failure/"
                   "p_invalid; per-member arrays were never written. Recomputing needs "
                   "inference over 5 members x 19 epochs x ~40k eval points per run, which "
                   "is not a minutes-scale job on available hardware."),
        "checked": "np.load(full_roa_per_point.npz).files == "
                   "[start_states,p_success,p_failure,p_invalid,true_labels,lambda_star,delta,attractor_radius]",
    }

    # ---- warm-start lineage audit
    audit = {}
    import yaml
    for pred, lvl, arm in itertools.product(PREDICTORS, LEVELS, ARMS):
        name = f"{pred}_{lvl}_{arm}"
        if name not in live:
            continue
        cfg = EXP / name / ".hydra" / "config.yaml"
        if not cfg.exists():
            audit[name] = {"warm_start": "NO CONFIG", "path": str(cfg)}
            continue
        try:
            c = yaml.safe_load(open(cfg))
            audit[name] = {"warm_start": c.get("warm_start"), "path": str(cfg)}
        except Exception as e:
            audit[name] = {"warm_start": f"parse error: {e}", "path": str(cfg)}
    D["link4"]["_warm_start_audit"] = audit
    D["sources"]["link4"] = {
        "diversity": "<run>/epoch_*/artifacts_v2.json :: acquisition.diagnostics."
                     "{epistemic_mean,aleatoric_mean,score_min} (POOL-level, not eval set)",
        "lineage": "<run>/.hydra/config.yaml :: warm_start; "
                   "engine.py:165 resume_ckpt = previous_best_checkpoint if (warm_start and ...) else None",
    }


def main():
    live = inventory()
    print(f"included {len(live)} runs; excluded {len(D['meta']['excluded'])}")
    link1(live); print("link1 done")
    link2(live); print("link2 done")
    link4(live); print("link4 done")
    link3(live); print("link3 done")
    (OUT / "diagnostics.json").write_text(json.dumps(D, indent=1, default=str))
    print(f"wrote {OUT/'diagnostics.json'}")


if __name__ == "__main__":
    main()
