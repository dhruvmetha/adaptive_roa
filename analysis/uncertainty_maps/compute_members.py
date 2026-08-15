#!/usr/bin/env python
"""Recompute per-member success probabilities over the evaluation grid.

The aleatoric/epistemic split cannot be read off any stored artifact: only the
ensemble marginal was serialised. This script reloads the surviving per-epoch
member checkpoints and re-evaluates them, writing one npz per (arm, epoch) with
a [M, N] array of per-member p(success).

    p_members[m, n] = P(state n lands in the success attractor | member m)

CLASSIFIER runs are one exact forward pass per member -- no sampling, so BALD is
unbiased and the whole level runs on CPU in minutes.

FLOW-MATCHING runs need K endpoint samples per member per state, so each p_m is
a K-sample binomial estimate carrying sampling noise. That noise inflates raw
between-member variance by mean_m[p_m(1-p_m)]/(K-1) and inflates BALD by roughly
(1/2K)(1-1/M); `common.decompose` carries a debiased variance for this reason and
K is recorded in the npz so downstream code can apply it.

Both paths mirror the production backends in
adaptive_roa/adaptive_v2/probability/ensemble_prob.py exactly: the classifier
path embeds and normalises before a raw net forward, the flow-matching path
passes RAW states to each member's own `predict_endpoint` (which normalises
internally) and classifies the endpoint with the run's own attractor radius.

Usage
-----
    python compute_members.py --run clf_high_total --out <dir>          # all epochs
    python compute_members.py --pred fm --level high --arms all --k 20 --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.uncertainty_maps.common import (  # noqa: E402
    ARMS, EXP, load_marginal, epochs_with_eval,
)

DATA_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories"


# ------------------------------------------------------------------ config
def load_cfg(run: str):
    """Run config with the interpolations the campaign resolved at launch.

    `${data_dir}` is a custom OmegaConf resolver registered by the training
    entrypoint, absent here, so it is re-registered against the same .env value
    rather than hand-editing the saved config.

    Only runs that completed a full launch carry a root-level `.hydra/`; the
    ones that were resumed after preemption wrote it under the epoch directory
    they restarted at instead. Falling back to the earliest epoch copy is safe
    because the fields read here (data_source, system, attractor_radius,
    flow_matcher target) are fixed for the life of a run.
    """
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda: DATA_DIR)
    root = EXP / run / ".hydra" / "config.yaml"
    if root.exists():
        return OmegaConf.load(root)
    for f in sorted((EXP / run).glob("epoch_*/.hydra/config.yaml")):
        return OmegaConf.load(f)
    raise FileNotFoundError(f"no .hydra/config.yaml anywhere under {EXP / run}")


def build_system(cfg):
    from adaptive_roa.systems.pendulum import PendulumSystem
    return PendulumSystem(dataset_dir=str(OmegaConf.select(cfg, "system.dataset_dir",
                                                           default=None)))


# -------------------------------------------------------------- classifier
def clf_members(run: str, epoch: int, states: np.ndarray, device: str) -> tuple:
    from adaptive_roa.probabilistic_classifier.bayesian import EnsembleProbabilisticClassifier

    cfg = load_cfg(run)
    system = build_system(cfg)
    clf = EnsembleProbabilisticClassifier.load_from_run(
        str(EXP / run), epoch, cfg, system, device=device)
    post = getattr(clf.handle, "posterior", clf.handle)

    out = []
    with torch.no_grad():
        for i in range(0, len(states), 16384):
            x = torch.as_tensor(states[i:i + 16384], dtype=torch.float32, device=device)
            emb = system.embed_state_for_model(system.normalize_state(x))
            out.append(torch.sigmoid(post.forward_all_members(emb)).squeeze(-1)
                       .double().cpu().numpy())
    return np.concatenate(out, axis=1), None      # [M, N], K=None (exact)


# ----------------------------------------------------------- flow matching
def fm_members(run: str, epoch: int, states: np.ndarray, device: str,
               k: int, batch: int) -> tuple:
    from hydra.utils import get_class

    cfg = load_cfg(run)
    system = build_system(cfg)
    radius = float(cfg.attractor_radius)
    cls = get_class(cfg.flow_matcher._target_)

    ep_dir = EXP / run / f"epoch_{epoch:03d}"
    member_dirs = sorted(ep_dir.glob("member_*"))
    if not member_dirs:
        raise FileNotFoundError(f"no member_* under {ep_dir}")

    x_all = torch.as_tensor(states, dtype=torch.float32, device=device)
    out = np.empty((len(member_dirs), len(states)), dtype=np.float64)

    for m, md in enumerate(member_dirs):
        ckpts = sorted(glob.glob(str(md / "**" / "best*.ckpt"), recursive=True))
        if not ckpts:
            raise FileNotFoundError(
                f"no best*.ckpt under {md}; member {m} is missing and an ensemble "
                "assembled without it would report a wrong epistemic estimate.")
        model = cls.load_from_checkpoint(ckpts[0], device=device)
        model.eval().to(device)
        hits = torch.zeros(len(states), dtype=torch.float64, device=device)
        with torch.no_grad():
            for _ in range(k):
                for i in range(0, len(states), batch):
                    xb = x_all[i:i + batch]
                    pred = model.predict_endpoint(xb)
                    lab = system.classify_attractor(pred, radius)
                    hits[i:i + batch] += (lab == 1).double()
        out[m] = (hits / float(k)).cpu().numpy()
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out, k


# --------------------------------------------------------------------- cli
def process(run: str, epochs: list[int], out_root: Path, device: str,
            k: int, batch: int, overwrite: bool) -> None:
    pred, level, arm = run.split("_", 2)
    dest = out_root / run
    dest.mkdir(parents=True, exist_ok=True)

    for ep in epochs:
        f = dest / f"members_epoch_{ep:03d}.npz"
        if f.exists() and not overwrite:
            print(f"  ep{ep:03d} exists, skip", flush=True)
            continue
        marg = load_marginal(pred, level, arm, ep)
        states = marg["states"]
        t0 = time.time()
        if pred == "clf":
            p_members, k_used = clf_members(run, ep, states, device)
        else:
            p_members, k_used = fm_members(run, ep, states, device, k, batch)
        dt = time.time() - t0

        # The recomputed marginal will not equal the stored one for flow
        # matching (different K, fresh sampling), but for the classifier it is
        # a deterministic forward pass and MUST match. Record the gap either
        # way so a silently mis-loaded checkpoint is visible downstream.
        drift = float(np.abs(p_members.mean(axis=0) - marg["p_success"]).max())
        np.savez_compressed(
            f,
            p_members=p_members.astype(np.float32),
            states=states.astype(np.float32),
            k=np.array(k_used if k_used is not None else 0),
            marginal_stored=marg["p_success"].astype(np.float32),
            marginal_drift_max=np.array(drift),
        )
        print(f"  ep{ep:03d}  M={p_members.shape[0]}  N={p_members.shape[1]}  "
              f"drift={drift:.4f}  {dt:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="explicit run name, e.g. clf_high_total")
    ap.add_argument("--pred", choices=("clf", "fm"))
    ap.add_argument("--level")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--epochs", default="all", help="'all' or comma list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k", type=int, default=20,
                    help="MC endpoint samples per member (flow matching only)")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if a.run:
        runs = [a.run]
    else:
        if not (a.pred and a.level):
            ap.error("give --run, or both --pred and --level")
        arms = ARMS if a.arms == "all" else tuple(a.arms.split(","))
        runs = [f"{a.pred}_{a.level}_{arm}" for arm in arms]

    out_root = Path(a.out)
    for run in runs:
        pred, level, arm = run.split("_", 2)
        avail = epochs_with_eval(pred, level, arm)
        eps = avail if a.epochs == "all" else [int(e) for e in a.epochs.split(",")]
        eps = [e for e in eps if e in avail]
        print(f"{run}: {len(eps)} epochs on {a.device}", flush=True)
        process(run, eps, out_root, a.device, a.k, a.batch, a.overwrite)

    (out_root / "manifest.json").write_text(json.dumps(
        {"runs": runs, "k": a.k, "device": a.device}, indent=2))


if __name__ == "__main__":
    main()
