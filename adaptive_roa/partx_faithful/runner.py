"""Standalone Part-X runner (spec §6).

Runs Alg. 4 over the trajectory pool and writes engine-format epoch directories
whenever the sample count reaches initial + e * samples_per_epoch. The snapshot
for epoch e is the current leaf partition, with each leaf's GP fit on the
initial set plus the first (b_e - initial) entries of the acquisition log
(spec §6.4). Checkpoints are written the moment the count crosses them. A run
can be killed at any point and restarted with the same command.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from adaptive_roa.adaptive_v2.engine import _atomic_write_json, _convert_numpy
from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_classifier
from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig, derive_seed
from adaptive_roa.partx_faithful.backends import backend_factory
from adaptive_roa.partx_faithful.readout import PiecewiseHandle, PiecewiseModel
from adaptive_roa.partx_faithful.region import RTYPES
from adaptive_roa.partx_faithful.sampling import PoolSampler
from adaptive_roa.partx_faithful.state import config_hash, load_state, save_state
from adaptive_roa.partx_faithful.volume import (classified_falsification_volume,
                                                gp_quantile_falsification_volume,
                                                remaining_plus_violating_volume)

FV_QUANTILES = (0.5, 0.05, 0.01)
_SNAPSHOT_STAGE = 7        # seed slot for readout fits, separate from the algorithm's


def schedule(initial: int, samples_per_epoch: int, n_epochs: int):
    """T = (n_epochs - 1) * samples_per_epoch new evaluations; one checkpoint per epoch."""
    budget = (int(n_epochs) - 1) * int(samples_per_epoch)
    return budget, [int(initial) + e * int(samples_per_epoch) for e in range(int(n_epochs))]


def resolve_n0(n0, state_dim: int) -> int:
    """Spec §6.5: the paper's 2-D setting (10) for d = 2, the F16 setting (30) otherwise."""
    if n0 is None or n0 == "auto":
        return 10 if int(state_dim) == 2 else 30
    return int(n0)


def snapshot_ids(px, n_new: int) -> set:
    """Sample ids an epoch may use: the initial set, the first n_new acquisitions, and the
    trajectory states of exactly those rollouts (spec 2026-09-15 §4.7)."""
    base = set(range(px.n_init)) | set(px.log[:n_new])
    allowed = set(base)
    traj = getattr(px, "traj_ids", {})
    for s in base:
        allowed.update(traj.get(s, ()))
    return allowed


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class CheckpointWriter:
    def __init__(self, *, output_dir, system, eval_states_file, attractor_radius, checkpoints,
                 initial, model_factory, pcfg, extra, decision_rule: str = "one_sided"):
        self.out = Path(output_dir)
        self.system, self.eval_states_file = system, str(eval_states_file)
        self.attractor_radius = float(attractor_radius)
        self.checkpoints, self.initial = list(checkpoints), int(initial)
        self.model_factory, self.pcfg, self.extra = model_factory, pcfg, extra
        self.decision_rule = str(decision_rule)
        self.next_e = 0
        self._cache: dict = {}

    def epoch_dir(self, e: int) -> Path:
        return self.out / f"epoch_{e:03d}"

    def complete(self, e: int) -> bool:
        d = self.epoch_dir(e)
        return (d / "artifacts_v2.json").exists() and (d / "full_roa_per_point.npz").exists()

    def n_new(self, e: int) -> int:
        return self.checkpoints[e] - self.initial

    def on_sample(self, px) -> None:
        """Write every checkpoint the sample count has reached (called after each batch).

        The final-budget epoch is left to finish(): the batch that reaches it can
        still be followed by that region's classification, and the epoch must
        record Part-X's final state.
        """
        while (self.next_e < len(self.checkpoints)
               and self.initial + px.used >= self.checkpoints[self.next_e]
               and self.n_new(self.next_e) < px.budget):
            e = self.next_e
            self.next_e += 1
            if not self.complete(e):
                self.write(px, e)
            else:
                self._fill_prev_d2(px, e)      # a crash may have landed between the two writes

    def finish(self, px) -> None:
        """After termination: epochs Part-X never reached carry the final state (D8)."""
        for e in range(self.next_e, len(self.checkpoints)):
            if not self.complete(e):
                self.write(px, e)
            else:
                self._fill_prev_d2(px, e)
        self.next_e = len(self.checkpoints)

    def _fill_prev_d2(self, px, e: int) -> None:
        """Match the other arms: epoch e-1 records the pool indices bought between e-1 and e."""
        if e == 0 or not self.complete(e - 1):
            return
        prev = self.epoch_dir(e - 1) / "artifacts_v2.json"
        a = json.loads(prev.read_text())
        lo, hi = min(self.n_new(e - 1), px.used), min(self.n_new(e), px.used)
        d2 = [int(i) for i in px.store.pool_idx(px.log[lo:hi])]
        if a["acquisition"].get("d2_indices") != d2:
            a["acquisition"]["d2_indices"] = d2
            _atomic_write_json(prev, a, indent=2)

    def _models(self, px, leaves, allowed: set) -> dict:
        models = {}
        for leaf in leaves:
            node = leaf
            ids = [s for s in node.samples if s in allowed]
            while not ids and node.parent >= 0:
                node = px.regions[node.parent]
                ids = [s for s in node.samples if s in allowed]
            if not ids:
                raise RuntimeError(f"leaf {leaf.rid}: no samples on the path to the root")
            key = (node.rid, len(ids))
            if key not in self._cache:
                model = self.model_factory(derive_seed(self.pcfg.seed, _SNAPSHOT_STAGE, node.rid, len(ids)))
                model.fit(px.store.X(ids), px.store.y(ids))
                self._cache = {k: v for k, v in self._cache.items() if k[0] != node.rid}
                self._cache[key] = model
            models[leaf.rid] = self._cache[key]
        return models

    def write(self, px, e: int) -> None:
        n_new = min(self.n_new(e), px.used)
        allowed = snapshot_ids(px, n_new)
        traj_on = bool(getattr(px, "traj_enabled", False))
        leaves = px.leaves()
        models = self._models(px, leaves, allowed)
        root = px.regions[0]
        d = self.epoch_dir(e)
        d.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_full_roa_classifier(
            classifier=PiecewiseHandle(PiecewiseModel(leaves, models, root.low, root.high)),
            system=self.system, eval_states_file=self.eval_states_file, lambda_star=0.5, delta=0.1,
            attractor_radius=self.attractor_radius, device="cpu", output_dir=str(d), verbose=False,
            decision_rule=self.decision_rule)
        np.savez(d / "partx_leaves.npz",
                 low=np.array([r.low for r in leaves]), high=np.array([r.high for r in leaves]),
                 rtype=np.array([r.rtype for r in leaves]), rid=np.array([r.rid for r in leaves]),
                 n_samples=np.array([sum(s in allowed for s in r.samples) for r in leaves]),
                 root_low=root.low, root_high=root.high)
        types = Counter(r.rtype for r in leaves)
        fv = gp_quantile_falsification_volume(
            leaves, models, self.pcfg.R, self.pcfg.M, FV_QUANTILES,
            np.random.default_rng(derive_seed(self.pcfg.seed, _SNAPSHOT_STAGE, e)), px.root_volume)
        diag = {
            "iteration": px.iteration, "phase": px.phase, "n_leaves": len(leaves),
            "leaf_types": {t: int(types.get(t, 0)) for t in RTYPES},
            "falsification_volume_classified": classified_falsification_volume(leaves, px.root_volume),
            "falsification_volume_px": remaining_plus_violating_volume(leaves, px.root_volume),
            "falsification_volume_gp": {str(q): v for q, v in fv.items()},
            "budget": px.budget, "used": int(n_new),
            "unspent": int(px.budget - px.used) if px.done else 0,
            "shortfall": int(px.shortfall), "n_fits": int(px.n_fits),
            "n_capped_fits": int(px.n_capped), "n_at_bound_fits": int(px.n_at_bound),
            "done": bool(px.done),
        }
        if traj_on:
            # States in THIS snapshot: those of the rollouts the epoch may use.
            diag["n_traj_states"] = int(len(allowed) - px.n_init - n_new)
        art = {
            "epoch": int(e), "train_trajectories": int(self.initial + n_new),
            "sampling_mode": "partx_faithful_traj" if traj_on else "partx_faithful",
            "threshold_state": {"lambda_star": 0.5, "delta_star": 0.1, "q_hat": None, "q_hat_eval": None,
                                "q_hat_success_eval": None, "q_hat_failure_eval": None},
            "acquisition": {"d1_indices": [], "d2_indices": [], "n_candidates_evaluated": None,
                            "n_certain_discarded": 0, "n_invalid_added": 0, "diagnostics": diag},
            "endpoint_error": {"skipped": "partx_faithful predicts no endpoints"},
            "eval_metrics": metrics, "d1_eval_metrics": None, "conformal_state": None,
            "extra": self.extra,
        }
        _atomic_write_json(d / "artifacts_v2.json", _convert_numpy(art), indent=2)
        self._fill_prev_d2(px, e)


def run_partx_faithful(*, system, starts, labels, eval_states_file, output_dir, initial: int,
                       samples_per_epoch: int, n_epochs: int, pcfg: PartXConfig, gp_kwargs: dict,
                       falsify: str, attractor_radius: float, extra: dict,
                       decision_rule: str = "one_sided", trajectory=None) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    starts = np.asarray(starts, dtype=float)
    labels = np.asarray(labels, dtype=float)
    budget, checkpoints = schedule(initial, samples_per_epoch, n_epochs)
    # Spec §3: S is the box the pool's start states occupy, padded by a relative 1e-4.
    # per_dim_bounds() describes where trajectories go, not where starts are drawn: on
    # quad3D its box holds the pool's starts in 0.54% of its volume, and every design
    # point drawn in the empty part snaps to a start on the edge of the data.
    lo, hi = starts.min(axis=0), starts.max(axis=0)
    pad = 1e-4 * np.maximum(hi - lo, 1e-9)
    root_low, root_high = lo - pad, hi + pad
    print(f"[partx_faithful] S = pool start box, low {np.array2string(root_low, precision=3)} "
          f"high {np.array2string(root_high, precision=3)}", flush=True)
    used = np.zeros(len(starts), dtype=bool)
    used[:initial] = True
    sampler = PoolSampler(starts, labels, used, scale=root_high - root_low)
    factory = backend_factory("gpc", falsify=falsify, **gp_kwargs)
    starts_digest = hashlib.sha256(np.ascontiguousarray(starts).tobytes()).hexdigest()[:16]
    labels_digest = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()[:16]
    hashed = {"pcfg": asdict(pcfg), "gp": gp_kwargs, "falsify": falsify, "initial": initial,
              "samples_per_epoch": samples_per_epoch, "n_epochs": n_epochs,
              "n_pool": len(starts), "eval_states_file": str(eval_states_file),
              "starts": starts_digest, "labels": labels_digest}
    if trajectory is not None:
        # Only when on, so every existing Part-X* checkpoint keeps its hash and still resumes.
        hashed["trajectory_states"] = True
    chash = config_hash(hashed)
    state_path = out / "partx_state.pkl"
    if state_path.exists():
        px = load_state(state_path, chash)
        px.attach(sampler, factory, trajectory)
        print(f"[partx_faithful] resuming at iteration {px.iteration}, used {px.used}/{px.budget}", flush=True)
    else:
        px = PartX(pcfg, root_low, root_high, sampler, factory, budget, init_X=starts[:initial],
                   init_y=labels[:initial], init_pool_idx=np.arange(initial), trajectory=trajectory)
    writer = CheckpointWriter(output_dir=out, system=system, eval_states_file=eval_states_file,
                              attractor_radius=attractor_radius, checkpoints=checkpoints,
                              initial=initial, model_factory=factory, pcfg=pcfg,
                              extra={**extra, "commit": _git_commit()}, decision_rule=decision_rule)
    px.on_sample = writer.on_sample
    writer.on_sample(px)                     # epoch 0, and anything a crash left unwritten
    t0 = time.time()
    if px.phase == "root":
        px.run_root()
        save_state(px, state_path, chash)
    while not px.done:
        iteration_before = px.iteration
        px.step()
        save_state(px, state_path, chash)
        if px.iteration == iteration_before:
            continue                             # this step only noticed termination
        leaves = px.leaves()
        types = Counter(r.rtype for r in leaves)
        states = f" traj_states={px.n_traj_states}" if px.traj_enabled else ""
        print(f"[partx_faithful] iter {px.iteration} phase={px.phase} used={px.used}/{px.budget} "
              f"leaves={len(leaves)} " + " ".join(f"{t}={types[t]}" for t in RTYPES if types.get(t))
              + f" fits={px.n_fits} capped={px.n_capped}{states} elapsed={time.time() - t0:.0f}s",
              flush=True)
    writer.finish(px)
    leaves = px.leaves()
    summary = {"budget": px.budget, "used": px.used, "unspent": px.budget - px.used,
               "iterations": px.iteration, "n_leaves": len(leaves),
               "leaf_types": dict(Counter(r.rtype for r in leaves)), "shortfall": px.shortfall,
               "n_fits": px.n_fits, "n_capped_fits": px.n_capped, "n_at_bound_fits": px.n_at_bound,
               "elapsed_s_this_process": time.time() - t0}
    if px.traj_enabled:
        summary.update(n_traj_states=px.n_traj_states, n_traj_rollouts=px.n_traj_rollouts,
                       n_timeout_rollouts=px.n_timeout_rollouts)
    _atomic_write_json(out / "final_results.json", _convert_numpy(summary), indent=2)
    return summary


def run_from_cfg(cfg) -> dict:
    """Build the inputs from a composed configs/adaptive_v2/partx_faithful.yaml."""
    import hydra
    from omegaconf import OmegaConf

    from adaptive_roa.adaptive.data_source import TrajectoryDataSourceConfig
    from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource
    from adaptive_roa.partx_faithful.trajectory import PoolTrajectoryStates

    if str(cfg.data_source.get("pool_format", "text")) != "npz":
        raise ValueError("partx_faithful needs an npz pool (data_source.pool_format=npz)")
    traj_on = bool(cfg.partx_faithful.get("trajectory_states", False))
    if traj_on and str(cfg.data_source.get("timeout_intermediates", "keep")) != "drop":
        raise ValueError("partx_faithful.trajectory_states needs ++data_source.timeout_intermediates=drop, "
                         "so a timeout contributes x0 only, exactly as it does for the neural arms")
    system = hydra.utils.instantiate(cfg.system)
    ds = NpzTrajectoryDataSource(TrajectoryDataSourceConfig(
        trajectories_dir=cfg.data_source.trajectories_dir,
        shuffled_indices_file=cfg.data_source.shuffled_indices_file,
        shuffled_labels_file=cfg.data_source.get("shuffled_labels_file", None),
        eval_states_file=cfg.data_source.get("eval_states_file", None),
        timeout_intermediates="drop" if traj_on else "keep"))
    if ds.labels is None:
        raise ValueError("the pool has no shuffled_labels_file; Part-X needs outcomes")
    starts = np.asarray(ds.start_states, dtype=float)
    alg = OmegaConf.to_container(cfg.partx_faithful.algorithm, resolve=True)
    alg["n0"] = resolve_n0(alg.get("n0"), starts.shape[1])
    alg["seed"] = int(cfg.get("seed", 42))
    pcfg = PartXConfig(**alg)
    gp = OmegaConf.to_container(cfg.partx_faithful.gp, resolve=True)
    gp_kwargs = dict(n_restarts=int(gp["n_restarts"]), max_points=int(gp["max_points"]),
                     lengthscale_bounds=tuple(gp["lengthscale_bounds"]),
                     sigma_f2_bounds=tuple(gp["sigma_f2_bounds"]))
    extra = {"partx_faithful": OmegaConf.to_container(cfg.partx_faithful, resolve=True),
             "n0_resolved": pcfg.n0}
    return run_partx_faithful(
        system=system, starts=starts, labels=np.asarray(ds.labels, dtype=float),
        eval_states_file=cfg.data_source.test_set_file, output_dir=cfg.output_dir,
        initial=int(cfg.initial_train_size), samples_per_epoch=int(cfg.samples_per_epoch),
        n_epochs=int(cfg.n_epochs), pcfg=pcfg, gp_kwargs=gp_kwargs,
        falsify=str(cfg.partx_faithful.falsify), attractor_radius=float(cfg.attractor_radius),
        extra=extra, decision_rule=str(cfg.get("decision_rule", "one_sided")),
        trajectory=PoolTrajectoryStates(ds) if traj_on else None)
