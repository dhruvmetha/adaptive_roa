"""TERA-FM endpoint counts and certified decisions on arbitrary start states.

The counts come from the run's own evaluator, not a reimplementation:
``FullROAEvaluator.evaluate_epoch`` -> ``evaluate_full_roa_fast`` in
``adaptive_roa/adaptive_v2/eval/full_roa.py``, called the way ``AdaptiveEngine`` calls it
(engine.py ~line 604), on a temporary eval-states file in the ``test_set.txt`` layout.

* System: ``hydra.utils.instantiate(cfg.system)`` from the run's ``resolved_config.yaml``
  (engine.py ~line 147).
* Flow matcher: ``<flow_matcher class>.load_from_checkpoint(<run>/epoch_010/checkpoints/best-*.ckpt)``,
  then ``.eval().to(device)``, as ``FlowMatchingTrainer.fit`` (~line 219) and engine.py ~line 433 do.
* Evaluator: ``FullROAEvaluator(cfg.eval, system, device)``, so K = ``eval.num_mc_samples_eval``,
  the attractor radius, ``refine_invalids`` and ``binary_outcomes`` are the run's own values.
  Batch size is the engine's ``cfg.get("val_batch_size", 2048)``.
* Decisions: ``in_S`` / ``in_F`` and the tie rounding from ``adaptive_v2/eval/certificates.py``.

The run's own RNG state at its epoch_010 evaluation cannot be recovered, so a rerun reproduces
its predictions in distribution (K = 100 binomial noise), not bit for bit. A rerun of this
function with the same seed, batch size and state order reproduces its own counts.
"""
from __future__ import annotations

import glob
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

DETV3 = Path(__file__).resolve().parents[3]  # the adaptive_roa checkout this folder sits in
SLICES = Path(__file__).resolve().parents[1]
SCRATCH = Path(__file__).resolve().parent / "_scratch"
INDEX_DIR = Path("/common/users/shared/pracsys/adaptive_roa_experiments/det_v3_seeds/_index")
MATRIX_ROWS = INDEX_DIR / "scored" / "matrix_rows.jsonl"
RUNS_INDEX = INDEX_DIR / "runs_index.tsv"
EPOCH = "epoch_010"

# The README's run table (seed rule: median c6c_rec of three seeds).
RUNS = {
    "cp-lqr": "x29_cplqr_fm_uh_a6d075_r1",
    "cp-ppo": "x161_cpppo_fm_uh_a6d075_r2",
    "q2d-ppo": "x31_q2dppo_fm_uh_a6d075_r2",
    "q3d-lqr": "x33_q3dlqr_fm_uh_a6d075_r2",
    "q3d-ppo": "x70_q3dppo_fm_uh_a6d075_r1",
}

# Paths whose writes we never allow, whatever a caller passes as out_dir.
_FORBIDDEN = ("/common/users/shared/pracsys/genMoPlan", "/common/users/shared/pracsys/adaptive_roa_experiments",
              "/research/projects/pracsys")


def _import_detv3():
    """Import adaptive_roa from the detv3 worktree, never from the editable install."""
    if "adaptive_roa" not in sys.modules and str(DETV3) not in sys.path:
        sys.path.insert(0, str(DETV3))
    import adaptive_roa
    where = Path(adaptive_roa.__file__).resolve()
    if DETV3 not in where.parents:
        raise ImportError(f"adaptive_roa imported from {where}, not {DETV3}. Set PYTHONPATH={DETV3} "
                          f"before anything imports adaptive_roa.")
    return adaptive_roa


def adaptive_roa_file() -> str:
    """Path of the adaptive_roa package that this module uses (proves which code ran)."""
    return _import_detv3().__file__


# --- run lookup and thresholds -------------------------------------------------------------------

def run_dir(run_name: str) -> str:
    """``real_dir`` of ``run_name`` in runs_index.tsv. Raises when absent or ambiguous."""
    with open(RUNS_INDEX) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_run, i_dir = header.index("run"), header.index("real_dir")
        dirs = {row[i_dir] for line in f if (row := line.rstrip("\n").split("\t"))[i_run] == run_name}
    if len(dirs) != 1:
        raise KeyError(f"{run_name}: {len(dirs)} distinct real_dir values in {RUNS_INDEX}: {sorted(dirs)}")
    d = dirs.pop()
    if not Path(d).is_dir():
        raise FileNotFoundError(f"{run_name}: real_dir {d} is not a directory on this host")
    return d


def thresholds(run_name: str) -> dict:
    """``dict(t_s, t_f, u_s, u_f)`` from the c6c_* columns of the matrix row named ``run_name``.

    Raises when the row is missing or duplicated, when c6c_pass is not true, or when any
    threshold is null. Never substitutes other thresholds.
    """
    rows = [r for line in MATRIX_ROWS.read_text().splitlines() if line.strip()
            and (r := json.loads(line)).get("name") == run_name]
    if len(rows) != 1:
        raise KeyError(f"{run_name}: {len(rows)} rows in {MATRIX_ROWS}, expected 1")
    r = rows[0]
    if r.get("c6c_pass") is not True:
        raise ValueError(f"{run_name}: c6c_pass = {r.get('c6c_pass')!r}, not true")
    out = {k: r.get(f"c6c_{k}") for k in ("t_s", "t_f", "u_s", "u_f")}
    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise ValueError(f"{run_name}: null c6c thresholds {missing}")
    return {k: float(v) for k, v in out.items()}


# --- model loading -------------------------------------------------------------------------------

# Runs trained on Amarel name their data under a /scratch copy of genMoPlan that this host cannot
# see. Strings in the run config and in the checkpoint hyper_parameters that start with a key are
# rewritten to the value, in memory only (never in the run directory or the checkpoint file).
PATH_REMAP = {
    "/scratch/dm1487/genMoPlan-exp/data_trajectories": "/common/users/shared/pracsys/genMoPlan/data_trajectories",
}

_CACHE: dict = {}


def _remap_tree(obj, where: str, log: list):
    """Copy of a dict/list tree with every PATH_REMAP prefix rewritten; one log line per change."""
    if isinstance(obj, dict):
        return {k: _remap_tree(v, f"{where}.{k}", log) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_remap_tree(v, f"{where}[{i}]", log) for i, v in enumerate(obj))
    if isinstance(obj, str):
        for old, new in PATH_REMAP.items():
            if obj == old or obj.startswith(old + "/"):
                out = new + obj[len(old):]
                log.append(f"{where}: {obj} -> {out}")
                return out
    return obj


class _serve_checkpoint:
    """While active, torch.load(<path>) returns the given in-memory checkpoint dict."""

    def __init__(self, path, ckpt):
        self.path, self.ckpt = str(Path(path)), ckpt

    def __enter__(self):
        import torch
        self._orig = torch.load

        def load(f, *a, **k):
            if isinstance(f, (str, os.PathLike)) and str(Path(f)) == self.path:
                return self.ckpt
            return self._orig(f, *a, **k)

        torch.load = load
        return self

    def __exit__(self, *exc):
        import torch
        torch.load = self._orig
        return False


def load_run(run_dir_: str, device: str = "cuda:1"):
    """(cfg, system, flow_matcher, evaluator, ckpt path) for ``run_dir_``, cached per (run_dir, device).

    Also caps torch CPU threads at 4: this host's CPUs are shared and the GPU loop needs few.
    """
    key = (str(Path(run_dir_).resolve()), device)
    if key in _CACHE:
        return _CACHE[key]
    _import_detv3()
    import hydra
    import torch
    from omegaconf import OmegaConf
    from adaptive_roa.adaptive_v2.engine import _instantiate

    torch.set_num_threads(min(4, torch.get_num_threads()))
    log: list[str] = []
    raw = OmegaConf.to_container(OmegaConf.load(Path(run_dir_) / "resolved_config.yaml"), resolve=False)
    cfg = OmegaConf.create(_remap_tree(raw, "cfg", log))
    if cfg.predictor.type != "generative":
        raise ValueError(f"{run_dir_}: predictor.type={cfg.predictor.type}, expected generative (FM)")
    system = hydra.utils.instantiate(cfg.system)
    ckpts = sorted(glob.glob(str(Path(run_dir_) / EPOCH / "checkpoints" / "best*.ckpt")))
    if len(ckpts) != 1:
        raise FileNotFoundError(f"{run_dir_}/{EPOCH}/checkpoints: expected one best*.ckpt, found {ckpts}")
    fm_cls = hydra.utils.get_class(cfg.flow_matcher._target_)
    # Read the checkpoint once; if its hyper_parameters name a remapped path, hand the patched dict
    # to load_from_checkpoint (in memory only). Otherwise the call is exactly the trainer's.
    ckpt_dict = torch.load(ckpts[0], map_location=device, weights_only=False)
    n_before = len(log)
    ckpt_dict["hyper_parameters"] = _remap_tree(ckpt_dict.get("hyper_parameters", {}), "ckpt.hparams", log)
    for line in log:
        print(f"[infer] path remap ({Path(run_dir_).name}) {line}", flush=True)
    if len(log) > n_before:
        with _serve_checkpoint(ckpts[0], ckpt_dict):
            fm = fm_cls.load_from_checkpoint(ckpts[0], device=device)
    else:
        del ckpt_dict
        fm = fm_cls.load_from_checkpoint(ckpts[0], device=device)
    fm.eval()
    fm.to(device)
    evaluator = _instantiate(cfg.eval, system, device)
    # No plots from our calls: an epoch outside plot_epochs gives make_plots=False.
    evaluator.plot_epochs = set()
    _CACHE[key] = (cfg, system, fm, evaluator, ckpts[0])
    return _CACHE[key]


def _check_out_dir(p: Path) -> Path:
    p = p.resolve()
    for bad in _FORBIDDEN:
        if str(p).startswith(bad):
            raise PermissionError(f"refusing to write under {bad}: {p}")
    if SLICES not in p.parents and p != SLICES:
        raise PermissionError(f"out_dir must be under {SLICES}: {p}")
    return p


# --- counts --------------------------------------------------------------------------------------

def run_counts(run_dir: str, model_states, device: str = "cuda:1", seed: int = 0, out_dir=None,
               batch_size: int | None = None, labels01=None, keep_files: bool = False,
               verbose: bool = False) -> dict:
    """Endpoint counts of TERA-FM on ``model_states`` (N, D) float32, dataset state convention.

    Returns ``dict(n_success, n_failure, n_invalid, K)`` (int arrays of length N and the int K),
    plus ``seconds`` (wall time of the evaluator call) and ``ckpt``.

    ``out_dir``: where the temporary eval file and the evaluator's full_roa_per_point.npz go;
    default ``lib/_scratch/``. Must be under ``slices/``. A fresh subdirectory is made per call
    and removed afterwards unless ``keep_files``.
    ``batch_size``: None uses the run's own (``val_batch_size``, 2048). Another value changes
    which noise draw each row gets (same distribution), not what is computed.
    ``labels01``: optional ground truth (1 success, 0 fail), only so the evaluator's own printed
    metrics mean something; it does not affect the counts.
    """
    _import_detv3()
    import torch
    from adaptive_roa.adaptive_v2.types import ThresholdState

    X = np.asarray(model_states)
    if X.ndim != 2:
        raise ValueError(f"model_states must be (N, D), got {X.shape}")
    if X.dtype != np.float32:
        raise TypeError(f"model_states must be float32 (the evaluator's dtype), got {X.dtype}")
    if not np.isfinite(X).all():
        raise ValueError("model_states has non-finite values")
    cfg, system, fm, evaluator, ckpt = load_run(run_dir, device)
    D = int(system.state_dim)
    if X.shape[1] != D:
        raise ValueError(f"model_states has {X.shape[1]} columns, the run's system has state_dim={D}")

    base = _check_out_dir(Path(out_dir) if out_dir is not None else SCRATCH)
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="counts_", dir=base))
    try:
        # The probabilistic test_set layout: D state columns + p_success. With state_dim known,
        # load_eval_states reads it unambiguously and returns end_states=None.
        lab = np.zeros(len(X)) if labels01 is None else np.asarray(labels01, dtype=float)
        eval_file = work / "eval_states.txt"
        # %.9g round-trips float32 exactly through np.loadtxt(float64).astype(float32).
        np.savetxt(eval_file, np.column_stack([X.astype(np.float64), lab]), fmt="%.9g", delimiter=",")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)              # seeds every CUDA device too
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        bs = int(batch_size) if batch_size is not None else int(cfg.get("val_batch_size", 2048))
        ts = ThresholdState(lambda_star=0.5, delta_star=0.05)   # only labels pred_conformal; counts ignore it
        t0 = time.time()
        evaluator.evaluate_epoch(fm, ts, {
            "eval_states_file": str(eval_file),
            "batch_size": bs,
            "output_dir": str(work),
            "verbose": verbose,
            "store_endpoints": False,
            "epoch": -1,
        })
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        seconds = time.time() - t0

        # Read and close the npz before the cleanup: an open handle on NFS leaves a .nfs* file
        # that makes the temp directory impossible to remove.
        with np.load(work / "full_roa_per_point.npz", allow_pickle=True) as z:
            got = {k: np.array(z[k]) for k in ("start_states", "p_success", "p_failure", "p_invalid")}
        if not np.array_equal(got["start_states"], X):
            raise RuntimeError("evaluator's start_states differ from model_states (file round trip or row order)")
        K = int(evaluator.num_mc_samples_eval)
        n = {}
        for k in ("success", "failure", "invalid"):
            c = np.asarray(got[f"p_{k}"], dtype=float) * K
            r = np.rint(c)
            if np.abs(c - r).max() > 1e-6:
                raise RuntimeError(f"p_{k} * K is not integral (K={K})")
            n[f"n_{k}"] = r.astype(np.int32)
        if not np.all(n["n_success"] + n["n_failure"] + n["n_invalid"] == K):
            raise RuntimeError("counts do not sum to K (binary_outcomes fold or a K mismatch)")
        return dict(**n, K=K, seconds=seconds, ckpt=ckpt)
    finally:
        if not keep_files:
            shutil.rmtree(work, ignore_errors=True)


# --- decisions -----------------------------------------------------------------------------------

def decide(counts: dict, thr: dict) -> np.ndarray:
    """int8 decisions: 1 Success, 0 Fail, -1 Abstain.

    ps = tie(n_success / K), pf = tie(n_failure / K); Success = in_S and not in_F,
    Fail = in_F and not in_S, Abstain otherwise. ``in_S``, ``in_F`` and ``tie`` are imported
    from the detv3 code. (certificates.report gives S priority on an S/F overlap; this rule
    abstains there. For the five README runs u_s < t_s, so the sets cannot overlap and the
    two rules agree.)
    """
    _import_detv3()
    from adaptive_roa.adaptive_v2.eval.certificates import in_S, in_F
    from adaptive_roa.adaptive_v2.eval.det_v2_rule import tie

    K = float(counts["K"])
    ps = tie(np.asarray(counts["n_success"], dtype=float) / K)
    pf = tie(np.asarray(counts["n_failure"], dtype=float) / K)
    s = in_S(ps, pf, thr["t_s"], thr["t_f"])
    f = in_F(ps, pf, thr["u_s"], thr["u_f"])
    out = np.full(ps.shape, -1, dtype=np.int8)
    out[s & ~f] = 1
    out[f & ~s] = 0
    return out


def counts_from_per_point(path, K: int = 100) -> dict:
    """Counts from a stored ``full_roa_per_point.npz`` (p_* times K; the five runs used K = 100)."""
    with np.load(path, allow_pickle=True) as z:
        out = {f"n_{k}": np.rint(np.asarray(z[f"p_{k}"], float) * K).astype(np.int32)
               for k in ("success", "failure", "invalid")}
    if not np.all(out["n_success"] + out["n_failure"] + out["n_invalid"] == K):
        raise ValueError(f"{path}: p_* x {K} do not sum to {K}; wrong K?")
    return dict(**out, K=K)
