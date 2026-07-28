from __future__ import annotations

import glob
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adaptive_roa.adaptive.data_source import load_eval_states
from adaptive_roa.flow_matching.base.checkpoint_utils import load_hydra_config
from .registry import get_probabilistic_classifier_class
# Import wrappers for registration side-effects.
from . import classifier as _clf  # noqa: F401
from . import flow_matching as _fm  # noqa: F401

_DATASET_KIND = {"classifier": "classification", "generative": "endpoint"}


def resolve_predictor_family(cfg, run_dir=None) -> str:
    """Family tag: drives which dataset files a run wrote.

    ``run_dir``, when given, is folded into the error messages so a failure
    identifies which run's config was unreadable.
    """
    loc = f" at {run_dir}" if run_dir is not None else ""
    predictor = cfg.get("predictor", None)
    if predictor is None:
        raise ValueError(
            f"run config{loc} has no 'predictor' entry; refusing to guess "
            f"the export type (classifier vs generative)."
        )
    if isinstance(predictor, str):
        return predictor
    family = predictor.get("type", None)
    if family is None:
        raise ValueError(
            f"run config predictor block{loc} has no 'type'; "
            f"cannot determine the export type."
        )
    return str(family)


def resolve_predictor_name(cfg, run_dir=None) -> str:
    """Arm name, falling back to the family tag for pre-``name`` runs."""
    predictor = cfg.get("predictor", None)
    if predictor is None or isinstance(predictor, str):
        return resolve_predictor_family(cfg, run_dir)
    return str(predictor.get("name", None) or resolve_predictor_family(cfg, run_dir))


def _read_ws(path):
    return pd.read_csv(path, header=None, sep=r"\s+").to_numpy()


def resolve_system(cfg):
    target = cfg.get("system", {}).get("_target_", "")
    mod, name = target.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)()


def load_cfg(run_dir):
    for cand in [run_dir, str(Path(run_dir) / "epoch_000")]:
        cfg = load_hydra_config(Path(cand))
        if cfg is not None:
            return cfg
    raise FileNotFoundError(f"No hydra config for {run_dir}")


def epoch_dirs(run_dir):
    dirs = [d for d in glob.glob(f"{run_dir}/epoch_*") if Path(d).is_dir()]
    return sorted(dirs, key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))


def epoch_num(d):
    return int(re.search(r"epoch_(\d+)", d).group(1))


def load_split_states(run_dir, split, predictor_type, system, cfg):
    """Return (query_state[N,D], gt_label[N] or None, end_state[N,D] or None).

    gt_label is None whenever it must be derived from endpoints at the eval
    radius (all FM splits). In that case end_state carries the endpoints for
    cal/test; for FM train/val end_state is None and the endpoints are re-read
    from the run dataset file later. For the classifier predictor the ground
    truth is radius-independent, so gt_label is returned directly.
    """
    sd = int(system.state_dim)
    if split in ("cal", "test"):
        key = "cal_set_file" if split == "cal" else "test_set_file"
        path = str(cfg["data_source"][key])
        states, end, labels = load_eval_states(path)
        if predictor_type == "classifier":
            return states, labels, None
        # FM: reclassify endpoints at the eval radius so cal/test share the same
        # radius-dependent basis as train/val (and as the MC probabilities).
        return states, None, end.astype(np.float32)
    # train / val come from the run-level dataset files
    kind = _DATASET_KIND[predictor_type]
    path = str(Path(run_dir) / "datasets" / f"{split}_{kind}_dataset.txt")
    data = _read_ws(path)
    states = data[:, :sd].astype(np.float32)
    if predictor_type == "classifier":
        labels = np.where(data[:, -1] > 0.5, 1, -1).astype(np.int64)
        return states, labels, None
    # FM: derive labels from endpoint columns once radius is known (caller fills)
    return states, None, None


def _labels_from_endpoints(end, system, radius):
    """Classify endpoints at the eval radius -> {1: success, -1: failure, 0: invalid}."""
    return system.classify_attractor(
        torch.as_tensor(np.asarray(end), dtype=torch.float32), radius=radius
    ).cpu().numpy().astype(np.int64)


def _fm_labels_from_endpoints(run_dir, split, system, radius):
    sd = int(system.state_dim)
    kind = _DATASET_KIND["generative"]
    path = str(Path(run_dir) / "datasets" / f"{split}_{kind}_dataset.txt")
    end = _read_ws(path)[:, sd:2 * sd].astype(np.float32)
    return _labels_from_endpoints(end, system, radius)


def write_split(out_dir, split, query_state, gt_label, probs, native_probs):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    d = {
        "query_state": np.asarray(query_state, dtype=np.float32),
        "gt_label": np.asarray(gt_label, dtype=np.int64),
    }
    for name in native_probs:
        d[name] = np.asarray(getattr(probs, name), dtype=np.float64)
    np.savez_compressed(Path(out_dir) / f"{split}.npz", **d)
    return len(gt_label)


def export_run(run_dir, out_dir, device="cuda", epochs=None):
    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    cfg = load_cfg(run_dir)
    predictor_family = resolve_predictor_family(cfg, run_dir)
    predictor_name = resolve_predictor_name(cfg, run_dir)
    system = resolve_system(cfg)
    pc_class = get_probabilistic_classifier_class(predictor_name)
    native = pc_class.native_probs

    splits = ["train", "val", "cal", "test"]
    split_states = {}
    split_labels = {}
    split_ends = {}
    for split in splits:
        try:
            states, labels, ends = load_split_states(run_dir, split, predictor_family, system, cfg)
            split_states[split] = states
            split_labels[split] = labels
            split_ends[split] = ends
        except (FileNotFoundError, KeyError, OSError) as e:
            print(f"[skip split {split}] {type(e).__name__}: {e}", flush=True)

    eds = epoch_dirs(run_dir)
    if epochs is not None:
        eds = [d for d in eds if epoch_num(d) in set(epochs)]

    counts = {}
    for ed in eds:
        ep = epoch_num(ed)
        odir = Path(out_dir) / f"epoch_{ep:03d}"
        try:
            pc = pc_class.load_from_run(run_dir, ep, cfg, system, device)
        except (FileNotFoundError, RuntimeError, IndexError) as e:
            print(f"[skip epoch {ep:03d}] load failed: {type(e).__name__}: {e}", flush=True)
            counts[ep] = {"error": str(e)}
            continue
        c = {}
        for split in split_states:
            try:
                states = split_states[split]
                labels = split_labels[split]
                if labels is None:  # FM: derive at the current eval radius
                    ends = split_ends.get(split)
                    if ends is not None:  # cal/test endpoints already in hand
                        labels = _labels_from_endpoints(ends, system, pc.attractor_radius)
                    else:  # train/val: endpoints live in the run dataset file
                        labels = _fm_labels_from_endpoints(
                            run_dir, split, system, pc.attractor_radius
                        )
                probs = pc.predict_cached(run_dir, ep, split, states)
                if probs is None:
                    probs = pc.predict(states)
                c[split] = write_split(odir, split, states, labels, probs, native)
            except Exception as e:  # never abort the whole run on one split
                print(f"[epoch {ep:03d} split {split}] ERROR {type(e).__name__}: {e}", flush=True)
                c[split] = {"error": str(e)}
        counts[ep] = c
        print(f"epoch {ep:03d}: {c}", flush=True)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    meta = {
        "run_dir": str(run_dir),
        "predictor": predictor_name,
        "native_probs": list(native),
        "gt_label_convention": (
            {"1": "success", "-1": "failure",
             "basis": "classifier binary ground truth (radius-independent)"}
            if predictor_family == "classifier"
            else {"1": "success", "-1": "failure", "0": "invalid/separatrix",
                  "basis": "system.classify_attractor(endpoint, eval radius) for ALL splits "
                           "(train/val/cal/test), matching the MC probability basis"}
        ),
        "prob_definitions": (
            "classifier: p_success = sigmoid(logit)"
            if predictor_family == "classifier"
            else "FM: p_{success,failure,invalid} = fraction of MC endpoints with mc_label 1 / -1 / 0"
        ),
        "epoch_counts": counts,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "metadata.json").write_text(json.dumps(meta, indent=2))
    return counts
