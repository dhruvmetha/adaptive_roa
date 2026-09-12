#!/usr/bin/env python
"""Mechanism figure: the TRUE success probability of what each acquisition bought.

For every adaptive epoch, `epoch_XXX/artifacts_v2.json` records the pool indices
the run acquired: `acquisition.d1_indices` (uniform draws, the non-adaptive arm)
and `acquisition.d2_indices` (scored picks, the BALD / variance arms). This
script maps each pool index to its rollout's start state and reports the
ground-truth success probability of the batch, two ways:

  (i)  the rollout's OWN label in train.npz (batch mean = realised success
       fraction of what was bought), and
  (ii) the continuous grid value from eval_success_prob.npz at the nearest grid
       start (batch mean / median, and the ambiguous share 0.2 <= p <= 0.8).

Pool index -> npz row follows NpzTrajectoryDataSource
(adaptive_roa/adaptive/npz_data_source.py): pool index k is npz row
rollout_ids[k], where rollout_ids is train_test_splits/shuffled_indices_<v>.txt
(bare integers on pendulum, `sequence_<row>.txt` filenames on cartpole and the
quadrotors). The start state of rollout r is states[offsets[r]], NOT starts[r]:
on quadrotor3D the two disagree (the data source says why).

No model inference. train.npz members are memory-mapped at their offset inside
the zip (they are ZIP_STORED), so only the rows actually acquired are read;
np.load(..., mmap_mode="r") is a full read for npz members in numpy 2.x.

Usage:
    python scripts/paper/acquired_true_p.py            # validate, then everything
    python scripts/paper/acquired_true_p.py --validate-only
    python scripts/paper/acquired_true_p.py --campaigns pend_low cprl_med --jobs 4
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import struct
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SCORER = ROOT / "scripts/score_stoch_incremental.py"
STYLER = ROOT / "scripts/paper/plot_levelsets_paper.py"

# =============================================================================
# STYLE -- every design choice is here. Colours, labels, level titles and system
# titles come from plot_levelsets_paper.py so this figure matches the others.
# =============================================================================

# Campaign -> system (paper name). Anything not listed here is out of scope.
CAMPAIGN_SYSTEM = {
    "pend_low": "pendulum_lqr", "pend_med": "pendulum_lqr", "pend_high": "pendulum_lqr",
    "cprl_base": "cartpole_ppo", "cprl_low": "cartpole_ppo",
    "cprl_med": "cartpole_ppo", "cprl_high": "cartpole_ppo",
    "q2d_csbase": "quad2d_rl", "q2d_cs": "quad2d_rl", "q2d_csloud": "quad2d_rl",
    # quad3d = 800k-pool campaign (decision 2026-09-10); 100k is obsolete
    "q3d800k_cs000": "quad3d_ppo800k", "q3d800k_cs012": "quad3d_ppo800k",
    "q3d800k_cs020": "quad3d_ppo800k", "q3d800k_cs040": "quad3d_ppo800k",
    "q3damb_f012a03": "quad3d_ppo800k", "q3damb_f020a04": "quad3d_ppo800k", "q3damb_f040a04": "quad3d_ppo800k",
}

# The scorer's campaign table points the q3d800k campaigns at the 100k `ppo/`
# data tree (the eval_success_prob.npz grids are byte-identical on both trees,
# so scoring is unaffected). The pool indices in artifacts_v2.json index the
# 800k tree's split (train_test_splits/shuffled_indices_0.txt, 800,000 rows;
# the runs' .hydra/config.yaml say controller=ppo_800k, shuffle_variant=0),
# so the dataset root is overridden here for those campaigns.
_Q3D800K_DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic/"
                     "quadrotor3D/corridor_sine_ambient/ppo_800k")
DATASET_ROOT_OVERRIDE = {
    "q3d800k_cs000": _Q3D800K_DATA / "f_0.00",
    "q3d800k_cs012": _Q3D800K_DATA / "f_0.12",
    "q3d800k_cs020": _Q3D800K_DATA / "f_0.20",
    "q3d800k_cs040": _Q3D800K_DATA / "f_0.40",
    "q3damb_f012a03": _Q3D800K_DATA / "f_0.12_a0.03",
    "q3damb_f020a04": _Q3D800K_DATA / "f_0.20_a0.04",
    "q3damb_f040a04": _Q3D800K_DATA / "f_0.40_a0.04",
}

# Run-dir name overrides, (campaign, arm) -> directory arm suffix. On q2d_csbase
# the LIVE FM runs are the `_fast` directories; the bare-name ones are cancelled
# 2-3 epoch twins (see plot_levelsets_paper.ARM_ALIAS). Nothing is renamed on disk.
RUN_DIR_ARM = {
    ("q2d_csbase", "dir00_s42"): "dir00_s42_fast",
    ("q2d_csbase", "epi_bald_greedy"): "epi_bald_greedy_fast",
    ("q2d_csbase", "epi_var_greedy"): "epi_var_greedy_fast",
}
# Campaign -> last epoch kept (budget cap: q2d_csbase stopped at 8,000
# trajectories = epoch 12; some arms have epoch dirs past it).
EPOCH_CAP = {"q2d_csbase": 12}

# Arm -> which index list the arm fills. The non-adaptive arm draws d1 only
# (d2_ratio=0); the scored arms draw d2 only (d2_ratio=1).
# BALD arms are the top-N (greedy) runs (decision 2026-09-10); greedy_diverse is out of the paper
ARM_INDEX_KEY = {
    "dir00_s42": "d1_indices",
    "epi_bald_greedy": "d2_indices",
    "epi_var_greedy": "d2_indices",
    "clf_epi_bald_greedy": "d2_indices",
    "bnn_mfvi_a1_greedy": "d2_indices",
}
ARM_ORDER = ["dir00_s42", "epi_bald_greedy", "epi_var_greedy", "clf_epi_bald_greedy", "bnn_mfvi_a1_greedy"]
OURS = "epi_bald_greedy"

# Rows: CSV column -> y label.
ROWS = [
    ("grid_p_mean", "mean true p(success)\nof acquired batch"),
    ("frac_ambiguous", "ambiguous share\n(0.2 ≤ p ≤ 0.8)"),
]
AMBIG_LO, AMBIG_HI = 0.2, 0.8
P_LOW, P_HIGH = 0.05, 0.95

# Dashed grey reference per row: the pool's own value of the same statistic
# (label mean for row 1, grid ambiguous share for row 2). Uniform sampling
# should sit on it; an acquisition that moves off it is the mechanism.
DRAW_POOL_REF = True
POOL_REF_STYLE = dict(color="0.45", ls="--", lw=0.9)
POOL_REF_LABEL = "pool average"

X_LABEL = "epoch"
# y range is shared across a row's panels. "auto": top = the row's max value
# (lines and pool reference) padded by Y_PAD and rounded up to Y_STEP, capped at
# 1; "fixed": always [0, 1]. Quadrotor 3D lives below 0.5 on both rows and is
# unreadable on a fixed axis.
Y_MODE = "auto"
Y_PAD = 1.12
Y_STEP = 0.1
Y_TICK_BINS = 5
LINE_ALPHA = 0.95

# Figure geometry (inches) and typography.
PANEL_W, PANEL_H = 2.6, 1.9
FIG_MIN_W = 7.0
TITLE_BLOCK_H = 0.55
XLABEL_BLOCK_H = 0.45
LEGEND_ROW_H = 0.26
LEGEND_MAX_W_PER_ENTRY = 1.3
LEFT_MARGIN, RIGHT_MARGIN = 0.7, 0.08
SUBPLOT_WSPACE, SUBPLOT_HSPACE = 0.12, 0.25
DPI = 200
FONT_FAMILY = "sans-serif"
FONT_SIZE = 8
TITLE_SIZE = 9
SUPTITLE_SIZE = 10
LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7.5
GRID = dict(color="0.85", lw=0.5)
SPINES_OFF = ("top", "right")
X_TICK_MAX = 8            # at most this many labelled epoch ticks per panel

# Output.
PAPER_DIR = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic/paper_draft")
TABLE_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"
CSV_OUT = TABLE_DIR / "acquired_true_p.csv"
MD_OUT = TABLE_DIR / "acquired_true_p.md"
FILE_MODE = 0o660
DIR_MODE = 0o770
FORMATS = ("png",)   # png only (2026-09-11)

CSV_FIELDS = ["system", "level", "campaign", "arm", "epoch", "n_acquired",
              "train_trajectories", "label_success_frac", "grid_p_mean",
              "grid_p_median", "frac_ambiguous", "frac_p_below_0.05",
              "frac_p_above_0.95", "nn_dist_median", "nn_dist_max"]

# Validation target: campaign, arm, epoch.
VALIDATE = ("pend_low", "epi_bald_greedy", 3)

# =============================================================================
# Plumbing
# =============================================================================


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SIB: dict = {}


def siblings() -> tuple:
    """(scorer, styler) modules, loaded once per process."""
    if not _SIB:
        _SIB["scorer"] = _load_module(SCORER, "score_stoch_incremental")
        _SIB["styler"] = _load_module(STYLER, "plot_levelsets_paper")
    return _SIB["scorer"], _SIB["styler"]


def run_dir(campaign: str) -> tuple[Path, Path]:
    """(dataset root, experiment root) for a campaign, from the scorer's table."""
    scorer, _ = siblings()
    c = scorer.CAMPAIGNS[campaign]
    root = DATASET_ROOT_OVERRIDE.get(campaign, c["root"] / c["level"])
    return root, (c.get("exp") or scorer.EXP)


def arm_dir(campaign: str, arm: str) -> Path:
    scorer, _ = siblings()
    c = scorer.CAMPAIGNS[campaign]
    return (c.get("exp") or scorer.EXP) / f"{c['prefix']}_{RUN_DIR_ARM.get((campaign, arm), arm)}"


def level_key(campaign: str) -> str:
    scorer, _ = siblings()
    return scorer.CAMPAIGNS[campaign]["key"]


# ---- npz access -------------------------------------------------------------

_NPY_HEADER_READERS = {
    (1, 0): np.lib.format.read_array_header_1_0,
    (2, 0): np.lib.format.read_array_header_2_0,
}


def npz_member(path: Path, key: str) -> np.ndarray:
    """One array of an .npz, memory-mapped in place when the member is stored
    uncompressed (np.savez). Falls back to a full read for a deflated member.

    np.load(path, mmap_mode="r")[key] reads the whole member on numpy 2.x, which
    is 2.2 GB for quadrotor3D's `states`; this touches only the rows indexed.
    """
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            with zf.open(info) as fh:
                return np.lib.format.read_array(fh)
    with open(path, "rb") as fh:
        # Local file header: 30 fixed bytes, then name and extra. The local
        # extra field can differ in length from the central directory's, so
        # the data offset has to come from the local header.
        fh.seek(info.header_offset)
        hdr = fh.read(30)
        (sig,) = struct.unpack("<I", hdr[:4])
        if sig != 0x04034B50:
            raise ValueError(f"{path}:{key}: bad local file header")
        n_name, n_extra = struct.unpack("<HH", hdr[26:30])
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(fh)
        reader = _NPY_HEADER_READERS.get(version)
        if reader is None:
            shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
        else:
            shape, fortran, dtype = reader(fh)
        arr_off = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=arr_off, shape=shape,
                     order="F" if fortran else "C")


def parse_rollout_ids(filepath: Path) -> np.ndarray:
    """Same rule as NpzTrajectoryDataSource._parse_rollout_ids: a bare integer,
    or the first integer in a name like `sequence_<row>.txt`."""
    ids = []
    with open(filepath) as fh:
        for line in fh:
            tok = line.strip()
            if not tok:
                continue
            try:
                ids.append(int(tok))
            except ValueError:
                m = re.search(r"(\d+)", tok)
                if m is None:
                    raise ValueError(f"{filepath}: cannot read a rollout id from {tok!r}")
                ids.append(int(m.group(1)))
    if not ids:
        raise ValueError(f"{filepath}: no rollout ids found")
    return np.asarray(ids, dtype=np.int64)


def labels_to_01(lab: np.ndarray) -> np.ndarray:
    """{-1,+1} -> {0,1}; {0,1} kept. Anything else is an error. Never
    .astype(bool): that maps -1 to True."""
    lab = np.asarray(lab).astype(np.int64)
    u = set(np.unique(lab).tolist())
    if u <= {-1, 1}:
        return (lab > 0).astype(np.int64)
    if u <= {0, 1}:
        return lab
    raise ValueError(f"labels take values {sorted(u)}, expected {{-1,+1}} or {{0,1}}")


class Pool:
    """train.npz + the split's rollout ids + the eval grid, for one dataset root."""

    def __init__(self, root: Path, variant: int):
        self.root = root
        npz = root / "train.npz"
        self.states = npz_member(npz, "states")            # memmap (sumT, D)
        self.offsets = np.asarray(npz_member(npz, "offsets")).astype(np.int64)
        self.labels = labels_to_01(np.asarray(npz_member(npz, "labels")))
        self.raw_label_values = sorted(np.unique(np.asarray(npz_member(npz, "labels"))).tolist())
        self.starts = npz_member(npz, "starts")            # memmap, only for the note
        self.rollout_ids = parse_rollout_ids(
            root / "train_test_splits" / f"shuffled_indices_{variant}.txt")
        n = len(self.offsets) - 1
        if self.rollout_ids.min() < 0 or self.rollout_ids.max() >= n:
            raise ValueError(f"{root}: shuffled ids out of range for {n} rollouts")
        self.pool_success_frac = float(self.labels[self.rollout_ids].mean())
        with np.load(root / "eval_success_prob.npz") as z:
            self.gt_starts = z["starts"].astype(np.float64)
            self.gt_p = z["p_success"].astype(np.float64)
        if self.gt_starts.shape[1] != self.states.shape[1]:
            raise ValueError(f"{root}: grid is {self.gt_starts.shape[1]}-D, "
                             f"states are {self.states.shape[1]}-D")
        self.tree = cKDTree(self.gt_starts)
        self.grid_p_mean = float(self.gt_p.mean())
        self.grid_frac_ambiguous = float(
            ((self.gt_p >= AMBIG_LO) & (self.gt_p <= AMBIG_HI)).mean())

    def start_states(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.states[self.offsets[rows]]).astype(np.float64)

    def batch(self, pool_idx: np.ndarray) -> dict:
        """Truth statistics of one acquired batch given its pool indices."""
        pool_idx = np.asarray(pool_idx, dtype=np.int64)
        if pool_idx.min() < 0 or pool_idx.max() >= len(self.rollout_ids):
            raise ValueError("pool index out of range")
        rows = self.rollout_ids[pool_idx]
        x = self.start_states(rows)
        lab = self.labels[rows]
        d, gi = self.tree.query(x, k=1, workers=1)
        p = self.gt_p[gi]
        starts_diff = float("nan")
        if self.starts.shape[1] == x.shape[1]:
            starts_diff = float(np.abs(np.asarray(self.starts[rows]).astype(np.float64) - x).max())
        return dict(
            n_acquired=int(len(pool_idx)),
            label_success_frac=float(lab.mean()),
            grid_p_mean=float(p.mean()),
            grid_p_median=float(np.median(p)),
            frac_ambiguous=float(((p >= AMBIG_LO) & (p <= AMBIG_HI)).mean()),
            **{"frac_p_below_0.05": float((p < P_LOW).mean()),
               "frac_p_above_0.95": float((p > P_HIGH).mean())},
            nn_dist_median=float(np.median(d)),
            nn_dist_max=float(d.max()),
            _starts_vs_states_maxdiff=starts_diff,
            _rows=rows,
        )


_POOLS: dict = {}


def get_pool(root: Path, variant: int) -> Pool:
    key = (str(root), variant)
    if key not in _POOLS:
        _POOLS[key] = Pool(root, variant)
    return _POOLS[key]


# ---- run access -------------------------------------------------------------

def hydra_cfg(rd: Path) -> dict:
    p = rd / ".hydra/config.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def shuffle_variant(rd: Path) -> int:
    return int(hydra_cfg(rd).get("shuffle_variant", 0))


def epoch_artifacts(rd: Path) -> list[tuple[Path, dict]]:
    out = []
    for d in sorted(rd.glob("epoch_*")):
        a = d / "artifacts_v2.json"
        if not a.exists():
            continue
        try:
            out.append((d, json.loads(a.read_text())))
        except ValueError:
            print(f"  [warn] unreadable {a}", file=sys.stderr)
    return out


def process(task: tuple[str, str]) -> dict:
    """One (campaign, arm): a row per epoch plus the pool reference values."""
    campaign, arm = task
    root, _ = run_dir(campaign)
    rd = arm_dir(campaign, arm)
    key = ARM_INDEX_KEY[arm]
    res = dict(campaign=campaign, arm=arm, rows=[], skipped=[], ref=None, error=None)
    if not rd.is_dir():
        res["error"] = f"missing run dir {rd}"
        return res
    try:
        pool = get_pool(root, shuffle_variant(rd))
        res["ref"] = dict(pool_success_frac=pool.pool_success_frac,
                          grid_p_mean=pool.grid_p_mean,
                          grid_frac_ambiguous=pool.grid_frac_ambiguous,
                          raw_label_values=pool.raw_label_values,
                          n_pool=int(len(pool.rollout_ids)))
        for d, art in epoch_artifacts(rd):
            if int(art.get("epoch", int(d.name.split("_")[1]))) > EPOCH_CAP.get(campaign, 1 << 30):
                continue
            idx = (art.get("acquisition") or {}).get(key) or []
            if not idx:
                res["skipped"].append(f"{d.name}: no {key}")
                continue
            b = pool.batch(np.asarray(idx))
            row = dict(system=CAMPAIGN_SYSTEM[campaign], level=level_key(campaign),
                       campaign=campaign, arm=arm, epoch=int(art.get("epoch", int(d.name.split("_")[1]))),
                       train_trajectories=int(art.get("train_trajectories", -1)))
            row.update({k: v for k, v in b.items() if not k.startswith("_")})
            res["rows"].append(row)
            res["ref"]["starts_vs_states_maxdiff"] = max(
                res["ref"].get("starts_vs_states_maxdiff", 0.0),
                b["_starts_vs_states_maxdiff"] if not math.isnan(b["_starts_vs_states_maxdiff"]) else 0.0)
    except Exception as e:  # keep the fleet going; report at the end
        res["error"] = f"{type(e).__name__}: {e}"
    return res


# ---- validation -------------------------------------------------------------

def validate() -> None:
    campaign, arm, ep = VALIDATE
    root, _ = run_dir(campaign)
    rd = arm_dir(campaign, arm)
    cfg = hydra_cfg(rd)
    variant = int(cfg.get("shuffle_variant", 0))
    spe = int(cfg.get("samples_per_epoch", -1))
    pool = get_pool(root, variant)
    art = json.loads((rd / f"epoch_{ep:03d}" / "artifacts_v2.json").read_text())
    idx = np.asarray(art["acquisition"][ARM_INDEX_KEY[arm]], dtype=np.int64)
    b = pool.batch(idx)
    rows = b["_rows"]
    print(f"VALIDATE {campaign}/{arm} epoch {art['epoch']} (train_trajectories={art['train_trajectories']}, "
          f"shuffle_variant={variant})")
    print(f"  n_indices={len(idx)} samples_per_epoch={spe} "
          f"{'OK' if len(idx) == spe else 'MISMATCH'}; pool index range [{idx.min()}, {idx.max()}] "
          f"of {len(pool.rollout_ids)}")
    print(f"  npz label values on disk: {pool.raw_label_values} "
          f"(mapped to {{0,1}}; the data source maps 0 -> -1 internally)")
    print(f"  label_success_frac={b['label_success_frac']:.4f}  grid_p_mean={b['grid_p_mean']:.4f}  "
          f"grid_p_median={b['grid_p_median']:.4f}  frac_ambiguous={b['frac_ambiguous']:.4f}")
    print(f"  nn_dist median={b['nn_dist_median']:.3g} max={b['nn_dist_max']:.3g}  "
          f"(grid spacing: theta {np.diff(np.unique(pool.gt_starts[:, 0]))[:1]}, "
          f"theta_dot {np.diff(np.unique(pool.gt_starts[:, 1]))[:1]})")
    print(f"  pool: success_frac={pool.pool_success_frac:.4f} grid_p_mean={pool.grid_p_mean:.4f} "
          f"grid_frac_ambiguous={pool.grid_frac_ambiguous:.4f}")

    # (a) The split's own identity: shuffled_labels[i] == labels[rollout_ids[i]].
    sl = labels_to_01(np.loadtxt(root / "train_test_splits" / f"shuffled_labels_{variant}.txt", dtype=int))
    agree = float((sl == pool.labels[pool.rollout_ids]).mean())
    print(f"  shuffled_labels[i] == labels[rollout_ids[i]]: {agree:.6f} "
          f"{'OK' if agree == 1.0 else 'BROKEN MAPPING'}")
    # (b) The run's own dataset listing names trajectories by npz row
    #     (NpzTrajectoryDataSource.trajectory_name), so every acquired row must
    #     be in train_trajectories.txt or val_trajectories.txt.
    names = set()
    for f in ("train_trajectories.txt", "val_trajectories.txt"):
        p = rd / "datasets" / f
        if p.exists():
            names |= {int(t) for t in p.read_text().split()}
    if names:
        hit = float(np.mean([int(r) in names for r in rows]))
        print(f"  acquired rows present in the run's datasets/*_trajectories.txt: {hit:.4f} "
              f"{'OK' if hit == 1.0 else 'BROKEN MAPPING'} ({len(names)} names listed)")
    # (c) The zip-offset memmap returns the same rows as a plain read.
    with np.load(root / "train.npz") as z:
        full = z["states"]
        ref = full[pool.offsets[rows]]
    same = np.array_equal(ref, np.asarray(pool.states[pool.offsets[rows]]))
    print(f"  memmap rows == np.load rows: {same}  (states {pool.states.shape}, memmap={isinstance(pool.states, np.memmap)})")
    # (d) sanity on the label/grid agreement: grid p at a rollout's start should
    #     predict its own label better than chance.
    p = pool.gt_p[pool.tree.query(pool.start_states(rows), k=1)[1]]
    lab = pool.labels[rows]
    print(f"  mean grid p | label=1: {p[lab == 1].mean() if (lab == 1).any() else float('nan'):.3f}   "
          f"| label=0: {p[lab == 0].mean() if (lab == 0).any() else float('nan'):.3f}")
    print(f"  {arm} diagnostics: {json.dumps(art['acquisition'].get('diagnostics', {}))[:200]}")
    if len(idx) != spe or agree != 1.0 or (names and hit != 1.0) or not same:
        raise SystemExit("validation failed; not running the fleet")


# ---- outputs ----------------------------------------------------------------

def _chmod(p: Path, mode: int = FILE_MODE) -> None:
    try:
        os.chmod(p, mode)
    except OSError as e:
        print(f"  [warn] chmod {p}: {e}", file=sys.stderr)


def _mkdir(p: Path) -> None:
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        _chmod(p, DIR_MODE)


def write_csv(rows: list[dict]) -> Path:
    _mkdir(TABLE_DIR)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    rows = sorted(rows, key=lambda r: (r["system"], r["level"], order.get(r["arm"], 99), r["epoch"]))
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    _chmod(CSV_OUT)
    return CSV_OUT


def _levels_in_order(levels: list[str]) -> list[str]:
    _, st = siblings()
    return [l for l in st.LEVEL_ORDER if l in levels] + sorted(l for l in levels if l not in st.LEVEL_ORDER)


def write_md(rows: list[dict], refs: dict) -> Path:
    _, st = siblings()
    by = defaultdict(list)
    for r in rows:
        by[(r["system"], r["level"], r["arm"])].append(r)
    lines = ["# True success probability of acquired trajectories", "",
             "Per (system, level, arm): mean over epochs of the acquired batch's grid "
             "p(success) mean and its ambiguous share (0.2 ≤ p ≤ 0.8), plus the realised "
             "label fraction. `pool` rows give the whole trainable pool's label success "
             "fraction and the eval grid's mean p / ambiguous share for reference.",
             "", f"Source: `{CSV_OUT}`", ""]
    for system in st.SYSTEM_TITLES:
        levels = sorted({r["level"] for r in rows if r["system"] == system})
        if not levels:
            continue
        lines += [f"## {st.SYSTEM_TITLES[system]}", "",
                  "| level | arm | epochs | grid p mean | ambiguous share | label success frac | NN dist median (max) |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for level in _levels_in_order(levels):
            ref = refs.get((system, level))
            if ref:
                lines.append(f"| {st.level_title(level)} | pool ({ref['n_pool']:,} rollouts) | | "
                             f"{ref['grid_p_mean']:.3f} | {ref['grid_frac_ambiguous']:.3f} | "
                             f"{ref['pool_success_frac']:.3f} | |")
            for arm in ARM_ORDER:
                rs = by.get((system, level, arm))
                if not rs:
                    continue
                label = st.SEED_LABEL if arm == "dir00_s42" else st.ARM_STYLES[arm][0]
                m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
                lines.append(f"| {st.level_title(level)} | {label} | {len(rs)} | {m('grid_p_mean'):.3f} | "
                             f"{m('frac_ambiguous'):.3f} | {m('label_success_frac'):.3f} | "
                             f"{np.median([r['nn_dist_median'] for r in rs]):.3g} "
                             f"({max(r['nn_dist_max'] for r in rs):.3g}) |")
        lines.append("")
    MD_OUT.write_text("\n".join(lines))
    _chmod(MD_OUT)
    return MD_OUT


def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def render(system: str, rows: list[dict], refs: dict) -> list[Path]:
    _, st = siblings()
    _apply_rc()
    levels = _levels_in_order(sorted({r["level"] for r in rows}))
    by = defaultdict(list)
    for r in rows:
        by[(r["level"], r["arm"])].append(r)

    n_rows, n_cols = len(ROWS), len(levels)
    block_w = PANEL_W * n_cols
    fig_w = max(block_w + LEFT_MARGIN + RIGHT_MARGIN, FIG_MIN_W)
    n_entries = len(ARM_ORDER) + int(DRAW_POOL_REF)
    legend_ncol = max(1, min(n_entries, int(fig_w / LEGEND_MAX_W_PER_ENTRY)))
    legend_rows = math.ceil(n_entries / legend_ncol)
    legend_ncol = math.ceil(n_entries / legend_rows)   # balance a wrapped legend
    legend_h = LEGEND_ROW_H * legend_rows
    fig_h = PANEL_H * n_rows + TITLE_BLOCK_H + XLABEL_BLOCK_H + legend_h
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(fig_w, fig_h))
    handles: dict[str, object] = {}

    # One y range per row, shared by every level of the system.
    y_top: dict[str, float] = {}
    for col, _ in ROWS:
        if Y_MODE == "fixed":
            y_top[col] = 1.0
            continue
        vals = [r[col] for r in rows]
        if DRAW_POOL_REF:
            vals += [ref["pool_success_frac"] if col == "grid_p_mean" else ref["grid_frac_ambiguous"]
                     for ref in (refs.get((system, l)) for l in levels) if ref]
        vmax = max(vals) if vals else 1.0
        y_top[col] = min(1.0, math.ceil(vmax * Y_PAD / Y_STEP - 1e-9) * Y_STEP)

    for j, level in enumerate(levels):
        ref = refs.get((system, level))
        ep_max = max(r["epoch"] for r in rows if r["level"] == level)
        for i, (col, ylab) in enumerate(ROWS):
            ax = axes[i][j]
            ax.grid(True, **GRID)
            for sp in SPINES_OFF:
                ax.spines[sp].set_visible(False)
            if DRAW_POOL_REF and ref:
                y = ref["pool_success_frac"] if col == "grid_p_mean" else ref["grid_frac_ambiguous"]
                ax.axhline(y, zorder=1, **POOL_REF_STYLE)
                handles.setdefault(POOL_REF_LABEL, Line2D([], [], **POOL_REF_STYLE))
            for arm in ARM_ORDER:
                rs = sorted(by.get((level, arm), []), key=lambda r: r["epoch"])
                if not rs:
                    continue
                xs = [r["epoch"] for r in rs]
                ys = [r[col] for r in rs]
                if arm == "dir00_s42":
                    ax.plot(xs, ys, color=st.SEED_MEAN_COLOR, lw=st.SEED_MEAN_LW, zorder=2)
                    handles.setdefault(st.SEED_LABEL, Line2D(
                        [], [], color=st.SEED_MEAN_COLOR, lw=st.SEED_MEAN_LW))
                else:
                    label, fam, color, lw, mk = st.ARM_STYLES[arm]
                    ax.plot(xs, ys, ls=st.FAMILY_LS[fam], color=color, lw=lw, marker=mk,
                            ms=st.MARKER_SIZE, alpha=LINE_ALPHA,
                            zorder=4 if arm == OURS else 3)
                    handles.setdefault(label, Line2D(
                        [], [], ls=st.FAMILY_LS[fam], color=color, lw=lw, marker=mk,
                        ms=st.MARKER_SIZE))
            step = max(1, math.ceil((ep_max + 1) / X_TICK_MAX))
            ax.set_xlim(-0.5, ep_max + 0.5)
            ax.set_xticks(list(range(0, ep_max + 1, step)))
            top = y_top[col]
            ax.set_ylim(-0.02 * top, top * 1.02)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_TICK_BINS, steps=[1, 2, 2.5, 5, 10]))
            if i == 0:
                ax.set_title(st.level_title(level))
            if i == n_rows - 1:
                ax.set_xlabel(X_LABEL)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.tick_params(labelleft=False)

    order = [st.SEED_LABEL] + [st.ARM_STYLES[a][0] for a in ARM_ORDER if a != "dir00_s42"] + [POOL_REF_LABEL]
    keys = [k for k in order if k in handles]
    fig.legend([handles[k] for k in keys], keys, loc="lower center",
               ncol=min(legend_ncol, len(keys)), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
    fig.suptitle(st.SYSTEM_TITLES.get(system, system), fontsize=SUPTITLE_SIZE, y=1 - 0.12 / fig_h)
    left_in = (fig_w - block_w - LEFT_MARGIN - RIGHT_MARGIN) / 2 + LEFT_MARGIN
    fig.subplots_adjust(left=left_in / fig_w, right=(left_in + block_w) / fig_w,
                        top=1 - TITLE_BLOCK_H / fig_h,
                        bottom=(legend_h + XLABEL_BLOCK_H) / fig_h,
                        wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE)
    _mkdir(FIG_DIR)
    outs = []
    for fmt in FORMATS:
        p = FIG_DIR / f"acquired_true_p_{system}.{fmt}"
        fig.savefig(p, dpi=DPI)
        _chmod(p)
        outs.append(p)
    plt.close(fig)
    return outs


# ---- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaigns", nargs="*", default=list(CAMPAIGN_SYSTEM),
                    help="subset of in-scope campaigns (default: all)")
    ap.add_argument("--arms", nargs="*", default=ARM_ORDER)
    ap.add_argument("--jobs", type=int, default=8, help="worker processes (max 8)")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    if not args.skip_validate:
        validate()
        if args.validate_only:
            return

    # Runs excluded from the paper (user instruction 2026-09-11): quad2d loud epi-var is in flight.
    EXCLUDE: set = set()   # quad2d loud epi-var finished 2026-09-11
    tasks = [(c, a) for c in args.campaigns for a in args.arms
             if c in CAMPAIGN_SYSTEM and a in ARM_INDEX_KEY and (c, a) not in EXCLUDE]
    jobs = max(1, min(8, args.jobs))
    print(f"\n{len(tasks)} (campaign, arm) tasks on {jobs} workers")
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            results = list(pool.imap_unordered(process, tasks))
    else:
        results = [process(t) for t in tasks]

    rows, refs, failed, skipped = [], {}, [], []
    for res in sorted(results, key=lambda r: (r["campaign"], r["arm"])):
        tag = f"{res['campaign']}/{res['arm']}"
        if res["error"]:
            failed.append(f"{tag}: {res['error']}")
            continue
        rows += res["rows"]
        skipped += [f"{tag} {s}" for s in res["skipped"]]
        key = (CAMPAIGN_SYSTEM[res["campaign"]], level_key(res["campaign"]))
        refs.setdefault(key, res["ref"])
        sd = res["ref"].get("starts_vs_states_maxdiff", 0.0)
        print(f"  {tag}: {len(res['rows'])} epochs"
              + (f", starts vs states[offsets] differ by up to {sd:.3g}" if sd > 1e-4 else ""))

    if not rows:
        raise SystemExit("no rows produced")
    print(f"\nwrote {write_csv(rows)} ({len(rows)} rows)")
    print(f"wrote {write_md(rows, refs)}")
    for system in dict.fromkeys(r["system"] for r in rows):
        outs = render(system, [r for r in rows if r["system"] == system], refs)
        print(f"wrote {', '.join(map(str, outs))}")
    for s in skipped:
        print(f"  skipped {s}")
    for f in failed:
        print(f"  FAILED {f}")


if __name__ == "__main__":
    main()
