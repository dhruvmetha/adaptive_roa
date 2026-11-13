"""Write slices/manifest.json for the picker page.

    $PY slices/lib/build_manifest.py [--out slices/manifest.json]

Scans data/<setting>/<plane>.npz and the full-resolution preds pred/<setting>/<plane>__r{1,2,3}.npz
(stride-2 ``*_s2`` files and ``.tmp.npz`` partials are ignored). One "done" entry per
(setting, plane, seed) with a pred; one "pending" GT-only entry per slice with fewer than 3 done
seeds. Metrics are the in-box ones of ``metrics.slice_metrics`` on the pred's decision; with Success
as the positive class n_FP = wrong_S (Success declared on a GT fail) and n_FN = wrong_F. NaN is
written as null. A pred's cached metrics are keyed on both the pred file and its data npz (mtime and
size), and a slice's GT metrics on the data npz, so rewriting a data file (e.g. a new box) refreshes
every entry of that slice. The output is written atomically (temp file in the same directory, then
os.replace). Metrics are cached in lib/_scratch/manifest_cache.json, keyed by each file's mtime and
size, so a rerun only loads new or changed files. No GPU, no detv3 import.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SLICES = Path(__file__).resolve().parents[1]
if str(SLICES) not in sys.path:
    sys.path.insert(0, str(SLICES))
from lib import metrics as M  # noqa: E402
from lib import style as S  # noqa: E402

CACHE = SLICES / "lib" / "_scratch" / "manifest_cache.json"
MATRIX_ROWS = Path("/common/users/shared/pracsys/adaptive_roa_experiments/det_v3_seeds/_index/scored/matrix_rows.jsonl")
PRED_RE = re.compile(r"^(?P<plane>.+)__(?P<rk>r[123])\.npz$")
SYSTEMS = {"cp": "Cartpole", "q2d": "Planar quadrotor", "q3d": "Spatial quadrotor"}
CACHE_VERSION = 1


def _clean(v):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _key(p: Path):
    st = p.stat()                       # follows symlinks: a re-pointed link shows up as a change
    return [st.st_mtime_ns, st.st_size]


def _fmt(v):
    return f"{v:.3g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _slice_info(path: Path) -> dict:
    """Everything the manifest needs from a slice, without model_states."""
    sl = S.load_slice(path)
    gt_m, _ = M.slice_metrics(sl)
    fixed = sl["fixed"]
    num = {k: v for k, v in fixed.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    other = {k: v for k, v in fixed.items() if k not in num}
    system = SYSTEMS.get(sl["setting"].split("-")[0], sl["system_label"])
    caveats = []
    if gt_m["in_box_frac"] < 1:
        caveats.append(f"{1 - gt_m['in_box_frac']:.0%} of the slice lies outside the D_test start box "
                       f"(FM extrapolation there; metrics use the in-box cells)")
    if len(np.unique(sl["labels"])) == 1:
        caveats.append("single class")
    for k, v in other.items():
        caveats.append(f"{k}: {v}")
    note = sl["source_note"].strip()
    if caveats:
        note = (note + " " if note else "") + "[" + "; ".join(caveats) + "]"
    return dict(setting=sl["setting"], system=system, controller=sl["controller"], plane=sl["plane"],
                dims=f"{sl['dim_names'][0]} vs {sl['dim_names'][1]}",
                fixed=", ".join(f"{k}={_fmt(v)}" for k, v in num.items()),
                source_note=note, gt_clarity=gt_m["gt_clarity"], n_in_box=int(gt_m["n_cells"]),
                labels_shape=list(sl["labels"].shape), gt_m=gt_m)


def _pred_metrics(slice_path: Path, pred_path: Path) -> dict:
    sl = S.load_slice(slice_path)
    with np.load(pred_path, allow_pickle=False) as z:
        dec = np.array(z["decision"]).astype(np.int8)
        run = str(np.asarray(z["run"]).item())
        stride = int(np.asarray(z["stride"]).item()) if "stride" in z.files else 1
    if stride != 1 or dec.shape != sl["labels"].shape:
        raise ValueError(f"{pred_path}: stride {stride}, decision {dec.shape} vs labels {sl['labels'].shape}")
    _, t = M.slice_metrics(sl, dec)
    return dict(run=run, R_s=t["R_s"], R_f=t["R_f"], abstain=t["abstain_frac"], FDR=t["FDR"], FOR=t["FOR"],
                wrong_count=int(t["n_wrong_s"]) + int(t["n_wrong_f"]), wrong_S=int(t["n_wrong_s"]),
                wrong_F=int(t["n_wrong_f"]), rank_score=t["rank_score"])


def _seeds() -> dict:
    out = {}
    for line in MATRIX_ROWS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r.get("name")] = r.get("seed")
    return out


def _load_cache() -> dict:
    try:
        c = json.loads(CACHE.read_text())
        if c.get("version") == CACHE_VERSION:
            return c
    except (OSError, ValueError):
        pass
    return {"version": CACHE_VERSION, "slices": {}, "preds": {}}


def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644 & ~_umask())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _umask() -> int:
    m = os.umask(0)
    os.umask(m)
    return m


def build(out: Path) -> dict:
    t0 = time.time()
    cache = _load_cache()
    seeds = _seeds()
    entries, warnings, n_loaded = [], [], 0
    live_slices, live_preds = set(), set()
    for slice_path in sorted((SLICES / "data").glob("*/*.npz")):
        setting, plane = slice_path.parent.name, slice_path.stem
        if plane.endswith("_s2"):
            continue
        sp = str(slice_path)
        live_slices.add(sp)
        key = _key(slice_path)
        c = cache["slices"].get(sp)
        if c is None or c["key"] != key:
            try:
                info = _slice_info(slice_path)
            except Exception as e:  # noqa: BLE001  a slice being rewritten; skip this round
                warnings.append(f"slice {sp}: {e}")
                continue
            info.pop("gt_m")
            c = cache["slices"][sp] = {"key": key, "info": info}
            n_loaded += 1
        info = c["info"]
        done = []
        for pred_path in sorted((SLICES / "pred" / setting).glob(f"{plane}__r*.npz")):
            m = PRED_RE.match(pred_path.name)
            if not m or m["plane"] != plane:
                continue
            pp = str(pred_path)
            live_preds.add(pp)
            pkey = _key(pred_path) + key
            pc = cache["preds"].get(pp)
            if pc is None or pc["key"] != pkey:
                try:
                    pm = _pred_metrics(slice_path, pred_path)
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"pred {pp}: {e}")
                    continue
                pc = cache["preds"][pp] = {"key": pkey, "metrics": pm}
                n_loaded += 1
            pm = pc["metrics"]
            stem = SLICES / "figs" / setting / f"{plane}__{m['rk']}"
            png, pdf = Path(f"{stem}.png"), Path(f"{stem}.pdf")
            seed = seeds.get(pm["run"])
            entries.append(dict(
                setting=setting, system=info["system"], controller=info["controller"], plane=plane,
                dims=info["dims"], fixed=info["fixed"], seed=int(seed) if seed is not None else None,
                rk=m["rk"], run=pm["run"], png=str(png) if png.exists() else None,
                pdf=str(pdf) if pdf.exists() else None, gt_clarity=info["gt_clarity"],
                R_s=pm["R_s"], R_f=pm["R_f"], abstain=pm["abstain"], FDR=pm["FDR"], FOR=pm["FOR"],
                wrong_count=int(pm["wrong_count"]), wrong_S=int(pm["wrong_S"]), wrong_F=int(pm["wrong_F"]),
                n_FP=int(pm["wrong_S"]), n_FN=int(pm["wrong_F"]),
                rank_score=pm["rank_score"], n_in_box=info["n_in_box"], status="done",
                source_note=info["source_note"]))
            done.append(m["rk"])
        if len(set(done)) < 3:
            stem = SLICES / "figs_gt" / setting / plane
            png, pdf = Path(f"{stem}.png"), Path(f"{stem}.pdf")
            entries.append(dict(
                setting=setting, system=info["system"], controller=info["controller"], plane=plane,
                dims=info["dims"], fixed=info["fixed"], seed=None, rk=None, run=None,
                png=str(png) if png.exists() else None, pdf=str(pdf) if pdf.exists() else None,
                gt_clarity=info["gt_clarity"], R_s=None, R_f=None, abstain=None, FDR=None, FOR=None,
                wrong_count=None, wrong_S=None, wrong_F=None, n_FP=None, n_FN=None, rank_score=None,
                n_in_box=info["n_in_box"],
                status="pending", source_note=info["source_note"], seeds_done=sorted(set(done))))
    # Forget cache rows for files that are gone.
    cache["slices"] = {k: v for k, v in cache["slices"].items() if k in live_slices}
    cache["preds"] = {k: v for k, v in cache["preds"].items() if k in live_preds}
    manifest = {"generated": datetime.now().astimezone().isoformat(timespec="seconds"), "entries": entries}
    text = json.dumps(_clean(manifest), indent=1, allow_nan=False)
    _atomic_write(out, text)
    _atomic_write(CACHE, json.dumps(_clean(cache), allow_nan=False))
    n_done = sum(e["status"] == "done" for e in entries)
    print(f"{out}: {len(entries)} entries ({n_done} done, {len(entries) - n_done} pending); "
          f"loaded {n_loaded} files, {time.time() - t0:.1f} s")
    for w in warnings:
        print("  skipped", w, file=sys.stderr)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--out", type=Path, default=SLICES / "manifest.json")
    build(p.parse_args().out)


if __name__ == "__main__":
    main()
