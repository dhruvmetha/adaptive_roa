#!/usr/bin/env python
"""Start a longer schedule from another schedule's finished cells (make_manifest.EXTENDS).

q3drl200 continues q3drl100 (user, 2026-11-02). Each neural cell's finished run dir is
copied to the new cell's output dir, and the copy's saved configs get the new n_epochs,
endpoint epochs and output dir. The resume wrappers then continue it from its last banked
epoch. The engine acquires a batch at every epoch, the last one included, and
dataset_builder_state.json holds that batch, so epoch 11 trains on the same 109,000
trajectories a 22-epoch run would have. The source dir is never written.

The copy is registered in launched.tsv as world "copied" with the source cell's commit,
which the resume wrappers read; rebalance.py ignores that world.

    extend_cells.py [--setting q3drl200] [--only NAME ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import yaml
from omegaconf import OmegaConf

HERE = Path("/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun")
SOURCE_EPOCHS = 11


def overrides_of(row: dict) -> dict[str, str]:
    return dict(re.findall(r"(?:^| )'?([\w.+]+)=(\S+?)'?(?= |$)", row["overrides"]))


def source_name(name: str, setting: str, source: str) -> str:
    return name.replace(f"dr_{setting}_", f"dr_{source}_", 1)


def source_commit(launched: Path, name: str) -> str:
    rows = [ln.rstrip("\n").split("\t") for ln in open(launched) if ln.strip()]
    mine = [f for f in rows if f[0] == name or f[0].startswith(name + "~")]
    if not mine or len(mine[-1]) < 7:
        raise SystemExit(f"{name}: no launch row with a commit")
    return mine[-1][6]


def patch_config(path: Path, out: Path, n_epochs: int, endpoint_epochs: list[int]) -> None:
    cfg = OmegaConf.load(path)
    OmegaConf.update(cfg, "output_dir", str(out), force_add=True)
    OmegaConf.update(cfg, "n_epochs", n_epochs, force_add=True)
    OmegaConf.update(cfg, "prediction_store.endpoint_epochs", endpoint_epochs, force_add=True)
    # The copy may resume where eval_success_prob.npz sits beside the dataset; score three ways.
    OmegaConf.update(cfg, "eval.binary_outcomes", False, force_add=True)
    tmp = path.with_name(path.name + ".tmp")
    OmegaConf.save(cfg, tmp)
    os.replace(tmp, path)


def patch_overrides(path: Path, out: Path, n_epochs: int, endpoint_epochs: list[int]) -> None:
    items = [o for o in yaml.safe_load(path.read_text())
             if not re.match(r"(output_dir|n_epochs|prediction_store\.endpoint_epochs|eval\.binary_outcomes)=", o)]
    items += [f"n_epochs={n_epochs}", "prediction_store.endpoint_epochs=[" + ",".join(map(str, endpoint_epochs)) + "]",
              "eval.binary_outcomes=false", f"output_dir={out}"]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(items, default_flow_style=False))
    os.replace(tmp, path)


def extend(row: dict, setting: str, source: str, launched: Path, dry_run: bool = False) -> str:
    ov = overrides_of(row)
    out = Path(ov["output_dir"])
    src_name = source_name(row["name"], setting, source)
    src = Path(str(out).replace(f"/{setting}/{row['name']}", f"/{source}/{src_name}"))
    n_epochs = int(ov["n_epochs"])
    endpoint_epochs = [int(e) for e in ov["prediction_store.endpoint_epochs"].strip("[]").split(",")]
    if out.exists():
        return f"{row['name']}: {out} exists, left as is"
    banked = sorted(src.glob("epoch_*/artifacts_v2.json"))
    if len(banked) != SOURCE_EPOCHS or not (src / "final_results.json").exists():
        raise SystemExit(f"{src_name}: not finished ({len(banked)}/{SOURCE_EPOCHS} epochs banked)")
    if (src / ".amarel_mirror").exists():
        raise SystemExit(f"{src_name}: still an Amarel mirror; sync it before copying")
    commit = source_commit(launched, src_name)
    if dry_run:
        return f"{row['name']}: would copy {src} at commit {commit}"
    part = out.with_name(out.name + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    if part.exists():
        shutil.rmtree(part)
    # final_results.json marks the 100k run finished; the copy writes its own at 22 epochs.
    subprocess.run(["rsync", "-a", "--exclude", "final_results.json", "--exclude", ".amarel_mirror",
                    f"{src}/", f"{part}/"], check=True)
    for cfg in (part / ".hydra" / "config.yaml", part / "resolved_config.yaml"):
        if cfg.exists():
            patch_config(cfg, out, n_epochs, endpoint_epochs)
    if (part / ".hydra" / "overrides.yaml").exists():
        patch_overrides(part / ".hydra" / "overrides.yaml", out, n_epochs, endpoint_epochs)
    (part / "extended_from.json").write_text(json.dumps(
        {"source": str(src), "source_epochs": SOURCE_EPOCHS, "n_epochs": n_epochs, "commit": commit,
         "when": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
    os.rename(part, out)
    with open(launched, "a") as f:
        f.write("\t".join([row["name"], "copied", "-", "-", time.strftime("%Y%m%d_%H%M"), "-", commit]) + "\n")
    return f"{row['name']}: copied from {src_name} at commit {commit}"


def main() -> None:
    from make_manifest import EXTENDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="q3drl200")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--here", type=Path, default=HERE)
    a = ap.parse_args()
    source = EXTENDS[a.setting]
    rows = [r for r in csv.DictReader(open(a.here / "manifest.tsv"), delimiter="\t")
            if r["system"] == a.setting and r["arm"] != "partx_faithful"
            and (a.only is None or r["name"] in a.only)]
    for r in rows:
        print(extend(r, a.setting, source, a.here / "launched.tsv", a.dry_run), flush=True)


if __name__ == "__main__":
    main()
