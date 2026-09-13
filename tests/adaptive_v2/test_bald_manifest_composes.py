"""Every row of the BALD campaign manifest composes under Hydra.

A launch manifest is a pile of override strings that nothing type-checks. The
specific failure this guards: `predictor.lightning_trainer.enable_progress_bar=
false` (no `++`) raises ConfigCompositionException on the two BNN BALD configs,
which do not define that key, so the run dies at compose time having consumed a
queue slot. That is exactly the kind of error that is obvious in a diff and
invisible in a 27-row TSV.

Composing costs milliseconds per row and catches the whole class.
"""
from __future__ import annotations

import csv
import shlex
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "adaptive_v2"
MANIFEST = ROOT / "slurm_logs" / "bald_arms" / "manifest.tsv"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="manifest is generated into gitignored slurm_logs/; run make_manifest.py first")


def _rows():
    with MANIFEST.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _ids(rows):
    return [r["name"] for r in rows]


ROWS = _rows() if MANIFEST.exists() else []


@pytest.mark.parametrize("row", ROWS, ids=_ids(ROWS))
def test_row_composes(row):
    overrides = shlex.split(row["overrides"])
    # output_dir points at the shared experiment tree; compose does not create
    # it, but keep it out of the way regardless.
    overrides = [o for o in overrides if not o.startswith("output_dir=")]
    overrides.append("output_dir=/tmp/compose_check")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default", overrides=overrides)
    assert cfg.acquisition.score == "epistemic_bald"
    assert cfg.seed == 42
    assert cfg.n_epochs == 11


@pytest.mark.parametrize("row", ROWS, ids=_ids(ROWS))
def test_row_carries_the_timeout_fix(row):
    # The campaign's whole point. A row missing this silently trains on timeout
    # intermediates paired with x_T and is not comparable to anything else.
    assert "++data_source.timeout_intermediates=drop" in row["overrides"], row["name"]


def test_manifest_covers_the_agreed_matrix():
    assert len(ROWS) == 27, f"expected 3 arms x 9 cells, got {len(ROWS)}"
    cells = {(r["system"], r["level"]) for r in ROWS}
    assert cells == {
        ("cp", "baseline"), ("cp", "low"), ("cp", "med"),
        ("q2d", "baseline"), ("q2d", "smooth"), ("q2d", "loud"),
        ("q3d", "f_0.00"), ("q3d", "f_0.12_a0.03"), ("q3d", "f_0.40_a0.04"),
    }, sorted(cells)
    assert {r["arm"] for r in ROWS} == {
        "bnn_mfvi_reg_bald", "bnn_ens_reg_bald", "fm_outcome_bald"}


def test_q3d_uses_the_controller_the_reported_figures_use():
    # ppo_1500K. The ppo800k in the figure filenames is a stale dict key at
    # scripts/paper/plot_timeout_fix.py, not a different controller.
    for r in ROWS:
        if r["system"] == "q3d":
            assert "+controller=ppo_1500K" in r["overrides"], r["name"]
            assert "ppo_800k" not in r["overrides"], r["name"]


def test_run_dirs_are_unique():
    names = [r["name"] for r in ROWS]
    assert len(set(names)) == len(names)
