"""The wiring that makes separatrix.py and fidelity.py reachable.

Both modules were individually excellent and had no caller: ``separatrix``
was imported by nothing outside its own test, ``fidelity_vs_reference`` had
zero call sites anywhere, and no shipped code constructed a
``FidelityResult`` from artifacts. These tests exercise them through the
on-disk path a real ``run_benchmark.py report`` takes.
"""
import json

import numpy as np
import pandas as pd
import pytest

from adaptive_roa.benchmark.fidelity import FidelityResult
from adaptive_roa.benchmark.pointwise import (
    HMC_DIAGNOSTICS_FILE,
    PER_POINT_FILE,
    fidelity_table,
    load_per_point,
    separatrix_table,
)

# A two-cluster 1D-ish grid: labels flip at x = 0, so the boundary band is
# the points either side of it, and both dimensions span the same order of
# magnitude so separatrix's own scale guard is satisfied.
_GRID_X = np.linspace(-1.0, 1.0, 40)


def _states_and_labels():
    states = np.stack([_GRID_X, np.zeros_like(_GRID_X)], axis=1)
    labels = np.where(_GRID_X > 0, 1, -1)          # the on-disk {-1,1} convention
    return states, labels


def _write_per_point(directory, probs, *, labels=None, states=None):
    directory.mkdir(parents=True, exist_ok=True)
    default_states, default_labels = _states_and_labels()
    np.savez(directory / PER_POINT_FILE,
             start_states=default_states if states is None else states,
             p_success=probs,
             p_failure=1.0 - probs,
             p_invalid=np.zeros_like(probs),
             true_labels=default_labels if labels is None else labels)


def _perfect_probs():
    return np.where(_GRID_X > 0, 0.9, 0.1)


def _boundary_blind_probs():
    # Correct in the interiors, wrong on the four points nearest the flip.
    probs = _perfect_probs().copy()
    near = np.argsort(np.abs(_GRID_X))[:4]
    probs[near] = 1.0 - probs[near]
    return probs


def _write_run(root, run_id, *, probs, epoch=0, diagnostics=None, labels=None):
    directory = root / run_id / f"epoch_{epoch:03d}"
    _write_per_point(directory, probs, labels=labels)
    if diagnostics is not None:
        (directory / HMC_DIAGNOSTICS_FILE).parent.mkdir(parents=True, exist_ok=True)
        (directory / HMC_DIAGNOSTICS_FILE).write_text(json.dumps(diagnostics))


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# load_per_point
# ---------------------------------------------------------------------------

def test_load_per_point_remaps_the_on_disk_label_convention(tmp_path):
    # full_roa_per_point.npz writes {-1, 1}; conditioned_metrics requires
    # {0, 1} and refuses anything else, because astype(bool) maps -1 and 1
    # alike to True. The remap has to be explicit and it has to be here.
    _write_per_point(tmp_path, _perfect_probs())
    data = load_per_point(tmp_path)
    assert set(np.unique(data["labels"]).tolist()) == {0, 1}
    assert data["labels"].sum() == int((_GRID_X > 0).sum())


def test_load_per_point_refuses_an_unknown_label_convention(tmp_path):
    _write_per_point(tmp_path, _perfect_probs(),
                     labels=np.full(len(_GRID_X), 2))
    with pytest.raises(ValueError, match="Refusing to guess"):
        load_per_point(tmp_path)


def test_load_per_point_reports_a_missing_file_by_path(tmp_path):
    with pytest.raises(FileNotFoundError, match=PER_POINT_FILE):
        load_per_point(tmp_path)


# ---------------------------------------------------------------------------
# separatrix_table -- Task 6, now reachable
# ---------------------------------------------------------------------------

def _separatrix_frame():
    return _frame([
        dict(run_id="good_ranked", epoch=0, arm="good", system="pendulum",
             acquisition="ranked"),
        dict(run_id="blind_ranked", epoch=0, arm="blind", system="pendulum",
             acquisition="ranked"),
    ])


def test_boundary_errors_are_visible_here_and_invisible_in_the_aggregate(tmp_path):
    # The whole point of Task 6: an arm wrong on every boundary point still
    # scores high overall, and only the conditioned split separates it.
    _write_run(tmp_path, "good_ranked", probs=_perfect_probs())
    _write_run(tmp_path, "blind_ranked", probs=_boundary_blind_probs())

    table = {row["arm"]: row for row in
             separatrix_table(_separatrix_frame(), tmp_path, k=3)}
    assert table["good"]["near_boundary"] == pytest.approx(1.0)
    assert table["blind"]["near_boundary"] < 0.5
    # ...while the aggregate barely moves.
    assert table["blind"]["overall"] > 0.85


def test_separatrix_table_keys_rows_by_system_arm_acquisition(tmp_path):
    _write_run(tmp_path, "good_ranked", probs=_perfect_probs())
    _write_run(tmp_path, "blind_ranked", probs=_boundary_blind_probs())
    table = separatrix_table(_separatrix_frame(), tmp_path, k=3)
    assert {(r["system"], r["arm"], r["acquisition"]) for r in table} == {
        ("pendulum", "good", "ranked"), ("pendulum", "blind", "ranked")}
    assert all(row["k"] == 3 for row in table)


def test_separatrix_table_averages_over_seeds_in_a_cell(tmp_path):
    _write_run(tmp_path, "s42", probs=_perfect_probs())
    _write_run(tmp_path, "s43", probs=_boundary_blind_probs())
    df = _frame([
        dict(run_id="s42", epoch=0, arm="mlp", system="pendulum",
             acquisition="ranked"),
        dict(run_id="s43", epoch=0, arm="mlp", system="pendulum",
             acquisition="ranked"),
    ])
    table = separatrix_table(df, tmp_path, k=3)
    assert len(table) == 1 and table[0]["n_runs"] == 2


def test_a_run_without_per_point_artifacts_is_refused_in_the_row_not_dropped(tmp_path):
    _write_run(tmp_path, "good_ranked", probs=_perfect_probs())
    # blind_ranked writes nothing at all.
    table = {row["arm"]: row for row in
             separatrix_table(_separatrix_frame(), tmp_path, k=3)}
    assert "blind" in table, "a run with no artifact must still get a row"
    assert PER_POINT_FILE in table["blind"]["refused"]
    assert table["blind"]["n_runs"] == 0


def test_unnormalized_states_are_refused_in_the_row_verbatim(tmp_path):
    # separatrix's own scale guard: a dimension spanning hundreds next to
    # one spanning 2 makes the k-NN band meaningless even though it would
    # still run and return an answer.
    states = np.stack([_GRID_X, _GRID_X * 500.0], axis=1)
    _, labels = _states_and_labels()
    directory = tmp_path / "good_ranked" / "epoch_000"
    _write_per_point(directory, _perfect_probs(), states=states, labels=labels)
    df = _frame([dict(run_id="good_ranked", epoch=0, arm="good",
                      system="pendulum", acquisition="ranked")])
    table = separatrix_table(df, tmp_path, k=3)
    assert "unnormalized" in table[0]["refused"]


def test_separatrix_table_refuses_a_frame_it_cannot_locate_runs_from(tmp_path):
    df = _frame([dict(arm="good", system="pendulum", acquisition="ranked")])
    with pytest.raises(ValueError, match="run_id|epoch"):
        separatrix_table(df, tmp_path)


# ---------------------------------------------------------------------------
# fidelity_table -- Task 5, now reachable
# ---------------------------------------------------------------------------

_CONVERGED = {"rhat_max": 1.02, "converged": True}
_DIVERGED = {"rhat_max": 91.4, "converged": False}


def _fidelity_frame():
    return _frame([
        dict(run_id="hmc_s42", epoch=0, arm="hmc", system="pendulum", seed=42),
        dict(run_id="bnn_s42", epoch=0, arm="bnn_mfvi", system="pendulum", seed=42),
    ])


def test_fidelity_is_computed_against_a_converged_reference(tmp_path):
    _write_run(tmp_path, "hmc_s42", probs=_perfect_probs(),
               diagnostics=_CONVERGED)
    _write_run(tmp_path, "bnn_s42", probs=_perfect_probs())
    results = fidelity_table(_fidelity_frame(), tmp_path, reference_arm="hmc")
    assert set(results) == {"bnn_mfvi"}
    assert isinstance(results["bnn_mfvi"], FidelityResult)
    assert results["bnn_mfvi"].available
    assert results["bnn_mfvi"].agreement == pytest.approx(1.0)
    assert results["bnn_mfvi"].total_variation == pytest.approx(0.0)


def test_a_non_converged_reference_withholds_the_number(tmp_path):
    # hmc_reg's measured rhat_max is 7.9-91 depending on seed; a fidelity
    # number against it is meaningless, not weak, so no number is produced.
    _write_run(tmp_path, "hmc_s42", probs=_perfect_probs(),
               diagnostics=_DIVERGED)
    _write_run(tmp_path, "bnn_s42", probs=_boundary_blind_probs())
    results = fidelity_table(_fidelity_frame(), tmp_path, reference_arm="hmc")
    result = results["bnn_mfvi"]
    assert not result.available
    assert result.agreement is None and result.total_variation is None
    assert "91.4" in result.reason


def test_the_threshold_is_honoured_rather_than_the_stored_converged_flag(tmp_path):
    # The artifact says converged=True at its own threshold; a stricter bar
    # here must still withhold.
    _write_run(tmp_path, "hmc_s42", probs=_perfect_probs(),
               diagnostics={"rhat_max": 1.05, "converged": True})
    _write_run(tmp_path, "bnn_s42", probs=_perfect_probs())
    strict = fidelity_table(_fidelity_frame(), tmp_path, reference_arm="hmc",
                            rhat_threshold=1.01)
    assert not strict["bnn_mfvi"].available
    loose = fidelity_table(_fidelity_frame(), tmp_path, reference_arm="hmc",
                           rhat_threshold=1.1)
    assert loose["bnn_mfvi"].available


def test_a_reference_without_diagnostics_raises_rather_than_withholding(tmp_path):
    # A missing convergence check invalidates every row at once, not one:
    # "assume it converged" is exactly what this gate exists to prevent.
    _write_run(tmp_path, "hmc_s42", probs=_perfect_probs())   # no diagnostics
    _write_run(tmp_path, "bnn_s42", probs=_perfect_probs())
    with pytest.raises(FileNotFoundError, match="hmc_diagnostics"):
        fidelity_table(_fidelity_frame(), tmp_path, reference_arm="hmc")


def test_an_arm_with_no_matching_reference_seed_is_withheld_with_the_reason(tmp_path):
    _write_run(tmp_path, "hmc_s42", probs=_perfect_probs(),
               diagnostics=_CONVERGED)
    _write_run(tmp_path, "hmc_s43", probs=_perfect_probs(),
               diagnostics=_CONVERGED)
    _write_run(tmp_path, "bnn_s99", probs=_perfect_probs())
    df = _frame([
        dict(run_id="hmc_s42", epoch=0, arm="hmc", system="pendulum", seed=42),
        dict(run_id="hmc_s43", epoch=0, arm="hmc", system="pendulum", seed=43),
        dict(run_id="bnn_s99", epoch=0, arm="bnn_mfvi", system="pendulum", seed=99),
    ])
    result = fidelity_table(df, tmp_path, reference_arm="hmc")["bnn_mfvi"]
    assert not result.available
    assert "seed" in result.reason


def test_a_frame_without_the_reference_arm_is_refused(tmp_path):
    _write_run(tmp_path, "bnn_s42", probs=_perfect_probs())
    df = _frame([dict(run_id="bnn_s42", epoch=0, arm="bnn_mfvi",
                      system="pendulum", seed=42)])
    with pytest.raises(ValueError, match="no runs for reference arm"):
        fidelity_table(df, tmp_path, reference_arm="hmc")


def test_arms_evaluated_on_different_grids_are_refused_not_compared(tmp_path):
    _write_run(tmp_path, "hmc_s42", probs=_perfect_probs(),
               diagnostics=_CONVERGED)
    short = tmp_path / "bnn_s42" / "epoch_000"
    _write_per_point(short, _perfect_probs()[:10],
                     states=_states_and_labels()[0][:10],
                     labels=_states_and_labels()[1][:10])
    with pytest.raises(ValueError, match="shape mismatch"):
        fidelity_table(_fidelity_frame(), tmp_path, reference_arm="hmc")
