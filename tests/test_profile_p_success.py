"""Guards for the p_success latency profiler.

The profiler's failure mode is silence, not a crash. `clf_ensemble` is absent
from the probabilistic-classifier registry, so `get_probabilistic_classifier_class`
falls back to the family alias "classifier" and hands back the SINGLE-MLP
wrapper. That loads cleanly, runs, and reports a number roughly M times too fast
with nothing to catch it. These tests pin the mapping and the guard that would
notice.

Tests needing the shared experiment tree skip when it is not mounted.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.profile_p_success import (
    ARM_SUFFIXES,
    SYSTEM_PREFIX,
    WRAPPER_ALIAS,
    check_member_count,
    last_epoch,
)
from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

DOCS = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
LEVEL_CSVS = [
    DOCS / "pendulum/lqr/gaussian_all_levels.csv",
    DOCS / "cartpole/lqr/gaussian_all_levels.csv",
    DOCS / "quadrotor2d/rl/quad2d_corridor_sine_ambient_all_levels.csv",
    DOCS / "quadrotor3d/lqr/quad3d_corridor_sine_ambient_all_levels.csv",
]

# Every arm in the docs, keyed to the run-dir suffix whose predictor it shares.
# Acquisition variants cost the same per query as the predictor they run on;
# this table is what lets six cells stand in for eighteen arms.
ARM_TO_PROFILED_SUFFIX = {
    "dir00_s42": "dir00_s42", "dir00_s43": "dir00_s42", "dir00_s44": "dir00_s42",
    "epi_var": "dir00_s42", "epi_bald": "dir00_s42", "epi_var_anch": "dir00_s42",
    "yield_a1": "dir00_s42", "yield_mlp": "dir00_s42",
    "clf_dir00": "clf_dir00", "clf_yield": "clf_dir00", "clf_epi_var": "clf_dir00",
    "clf_epi_bald": "clf_dir00", "clf_epi_var_anch": "clf_dir00",
    "bnn_ens": "bnn_ens",
    "bnn_mfvi": "bnn_mfvi", "bnn_mfvi_bald": "bnn_mfvi",
    "bnn_lap": "bnn_lap",
    "partx_fix": "partx_fix",
}


def _arms_in(path: Path) -> set[str]:
    with open(path) as f:
        return {row["arm"] for row in csv.DictReader(f)}


@pytest.mark.parametrize("csv_path", LEVEL_CSVS, ids=lambda p: p.parent.parent.name)
def test_every_documented_arm_maps_to_a_profiled_cell(csv_path):
    """No arm in the stochastic docs may be silently absent from the profile."""
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not mounted")
    unmapped = _arms_in(csv_path) - set(ARM_TO_PROFILED_SUFFIX)
    assert not unmapped, (
        f"{csv_path.name} carries arms with no profiled predictor: {sorted(unmapped)}. "
        f"Add them to ARM_TO_PROFILED_SUFFIX, or to ARM_SUFFIXES if they are a new family."
    )


def test_profiled_suffixes_cover_every_mapping_target():
    """The mapping cannot point at a cell the profiler does not run."""
    targets = set(ARM_TO_PROFILED_SUFFIX.values())
    assert targets <= set(ARM_SUFFIXES), (
        f"mapping targets not profiled: {sorted(targets - set(ARM_SUFFIXES))}"
    )


def test_clf_ensemble_alias_avoids_the_single_mlp_wrapper():
    """Without the alias, clf_ensemble loads one MLP instead of M and no error fires."""
    aliased = get_probabilistic_classifier_class(WRAPPER_ALIAS["clf_ensemble"])
    assert aliased.posterior_kind == "ensemble"

    # This is the wrong answer the alias exists to prevent. It resolves via the
    # family alias, it does not raise, and it is a single-member network.
    fallback = get_probabilistic_classifier_class("classifier")
    assert fallback is not aliased
    assert not hasattr(fallback, "posterior_kind")


def test_check_member_count_rejects_a_short_ensemble():
    cfg = {"predictor": {"name": "fm_ensemble", "ensemble": {"n_members": 5}}}

    class Backend:
        n_members = 1

    with pytest.raises(RuntimeError, match="spuriously fast"):
        check_member_count("fm_ensemble", cfg, Backend())


def test_check_member_count_passes_a_full_ensemble():
    cfg = {"predictor": {"name": "clf_ensemble", "bnn": {"n_members": 5}}}

    class Backend:
        n_members = 5

    check_member_count("clf_ensemble", cfg, Backend())


@pytest.mark.parametrize("system", sorted(SYSTEM_PREFIX))
def test_every_profiled_run_dir_exists_with_checkpoints(system):
    """A missing cell should be found here, not after an hour of FM timing."""
    prefix = SYSTEM_PREFIX[system]
    if not prefix.parent.is_dir():
        pytest.skip(f"{prefix.parent} not mounted")
    for suffix in ARM_SUFFIXES:
        run_dir = Path(f"{prefix}_{suffix}")
        assert run_dir.is_dir(), f"missing run dir {run_dir}"
        epoch = last_epoch(run_dir)
        epoch_dir = run_dir / f"epoch_{epoch:03d}"
        ckpts = list(epoch_dir.rglob("best*.ckpt")) + list(epoch_dir.rglob("gp.pt"))
        assert ckpts, f"no checkpoint under {epoch_dir}"


# ── query-state loading ──────────────────────────────────────────────────────
# Regression guards. The first version of this reader went through
# `adaptive.data_source.load_eval_states`, which parses the ENDPOINT layout
# (start, end, label). The stochastic test sets carry (state, p_success)
# instead: that reader rejected it outright on both quadrotors and, worse,
# mis-parsed pendulum and returned numbers that looked fine.

def _write(tmp_path, rows):
    p = tmp_path / "test_set.txt"
    p.write_text("\n".join(",".join(f"{v}" for v in r) for r in rows) + "\n")
    return p


class _Sys:
    def __init__(self, d):
        self.state_dim = d


def test_query_states_reads_the_p_success_layout(tmp_path):
    from scripts.profile_p_success import load_query_states

    # state_dim + 1: two state columns then the ground-truth probability.
    path = _write(tmp_path, [(1.0, 2.0, 0.8), (3.0, 4.0, 0.1)])
    cfg = {"data_source": {"test_set_file": str(path)}}
    states = load_query_states(cfg, _Sys(2), n_needed=2)
    assert states.shape == (2, 2)
    assert states[1].tolist() == [3.0, 4.0]


def test_query_states_reads_the_endpoint_layout(tmp_path):
    from scripts.profile_p_success import load_query_states

    # 2*state_dim + 1: start, end, label. The query state is still leading.
    path = _write(tmp_path, [(1.0, 2.0, 9.0, 9.0, 1)])
    cfg = {"data_source": {"test_set_file": str(path)}}
    states = load_query_states(cfg, _Sys(2), n_needed=1)
    assert states.tolist() == [[1.0, 2.0]]


def test_query_states_rejects_an_unrecognized_layout(tmp_path):
    from scripts.profile_p_success import load_query_states

    # 4 columns is neither 3 nor 5 for a 2-D state. Slicing the first two
    # anyway is what silently profiled pendulum on mis-parsed values.
    path = _write(tmp_path, [(1.0, 2.0, 3.0, 4.0)])
    cfg = {"data_source": {"test_set_file": str(path)}}
    with pytest.raises(ValueError, match="expected 3 .* or 5 "):
        load_query_states(cfg, _Sys(2), n_needed=1)


def test_query_states_resamples_when_the_file_is_short(tmp_path):
    from scripts.profile_p_success import load_query_states

    path = _write(tmp_path, [(1.0, 2.0, 0.5), (3.0, 4.0, 0.5)])
    cfg = {"data_source": {"test_set_file": str(path)}}
    assert load_query_states(cfg, _Sys(2), n_needed=7).shape == (7, 2)


# ── MC batching ──────────────────────────────────────────────────────────────
# Batching the K draws is only valid if each state's K rows stay grouped.
# `repeat` instead of `repeat_interleave` tiles the states, so the reshape
# below would average across DIFFERENT states and still return numbers in
# [0, 1] that look entirely reasonable.

import numpy as np
import torch


class _IdentityHandle:
    """predict_endpoint_member returns its input, so the label is the state."""
    n_members = 2

    class _P:
        n_members = 2

    posterior = _P()

    def predict_endpoint_member(self, m, x, **kw):
        return x


class _SignSystem:
    """Success iff the first coordinate is positive. Deterministic, no noise."""
    state_dim = 1

    def classify_attractor(self, pred, radius):
        return torch.where(pred[:, 0] > 0, 1, -1)


def _batched_backend(k):
    from omegaconf import OmegaConf
    from scripts.profile_p_success import MCBatchedEnsembleEndpointMC

    cfg = OmegaConf.create({"attractor_radius": 0.1, "num_mc_samples": k})
    b = MCBatchedEnsembleEndpointMC(cfg, _SignSystem(), "cpu")
    b.bind_model(_IdentityHandle())
    return b


def test_mc_batching_keeps_each_states_draws_together():
    """A tiled (rather than interleaved) repeat would return 0.5 for both."""
    backend = _batched_backend(k=8)
    # One certain-success state and one certain-failure state.
    p = backend.estimate_members(np.array([[1.0], [-1.0]], dtype=np.float32))
    assert p.shape == (2, 2)                      # [M, N]
    np.testing.assert_allclose(p[0], [1.0, 0.0])
    np.testing.assert_allclose(p[1], [1.0, 0.0])


def test_mc_batching_survives_chunking():
    """Chunk boundaries must not shift the state-to-row mapping."""
    backend = _batched_backend(k=4)
    backend.max_rows_per_call = 8                 # 2 states per chunk
    states = np.array([[1.0], [-1.0], [1.0], [-1.0], [1.0]], dtype=np.float32)
    p = backend.estimate_members(states)
    np.testing.assert_allclose(p[0], [1.0, 0.0, 1.0, 0.0, 1.0])


def test_mc_batching_reports_its_mode():
    from scripts.profile_p_success import cost_drivers

    backend = _batched_backend(k=8)
    d = cost_drivers({"predictor": {}}, backend, _IdentityHandle())
    assert d["mc_mode"] == "batched"
    assert d["num_mc_samples"] == 8
