"""partx_faithful_traj: evaluated rollouts contribute their intermediate states (spec 2026-09-15)."""
import json
import os
import pickle

import numpy as np
import pytest

from adaptive_roa.partx.tree import build_root
from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig
from adaptive_roa.partx_faithful.runner import run_partx_faithful, snapshot_ids
from adaptive_roa.partx_faithful.sampling import PoolSampler
from adaptive_roa.partx_faithful.state import load_state, save_state
from adaptive_roa.partx_faithful.trajectory import PoolTrajectoryStates
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.utils.env_config import get_data_dir


class _Stub:
    """g(x) = x0 - 0.5, sd 0.1/sqrt(n). Deterministic and instant."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        self.n = max(len(X), 1)
        return self

    def latent(self, X):
        X = np.atleast_2d(X)
        return X[:, 0] - 0.5, np.full(len(X), 0.1 / np.sqrt(self.n))


class _Mirror:
    """Rollout p walks from its start toward (centre - x0, x1) in p % 4 states; every 7th times out."""

    def __init__(self, starts, centre):
        self.starts, self.centre = np.asarray(starts, dtype=float), float(centre)

    def states(self, p):
        p = int(p)
        if p % 7 == 0:
            return np.empty((0, self.starts.shape[1])), True
        x = self.starts[p]
        end = x.copy()
        end[0] = self.centre - x[0]
        t = np.linspace(0.0, 1.0, p % 4 + 2)[1:-1]
        return x + t[:, None] * (end - x), False


class _Nothing:
    def states(self, p):
        return np.empty((0, 2)), False


def _n_states(p):
    return 0 if p % 7 == 0 else p % 4


def _px(budget, provider="mirror", n=4000, seed=0, initial=40):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 1, size=(n, 2))
    labels = np.where(starts[:, 0] < 0.5, 1.0, -1.0)
    used = np.zeros(n, dtype=bool)
    used[:initial] = True
    ps = PoolSampler(starts, labels, used, scale=np.ones(2))
    traj = {"mirror": _Mirror(starts, 1.0), "nothing": _Nothing(), None: None}[provider]
    cfg = PartXConfig(n0=4, n_bo=2, n_c=6, R=3, M=40, seed=seed)
    px = PartX(cfg, np.zeros(2), np.ones(2), ps, _Stub, budget, init_X=starts[:initial],
               init_y=labels[:initial], init_pool_idx=np.arange(initial), trajectory=traj)
    return px, ps, traj


def test_the_initial_rollouts_contribute_their_states_to_the_root():
    px, _, _ = _px(budget=60)
    expected = sum(_n_states(p) for p in range(40))
    assert px.n_traj_states == expected > 0
    assert len(px.regions[0].samples) == 40 + expected
    assert sorted(px.traj_ids) == list(range(40))
    assert px.n_timeout_rollouts == sum(p % 7 == 0 for p in range(40))


def test_every_evaluated_rollout_adds_its_states_with_its_label():
    px, _, _ = _px(budget=200)
    px.run()
    pool = px.store.pool_idx(np.arange(px.store.n))
    rollouts = np.flatnonzero(pool >= 0)
    assert sorted(px.traj_ids) == rollouts.tolist()
    for sid in rollouts:
        ids = px.traj_ids[int(sid)]
        assert len(ids) == _n_states(int(pool[sid]))
        if ids:
            assert np.all(px.store.pool_idx(ids) == -1)
            assert np.all(px.store.y(ids) == px.store.y([int(sid)])[0])
    assert px.n_traj_states == sum(len(v) for v in px.traj_ids.values())


def test_states_sit_in_exactly_one_leaf_inside_its_box():
    px, _, _ = _px(budget=200)
    px.run()
    owner = {}
    for leaf in px.leaves():
        assert np.all(leaf.contains(px.store.X(leaf.samples)))
        for s in leaf.samples:
            assert s not in owner, f"sample {s} is in two leaves"
            owner[s] = leaf.rid
    states = [s for v in px.traj_ids.values() for s in v]
    assert states and all(s in owner for s in states)
    # The walk toward the mirror point crosses x0 = 0.5, so states must land outside their rollout's leaf.
    assert sum(owner[s] != owner[sid] for sid, v in px.traj_ids.items() for s in v) > 0


def test_states_never_touch_the_budget_or_the_log():
    px, _, _ = _px(budget=200)
    px.run()
    assert px.used == 200 == len(px.log) == len(set(px.log))
    assert np.all(px.store.pool_idx(px.log) >= 0)


def test_a_provider_with_no_states_changes_nothing():
    a, _, _ = _px(budget=150, provider=None)
    a.run()
    b, _, _ = _px(budget=150, provider="nothing")
    b.run()
    assert a.log == b.log and a.store.n == b.store.n
    shape = lambda px: [(r.rid, r.rtype, tuple(r.low), tuple(r.high)) for r in px.leaves()]  # noqa: E731
    assert shape(a) == shape(b)


def test_a_pickled_run_with_states_resumes_exactly(tmp_path):
    full, _, _ = _px(budget=150)
    full.run()
    part, _, _ = _px(budget=150)
    part.run_root()
    part.step()
    part.step()
    save_state(part, tmp_path / "s.pkl", "h")
    resumed = load_state(tmp_path / "s.pkl", "h")
    _, fresh, traj = _px(budget=150)
    with pytest.raises(RuntimeError, match="provider"):
        resumed.attach(fresh, _Stub)
    resumed.attach(fresh, _Stub, traj)
    resumed.run()
    assert resumed.log == full.log
    assert resumed.traj_ids == full.traj_ids


class _FakeDS:
    start_states = np.zeros((3, 2))
    trajs = {0: np.arange(10.0).reshape(5, 2), 1: np.arange(4.0).reshape(2, 2),
             2: np.arange(6.0).reshape(3, 2)}

    def get_trajectory_length(self, p):
        return len(self.trajs[p])

    def _drops_intermediates(self, p, n_rows):
        return p == 2

    def load_trajectory(self, p):
        return self.trajs[p]


def test_pool_provider_returns_the_rows_between_x0_and_xT():
    t = PoolTrajectoryStates(_FakeDS())
    rows, timed_out = t.states(0)
    np.testing.assert_array_equal(rows, np.arange(10.0).reshape(5, 2)[1:-1])
    assert not timed_out
    rows, timed_out = t.states(1)
    assert rows.shape == (0, 2) and not timed_out
    rows, timed_out = t.states(2)
    assert rows.shape == (0, 2) and timed_out


_POOL = os.path.join(get_data_dir(), "stochastic/pendulum/gaussian_signal/lqr/med")


@pytest.mark.skipif(not os.path.isdir(_POOL), reason="needs the pendulum lqr med pool")
def test_the_states_are_the_rows_the_neural_arms_train_on_minus_x0():
    from adaptive_roa.adaptive.data_source import TrajectoryDataSourceConfig
    from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource

    ds = NpzTrajectoryDataSource(TrajectoryDataSourceConfig(
        trajectories_dir=_POOL,
        shuffled_indices_file=os.path.join(_POOL, "train_test_splits/shuffled_indices_0.txt"),
        shuffled_labels_file=os.path.join(_POOL, "train_test_splits/shuffled_labels_0.txt"),
        timeout_intermediates="drop"))
    t = PoolTrajectoryStates(ds)
    n_timeouts = 0
    for p in range(300):
        neural, _ = ds.get_all_endpoint_pairs_from_trajectory(p, "train")
        rows, timed_out = t.states(p)
        n_timeouts += int(timed_out)
        np.testing.assert_array_equal(np.asarray(rows, float), np.asarray(neural, float)[1:])
    assert 0 < n_timeouts < 300          # this pool exercises both branches


def _p_true(X):
    return 1.0 / (1.0 + np.exp(-4.0 * (1.0 - np.abs(X[:, 0]))))


@pytest.fixture
def toy(tmp_path):
    system = PendulumSystem()
    root = build_root(system)
    rng = np.random.default_rng(0)
    starts = rng.uniform(root.low, root.high, size=(3000, 2))
    labels = np.where(rng.random(3000) < _p_true(starts), 1.0, -1.0)
    g0, g1 = np.meshgrid(np.linspace(-3.0, 3.0, 15), np.linspace(-6.0, 6.0, 15))
    grid = np.round(np.column_stack([g0.ravel(), g1.ravel()]), 6)
    droot = tmp_path / "data"
    droot.mkdir()
    np.savetxt(droot / "test_set.txt", np.column_stack([grid, _p_true(grid)]), delimiter=",", fmt="%.6f")
    return system, starts, labels, droot


def _go(toy, out):
    system, starts, labels, droot = toy
    return run_partx_faithful(
        system=system, starts=starts, labels=labels, eval_states_file=str(droot / "test_set.txt"),
        output_dir=str(out), initial=60, samples_per_epoch=30, n_epochs=4,
        pcfg=PartXConfig(n0=5, n_bo=2, n_c=10, R=4, M=50, seed=0),
        gp_kwargs=dict(n_restarts=1, max_points=500), falsify="failure",
        attractor_radius=0.1, extra={"test": True}, trajectory=_Mirror(starts, 0.0))


def test_each_epoch_snapshots_exactly_the_states_of_its_rollouts(toy, tmp_path):
    out = tmp_path / "run"
    summary = _go(toy, out)
    with open(out / "partx_state.pkl", "rb") as f:
        px = pickle.load(f)["px"]
    assert summary["n_traj_states"] == px.n_traj_states > 0
    for e, n_new in enumerate([0, 30, 60, 90]):
        d = out / f"epoch_{e:03d}"
        a = json.loads((d / "artifacts_v2.json").read_text())
        assert a["sampling_mode"] == "partx_faithful_traj"
        base = set(range(px.n_init)) | set(px.log[:n_new])
        n_states = sum(len(px.traj_ids.get(s, [])) for s in base)
        allowed = snapshot_ids(px, n_new)
        assert len(allowed) == px.n_init + n_new + n_states
        assert a["acquisition"]["diagnostics"]["n_traj_states"] == n_states
        with np.load(d / "partx_leaves.npz") as z:
            assert int(z["n_samples"].sum()) == len(allowed)


def test_the_traj_arm_resumes_after_a_crash(toy, tmp_path, monkeypatch):
    _go(toy, tmp_path / "a")
    real_step = PartX.step
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated preemption")
        return real_step(self)

    monkeypatch.setattr(PartX, "step", flaky)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        _go(toy, tmp_path / "b")
    monkeypatch.setattr(PartX, "step", real_step)
    _go(toy, tmp_path / "b")
    fa = json.loads((tmp_path / "a" / "final_results.json").read_text())
    fb = json.loads((tmp_path / "b" / "final_results.json").read_text())
    fa.pop("elapsed_s_this_process")
    fb.pop("elapsed_s_this_process")
    assert fa == fb
