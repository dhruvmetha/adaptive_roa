"""Column count alone cannot identify the eval-states layout.

A 5-column file is "2 start + 2 end + label" for a deterministic 2-D system and
"4 state + p_success" for a probabilistic 4-D system. Guessing chose the first
reading for stochastic cartpole and returned 2-D states for a 4-D system; the
failure surfaced far away as `IndexError: index 2 is out of bounds` inside
CartPoleSystem.normalize_state, ten minutes into a job.
"""
import numpy as np
import pytest

from adaptive_roa.adaptive.data_source import load_eval_states


def _write(tmp_path, name, rows):
    p = tmp_path / name
    np.savetxt(p, np.asarray(rows, dtype=float), delimiter=",")
    return str(p)


def test_probabilistic_4d_is_not_mistaken_for_deterministic_2d(tmp_path):
    """The exact cartpole failure: 5 cols = 4 state + p_success."""
    f = _write(tmp_path, "cal.txt", [
        [-3.0, -5.0, -2.64, 0.0, 0.10],
        [+1.5, +2.0, +0.30, 1.0, 0.90],
    ])
    starts, ends, labels = load_eval_states(f, state_dim=4)
    assert starts.shape == (2, 4), "must return 4-D states, not the first 2 columns"
    assert ends is None, "probabilistic files carry no end states"
    assert labels.tolist() == [-1, 1], "p<0.5 -> failure, p>=0.5 -> success"

    # Without state_dim the legacy guess silently returns 2-D — the original bug.
    legacy_starts, legacy_ends, _ = load_eval_states(f)
    assert legacy_starts.shape == (2, 2)
    assert legacy_ends is not None


def test_deterministic_4d_still_reads_as_start_end_label(tmp_path):
    """9 cols = 4 start + 4 end + label must not be read as probabilistic."""
    f = _write(tmp_path, "det.txt", [[0, 1, 2, 3, 4, 5, 6, 7, 1]])
    starts, ends, labels = load_eval_states(f, state_dim=4)
    assert starts.shape == (1, 4) and ends.shape == (1, 4)
    assert starts[0].tolist() == [0, 1, 2, 3]
    assert ends[0].tolist() == [4, 5, 6, 7]
    assert labels.tolist() == [1]


def test_pendulum_paths_are_unchanged_by_the_new_argument(tmp_path):
    """Every existing caller passes no state_dim; behaviour must be identical."""
    prob = _write(tmp_path, "p.txt", [[0.1, 0.2, 0.8], [0.3, 0.4, 0.2]])
    a = load_eval_states(prob)
    b = load_eval_states(prob, state_dim=2)
    assert a[0].shape == b[0].shape == (2, 2)
    assert a[1] is None and b[1] is None
    assert a[2].tolist() == b[2].tolist() == [1, -1]

    det = _write(tmp_path, "d.txt", [[0.1, 0.2, 0.3, 0.4, 1]])
    c = load_eval_states(det)
    d = load_eval_states(det, state_dim=2)
    assert np.array_equal(c[0], d[0]) and np.array_equal(c[1], d[1])


def test_mismatched_column_count_raises_instead_of_guessing(tmp_path):
    f = _write(tmp_path, "bad.txt", [[1, 2, 3, 4, 5, 6, 7]])
    with pytest.raises(ValueError, match="neither the probabilistic"):
        load_eval_states(f, state_dim=4)
