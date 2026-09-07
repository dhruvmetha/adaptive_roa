from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_deterministic_pendulum_fair_eval.py"
SPEC = importlib.util.spec_from_file_location("prepare_deterministic_pendulum_fair_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_split_excludes_training_names_and_partitions_heldout(tmp_path):
    source = tmp_path / "source"
    splits = source / "train_test_splits"
    splits.mkdir(parents=True)
    rows = np.array(
        [[i, i + 0.5, i + 1, i + 1.5, i % 2] for i in range(8)], dtype=float
    )
    np.savetxt(source / "eval_states.txt", rows, delimiter=",")
    names = [f"sequence_{i}.txt" for i in range(8)]
    (splits / "all_shuffled_indices.txt").write_text("\n".join(names) + "\n")
    (splits / "shuffled_indices_0.txt").write_text(
        "sequence_1.txt\nsequence_6.txt\n"
    )

    output = tmp_path / "output"
    metadata = MODULE.build_split(source, output, cal_size=2, seed=3)

    heldout = np.loadtxt(output / "eval_states.txt", delimiter=",")
    cal = np.loadtxt(output / "cal_set.txt", delimiter=",")
    test = np.loadtxt(output / "test_set.txt", delimiter=",")
    assert metadata["n_all"] == 8
    assert metadata["n_train_excluded"] == 2
    assert metadata["n_heldout"] == 6
    assert cal.shape == (2, 5)
    assert test.shape == (4, 5)
    assert set(heldout[:, 0]) == {0, 2, 3, 4, 5, 7}
    assert set(cal[:, 0]) | set(test[:, 0]) == set(heldout[:, 0])
    assert set(cal[:, 0]).isdisjoint(set(test[:, 0]))
