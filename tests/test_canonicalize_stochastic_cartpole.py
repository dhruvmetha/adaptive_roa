import json

import numpy as np

from scripts.canonicalize_stochastic_cartpole import convert_level, sha256


def _make_source(root):
    level = root / "sigma_015.0"
    splits = level / "train_test_splits"
    splits.mkdir(parents=True)

    states = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    starts = np.array([[1, 2, 3, 4]], dtype=np.float64)
    np.savez(
        level / "train.npz",
        states=states,
        offsets=np.array([0, 2], dtype=np.int64),
        starts=starts,
        labels=np.array([1], dtype=np.uint8),
        seeds=np.array([42], dtype=np.int64),
    )
    np.savez(
        level / "eval_success_prob.npz",
        starts=starts,
        successes=np.array([7], dtype=np.int32),
        trials=np.array([10], dtype=np.int32),
        p_success=np.array([0.7], dtype=np.float64),
        n_batches=np.array(10, dtype=np.int64),
    )
    for name in (
        "eval_states.txt",
        "cal_set.txt",
        "test_set.txt",
        "success_probabilities.txt",
    ):
        (level / name).write_text("1.000000,2.000000,3.000000,4.000000,0.700000\n")
    (splits / "shuffled_indices_0.txt").write_text("0\n")
    (splits / "shuffled_labels_0.txt").write_text("1\n")
    (level / "dataset_description.json").write_text(
        json.dumps(
            {
                "state_space": {
                    "state_order": ["x", "x_dot", "theta", "theta_dot"]
                },
                "collection": {
                    "state_order": ["x", "x_dot", "theta", "theta_dot"]
                },
            }
        )
    )
    for name in ("train_description.json", "eval_description.json"):
        (level / name).write_text(
            json.dumps({"state_order": ["x", "x_dot", "theta", "theta_dot"]})
        )
    return level


def test_convert_level_reorders_all_states_and_preserves_targets(tmp_path):
    source = _make_source(tmp_path / "source")
    destination = tmp_path / "destination" / source.name
    split_hash = sha256(source / "train_test_splits" / "shuffled_indices_0.txt")

    convert_level(source, destination)

    with np.load(destination / "train.npz") as converted:
        assert np.array_equal(
            converted["states"],
            np.array([[1, 3, 2, 4], [5, 7, 6, 8]], dtype=np.float32),
        )
        assert converted["states"].dtype == np.float32
        assert converted["starts"].dtype == np.float64
        assert converted["labels"].tolist() == [1]
        assert converted["seeds"].tolist() == [42]

    with np.load(destination / "eval_success_prob.npz") as converted:
        assert converted["starts"].tolist() == [[1, 3, 2, 4]]
        assert converted["successes"].tolist() == [7]
        assert converted["trials"].tolist() == [10]
        assert converted["p_success"].tolist() == [0.7]

    assert (destination / "eval_states.txt").read_text() == (
        "1.000000,3.000000,2.000000,4.000000,0.700000\n"
    )
    assert sha256(destination / "train_test_splits" / "shuffled_indices_0.txt") == split_hash

    description = json.loads((destination / "dataset_description.json").read_text())
    assert description["state_space"]["state_order"] == [
        "x",
        "theta",
        "x_dot",
        "theta_dot",
    ]
    assert description["canonicalization"]["labels_changed"] is False
    assert description["canonicalization"]["probabilities_changed"] is False
