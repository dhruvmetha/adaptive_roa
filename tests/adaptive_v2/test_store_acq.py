"""Acquisition profile from the pick log (H09)."""
import json

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.store_acq import acquisition_profile


def test_rounds_split_uniform_and_acquired_picks(tmp_path):
    lengths = np.array([5, 9, 3, 7, 11, 4])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.savez(tmp_path / "train.npz", offsets=offsets, states=np.zeros((offsets[-1], 2)))
    np.savetxt(tmp_path / "order.txt", np.arange(6), fmt="%d")
    np.savetxt(tmp_path / "labels.txt", np.array([1, 0, 0, 1, 0, 1]), fmt="%d")
    OmegaConf.save(OmegaConf.create({
        "initial_train_size": 2, "samples_per_epoch": 2,
        "data_source": {"trajectories_dir": str(tmp_path / "train.npz"),
                        "shuffled_indices_file": str(tmp_path / "order.txt"),
                        "shuffled_labels_file": str(tmp_path / "labels.txt")}}), tmp_path / "resolved_config.yaml")
    log = [[0, 0, "initial"], [1, 0, "initial"], [2, 0, "uniform"], [3, 3, "acquisition"],
           [4, 0, "uniform"], [1, 0, "acquisition"]]
    (tmp_path / "dataset_builder_state.json").write_text(json.dumps({"mode": "suffix", "pick_log": log}))
    r0, r1 = acquisition_profile(tmp_path)
    assert (r0["n_uniform"], r0["n_acquired"], r0["success_share_acquired"]) == (1, 1, 1.0)
    assert r0["acquired_row_fraction_mean"] == 3 / 6
    assert r1["picks_on_seen_trajectories"] == 1       # trajectory 1 again
