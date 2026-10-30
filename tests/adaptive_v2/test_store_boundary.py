"""H06 decision maps on a hand-built case and a fake run dir."""
import json

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.store_boundary import ABSTAIN, FAILURE, SUCCESS, decision_maps, decisions, shares


def test_decisions_join_invalid_and_uncertain():
    p_s = np.array([.9, .1, .5, .0])
    p_i = np.array([0., 0., 0., .9])
    dec = decisions(p_s, 1 - p_s - p_i, p_i, lam=0.5, delta=0.2, rule="two_sided", inv_thr=0.3)
    assert dec.tolist() == [SUCCESS, FAILURE, ABSTAIN, ABSTAIN]


def test_shares():
    dec = np.array([SUCCESS, SUCCESS, FAILURE, ABSTAIN])
    y = np.array([1, 0, 0, 1])
    s = shares(dec, y)
    assert s["committed_share"] == 0.75 and s["abstain_share"] == 0.25
    assert s["wrong_share_of_committed"] == 1 / 3
    assert s["success_recovery"] == 0.5 and s["success_precision"] == 0.5


def test_decision_maps_reads_epochs(tmp_path):
    OmegaConf.save(OmegaConf.create({"decision_rule": "two_sided"}), tmp_path / "resolved_config.yaml")
    rng = np.random.default_rng(0)
    for e, budget in ((0, 100), (1, 150)):
        d = tmp_path / f"epoch_{e:03d}"
        d.mkdir()
        p_s = rng.uniform(size=5)
        np.savez(d / "full_roa_per_point.npz", start_states=rng.normal(size=(5, 3)), p_success=p_s,
                 p_failure=1 - p_s, p_invalid=np.zeros(5), true_labels=np.array([1, -1, 1, -1, 1]))
        (d / "artifacts_v2.json").write_text(json.dumps({
            "epoch": e, "train_trajectories": budget,
            "eval_metrics": {"lambda_delta": {"lambda_star": 0.5, "delta": 0.1}}}))
    maps = decision_maps(tmp_path, epochs={1}, dims=(0, 2))
    assert [m["train_trajectories"] for m in maps] == [150]
    assert maps[0]["xy"].shape == (5, 2) and maps[0]["label"].tolist() == [1, 0, 1, 0, 1]
