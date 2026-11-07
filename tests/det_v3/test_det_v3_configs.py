"""det_v3 run configs: every one composes and carries the det_v3 protocol (Sumanth, 2026-11-07)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((ROOT / "scripts/det_v3/run_plan.json").read_text())
RUNS = sorted(PLAN["main"] + PLAN["abl"], key=lambda r: r["rank"])
SPLITS = {"q3d-lqr": (1000, 5000), "q3d-ppo": (5000, 10000), "q2d-ppo": (1500, 5000),
          "cp-lqr": (500, 1000), "cp-ppo": (500, 2000), "pen-lqr": (500, 1000)}
BUDGET = {"q3d-lqr": (10000, 2000), "q3d-ppo": (10000, 3000), "q2d-ppo": (2000, 1000),
          "cp-lqr": (500, 150), "cp-ppo": (500, 150), "pen-lqr": (50, 35)}


def test_plan_shape_and_ranks():
    assert len(RUNS) == 420 and len(PLAN["main"]) == 60
    assert [r["rank"] for r in RUNS] == list(range(1, 421))
    assert all(r["kind"] == "main" for r in RUNS[:60]) and all(r["kind"] == "abl" for r in RUNS[60:])
    for tier in (RUNS[:60], RUNS[60:]):                       # longest first inside each tier
        assert all(a["hours"] >= b["hours"] for a, b in zip(tier, tier[1:]))
    assert not any(r.get("ab") == "A9" or r["model"] == "Part-X" for r in RUNS)
    assert {r["owner"] for r in RUNS if r["setting"] in ("q3d-ppo", "q2d-ppo")} == {"Sumanth"}
    assert {r["owner"] for r in RUNS if r["setting"] not in ("q3d-ppo", "q2d-ppo")} == {"Dhruv"}
    assert len({r["name"] for r in RUNS}) == 420 and len({r["id"] for r in RUNS}) == 420


def test_every_run_config_composes_with_the_det_v3_protocol():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    spec = importlib.util.spec_from_file_location("rp", ROOT / "scripts/run_partx_faithful.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                              # registers the run scripts' resolvers
    with initialize_config_dir(config_dir=str(ROOT / "configs/adaptive_v2"), version_base=None):
        for r in RUNS:
            c = OmegaConf.to_container(compose(config_name="default",
                                               overrides=[f"+experiment=det_v3/runs/{r['name']}"]), resolve=True)
            change = r.get("change") or ""
            assert c["output_dir"].endswith(f"det_v3/{r['setting']}/{r['name']}_s99512")
            assert c["n_epochs"] == 11 and c["seed"] == 99512 and c["prediction_store"]["enabled"]
            # det_v3 protocol
            assert c["val_split"] == "external"
            assert c["predictor"]["lightning_trainer"]["min_steps"] == 2000
            assert (c["data_source"]["val_set_max_rows"], c["data_source"]["cal_set_max_rows"]) == SPLITS[r["setting"]]
            want_tol = 0.0 if change == "plain top-k (tie_tol = 0)" else 0.001
            assert c["acquisition"]["tie_tol"] == want_tol
            assert c["acquisition"]["selection"] == ("proportional" if change == "proportional sampling" else "topk")
            n0, k = BUDGET[r["setting"]]
            n0_run, k_run = c["initial_train_size"], c["samples_per_epoch"]
            if change == "initial batch x 0.5":
                assert n0_run == n0 // 2
            elif change == "initial batch x 2":
                assert n0_run == 2 * n0
            else:
                assert (n0_run, k_run) == (n0, k)
            assert 0 <= (n0_run + 10 * k_run) - (n0 + 10 * k) <= 5   # same final budget (pendulum half: +5)
            # det_v2 protocol kept
            assert c["threshold"]["_target_"].endswith("JSelThresholdBackend")
            assert not c["threshold"]["_target_"].endswith("HysteresisJSelThresholdBackend")
            for key in ("acquisition", "threshold", "probability"):
                assert c[key]["num_mc_samples"] == 100
            assert c["acquisition"]["score"] == ("u_h" if change.startswith("U_H") else "q_a")
            want_d2 = 0.0 if r.get("arm") == "uniform" else (float(change.split("= ")[1]) if change.startswith("d2_ratio") else 1.0)
            assert c["acquisition"]["d2_ratio"] == want_d2
            want_obj = "j_ver" if change.startswith("J_ver") else "j_sel"
            want_w = float(change.split("w = ")[1]) if "w = " in change else 0.1
            assert (c["threshold"]["objective"], c["threshold"]["w"]) == (want_obj, want_w)
            assert c["predictor"]["batched_indexing"] is True


def test_generated_configs_match_the_plan():
    import subprocess, sys
    out = subprocess.run([sys.executable, str(ROOT / "scripts/det_v3/make_run_configs.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
