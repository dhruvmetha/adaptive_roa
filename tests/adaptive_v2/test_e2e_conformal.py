# tests/adaptive_v2/test_e2e_conformal.py
"""End-to-end regression test: AdaptiveEngine in conformal acquisition mode, pendulum.

Locks a q_hat regression introduced by d26365c: that refactor moved q_hat
calibration onto a separate `ConformalPredictor` instance owned by the
calibration backend, leaving `threshold_backend.predictor.q_hat` permanently
`None`. In `AdaptiveEngine.run()` (adaptive_roa/adaptive_v2/engine.py, the
`if acquisition_mode in ("conformal", "partx") and need_d2_acquisition:`
block), `predictor.evaluate(X_test, y_test)` raises `RuntimeError` whenever
`predictor.q_hat is None`. The fix mirrors `predictor.q_hat = q_hat` onto the
threshold backend's predictor before calling `.evaluate()`. Before that fix,
every conformal-mode (and partx-mode) run would crash at this line -- but no
end-to-end test exercised acquisition=conformal, so the regression shipped
silently. This test exercises that exact code path and asserts
`test_coverage` is populated, which is only possible once `predictor.evaluate()`
has run successfully past the q_hat mirror line.

Composition mirrors scripts/run_adaptive.py and tests/partx/test_e2e_pendulum.py:
config_name="default" plus Hydra group overrides (system=pendulum,
predictor=classifier, acquisition=conformal, eval=full_roa).
"""
import json
import os
from pathlib import Path

import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

DATASET_DIR = (
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum_lqr_50k"
)
pytestmark = pytest.mark.skipif(
    not Path(DATASET_DIR).exists(),
    reason="shared pendulum dataset not available",
)
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs/adaptive_v2")


def _register_resolvers() -> None:
    """Register Hydra/OmegaConf resolvers used by the adaptive_v2 configs."""
    from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_shared_data_base

    for name, fn in [
        ("net_id", get_net_id),
        ("exp_dir", get_exp_dir),
        ("shared_data_base", get_shared_data_base),
    ]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, (lambda f: (lambda d="": f() or d))(fn))

    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver(
            "data_dir",
            lambda d="": os.environ.get("DATA_DIR", d) or d,
        )


@pytest.mark.slow
def test_conformal_pendulum_end_to_end(tmp_path, monkeypatch):
    """Two-epoch CPU run of AdaptiveEngine with acquisition=conformal, predictor=classifier.

    Regression guard for d26365c: prior to the fix, `threshold_backend.predictor.q_hat`
    stayed `None` after `calibration_backend.calibrate(...)`, so the subsequent
    `predictor.evaluate(X_test, y_test)` call in engine.py raised RuntimeError,
    crashing every conformal-mode run. Asserting `test_coverage` is a populated
    float proves `predictor.evaluate()` completed successfully past that line.
    """
    monkeypatch.setenv(
        "DATA_DIR",
        "/common/users/shared/pracsys/genMoPlan/data_trajectories",
    )
    monkeypatch.setenv("MPLBACKEND", "Agg")

    _register_resolvers()

    output_dir = str(tmp_path / "engine_out")
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="default",
            overrides=[
                "system=pendulum",
                "predictor=classifier",
                "acquisition=conformal",
                "eval=full_roa",
                f"output_dir={output_dir}",
                "n_epochs=2",
                "samples_per_epoch=20",
                "initial_train_size=60",
                "device=cpu",
                # d2_ratio must be > 0 so need_d2_acquisition is true and the
                # q_hat calibration + predictor.evaluate() block actually runs.
                "acquisition.d2_ratio=0.5",
                # Keep classifier training fast for a CPU unit test.
                "predictor.classifier.max_epochs=5",
            ],
        )

    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine

    result = AdaptiveEngine(cfg).run()

    assert len(result["epoch_results"]) == 2

    for i, epoch_result in enumerate(result["epoch_results"]):
        test_coverage = epoch_result["test_coverage"]
        assert test_coverage is not None, (
            f"epoch {i}: test_coverage is None -- predictor.evaluate() did not "
            "run (q_hat calibration regression, see d26365c)"
        )
        assert isinstance(test_coverage, float)
        assert 0.0 <= test_coverage <= 1.0

        art = json.loads((Path(output_dir) / f"epoch_{i:03d}" / "artifacts_v2.json").read_text())
        d1_eval_metrics = art["d1_eval_metrics"]
        assert d1_eval_metrics is not None
        assert d1_eval_metrics["coverage"] == pytest.approx(test_coverage)
