from pathlib import Path

import pytest
from adaptive_roa.benchmark.manifest import ALL_ARMS, ARM_CONFIG_GROUP, RunSpec, expand_manifest

PREDICTOR_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs/adaptive_v2/predictor"


def _spec(**kw):
    base = dict(arm="bnn_mfvi", system="pendulum", tier="production",
                acquisition="ranked", seed=42, n_epochs=10, overrides=())
    base.update(kw)
    return RunSpec(**base)


def test_run_id_is_stable_across_processes():
    # Must not depend on hash randomization, dict order, or wall clock.
    assert _spec().run_id == _spec().run_id
    assert _spec(overrides=("a=1", "b=2")).run_id == _spec(overrides=("a=1", "b=2")).run_id


def test_run_id_separates_every_field_that_changes_the_experiment():
    base = _spec()
    for field, other in [("arm", "bnn_laplace"), ("system", "cartpole"),
                         ("tier", "reference"), ("acquisition", "random"),
                         ("seed", 43), ("overrides", ("x=1",))]:
        assert _spec(**{field: other}).run_id != base.run_id, field


def test_n_epochs_does_not_change_identity():
    # A run extended from 10 to 20 epochs is the SAME run resumed, not a new one.
    assert _spec(n_epochs=10).run_id == _spec(n_epochs=20).run_id


def test_expand_builds_the_full_cross_product():
    specs = expand_manifest({
        "arms": ["bnn_mfvi", "gp_reg"],
        "systems": ["pendulum", "cartpole"],
        "tier": "production",
        "acquisition": ["ranked", "random"],
        "seeds": [42, 43],
        "n_epochs": 10,
    })
    assert len(specs) == 2 * 2 * 2 * 2
    assert len({s.run_id for s in specs}) == len(specs)


def test_mlp_det_is_never_expanded_adaptively():
    # Its outcome probability collapses to {0,1}: no ranking signal exists.
    specs = expand_manifest({
        "arms": ["mlp_det"], "systems": ["pendulum"], "tier": "production",
        "acquisition": ["ranked", "random"], "seeds": [42], "n_epochs": 10,
    })
    assert [s.acquisition for s in specs] == ["random"]


def test_hydra_overrides_use_the_baseline_experiment_for_mlp_det():
    # Bare `predictor=mlp_det` silently runs adaptively: Hydra's defaults list
    # puts `predictor` before `acquisition`, so a d2_ratio set in the predictor
    # group is discarded.
    ov = _spec(arm="mlp_det", acquisition="random").hydra_overrides()
    assert "+experiment=mlp_det_baseline" in ov


def test_reference_tier_emits_its_experiment_config():
    assert "+experiment=reference_tier" in _spec(tier="reference").hydra_overrides()


def test_unknown_arm_is_rejected_at_expansion():
    with pytest.raises(ValueError, match="unknown arm"):
        expand_manifest({"arms": ["bnn_typo"], "systems": ["pendulum"],
                         "tier": "production", "acquisition": ["ranked"],
                         "seeds": [42], "n_epochs": 10})


def test_hmc_is_rejected_outside_the_reference_tier():
    # HMC is only a valid reference at the width the approximations use.
    with pytest.raises(ValueError, match="reference tier"):
        expand_manifest({"arms": ["hmc"], "systems": ["pendulum"],
                         "tier": "production", "acquisition": ["ranked"],
                         "seeds": [42], "n_epochs": 10})


def test_hydra_overrides_translate_arm_identity_to_its_config_group():
    # classifier.yaml declares `predictor.name: mlp`; generative.yaml declares
    # `predictor.name: fm`. Hydra selects a config group by FILENAME, not by
    # that field, so `predictor=mlp` / `predictor=fm` would fail to resolve.
    assert "predictor=classifier" in _spec(arm="mlp").hydra_overrides()
    assert "predictor=generative" in _spec(arm="fm").hydra_overrides()
    # An arm whose filename matches its identity passes through unchanged.
    assert "predictor=bnn_mfvi" in _spec(arm="bnn_mfvi").hydra_overrides()


def test_every_arm_config_group_exists_on_disk():
    # The check that would have caught the mlp/fm mismatch without a human
    # noticing, and will catch the next arm someone adds with a mismatched
    # filename.
    available = {p.stem for p in PREDICTOR_CONFIG_DIR.glob("*.yaml")}
    for arm in ALL_ARMS:
        config_group = ARM_CONFIG_GROUP.get(arm, arm)
        assert config_group in available, (
            f"arm {arm!r} resolves to predictor config group {config_group!r}, "
            f"which has no {config_group}.yaml in {PREDICTOR_CONFIG_DIR}"
        )
