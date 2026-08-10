import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from adaptive_roa.benchmark.manifest import ALL_ARMS, ARM_CONFIG_GROUP, RunSpec, expand_manifest

PREDICTOR_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs/adaptive_v2/predictor"
ADAPTIVE_V2_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs/adaptive_v2")


def _spec(**kw):
    base = dict(arm="bnn_mfvi", system="pendulum", tier="production",
                acquisition="ranked", seed=42, n_epochs=10, overrides=())
    base.update(kw)
    return RunSpec(**base)


def test_run_id_is_stable_across_processes():
    # Comparing two RunSpec()s built in THIS process cannot catch a regression
    # to builtin hash(): a single process has one fixed hash seed for its
    # entire lifetime, so a hash()-based run_id would still agree with itself
    # in-process and this test would pass against broken code. Spawn separate
    # subprocesses with DIFFERENT PYTHONHASHSEED values instead -- that is the
    # actual axis hash() varies on, and the only way to distinguish it from
    # hashlib.sha1 (which does not depend on PYTHONHASHSEED at all).
    code = (
        "from adaptive_roa.benchmark.manifest import RunSpec\n"
        "print(RunSpec(arm='bnn_mfvi', system='pendulum', tier='production', "
        "acquisition='ranked', seed=42, n_epochs=10, "
        "overrides=('a=1', 'b=2')).run_id)"
    )
    run_ids = set()
    for hash_seed in ("0", "1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                 capture_output=True, text=True, check=True)
        run_ids.add(result.stdout.strip())
    assert len(run_ids) == 1, f"run_id varied across PYTHONHASHSEED values: {run_ids}"


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


def test_mlp_det_is_rejected_in_the_reference_tier():
    # Nothing upstream blocks this combination, and it is not merely awkward
    # to encode: mlp_det has no posterior at all, so there is nothing for the
    # reference tier's HMC comparison to be a reference FOR. It also collides
    # at the Hydra level -- reference_tier and mlp_det_baseline are both
    # +experiment= overrides, and hydra.compose() rejects two of those
    # ("Multiple values for experiment").
    with pytest.raises(ValueError, match="no posterior"):
        expand_manifest({"arms": ["mlp_det"], "systems": ["pendulum"],
                         "tier": "reference", "acquisition": ["ranked"],
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


@pytest.mark.parametrize("spec", [
    _spec(arm="bnn_mfvi"),
    _spec(arm="mlp"),
    _spec(arm="fm"),
    _spec(arm="mlp_det", acquisition="random"),
    _spec(arm="hmc", tier="reference"),
], ids=["bnn_mfvi-plain", "mlp-translated-to-classifier",
        "fm-translated-to-generative", "mlp_det-baseline",
        "hmc-reference-tier"])
def test_hydra_overrides_actually_compose(spec):
    # test_every_arm_config_group_exists_on_disk checks that a config FILE
    # exists; it says nothing about whether the composed override SET is
    # valid together (that is exactly the blind spot that let the original
    # mlp/fm bug through -- the file-existence check alone would not have
    # caught it if the arm names had merely collided with an unrelated file).
    # This drives the override list through real hydra.compose() against the
    # actual config tree, so a bad `predictor=`, a duplicate `+experiment=`,
    # or any other composition-time error surfaces here before a 450-run
    # campaign launch would hit it.
    with initialize_config_dir(config_dir=ADAPTIVE_V2_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default", overrides=list(spec.hydra_overrides()))
    assert cfg is not None
