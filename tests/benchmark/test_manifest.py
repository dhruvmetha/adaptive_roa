import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import yaml

from adaptive_roa.benchmark.manifest import (
    ALL_ARMS,
    ARM_CONFIG_GROUP,
    NON_ADAPTIVE_ACQUISITION,
    RunSpec,
    expand_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_CONFIG_DIR = REPO_ROOT / "configs/adaptive_v2/predictor"
ADAPTIVE_V2_CONFIG_DIR = str(REPO_ROOT / "configs/adaptive_v2")
BENCHMARK_MANIFEST_DIR = REPO_ROOT / "configs/benchmark"


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
    # Every value here names a real Hydra config GROUP -- `cartpole_pybullet`
    # not `cartpole`, `direct` not `random` -- because expand_manifest now
    # resolves each one against configs/adaptive_v2/ and refuses a name that
    # is not a file. That refusal is the point: the shipped pilot manifest
    # used the two plausible-English names above and 93 of its 126 runs
    # could not compose.
    specs = expand_manifest({
        "arms": ["bnn_mfvi", "gp_reg"],
        "systems": ["pendulum", "cartpole_pybullet"],
        "tier": "production",
        "acquisition": ["ranked", "direct"],
        "seeds": [42, 43],
        "n_epochs": 10,
    })
    assert len(specs) == 2 * 2 * 2 * 2
    assert len({s.run_id for s in specs}) == len(specs)


def test_mlp_det_is_never_expanded_adaptively():
    # Its outcome probability collapses to {0,1}: no ranking signal exists.
    # The mode it IS recorded under must be the group it actually composes
    # under (default.yaml's `acquisition: direct`), so run_id agrees with the
    # `acquisition` column collect_runs reads back out of the artifact.
    specs = expand_manifest({
        "arms": ["mlp_det"], "systems": ["pendulum"], "tier": "production",
        "acquisition": ["ranked", "direct"], "seeds": [42], "n_epochs": 10,
    })
    assert [s.acquisition for s in specs] == [NON_ADAPTIVE_ACQUISITION]
    assert NON_ADAPTIVE_ACQUISITION == "direct"


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


# ---------------------------------------------------------------------------
# C1: every config-group override a RunSpec emits is validated, not just the
# predictor. `acquisition=random` and `system=cartpole` are both plausible
# English and neither is a file; they shipped in configs/benchmark/pilot.yaml
# and would have killed ~90 jobs one MissingConfigException at a time.
# ---------------------------------------------------------------------------

# Exactly the four groups RunSpec.hydra_overrides() can emit, and the value
# each carries for the spec built by _fake_config_root's caller below.
_EMITTED_GROUPS = {
    "system": "pendulum",
    "predictor": "bnn_mfvi",
    "acquisition": "ranked",
    "experiment": "reference_tier",
}


def _fake_config_root(tmp_path: Path, omit: str) -> Path:
    """A config tree carrying every emitted group except one value."""
    root = tmp_path / "adaptive_v2"
    for group, value in _EMITTED_GROUPS.items():
        (root / group).mkdir(parents=True, exist_ok=True)
        if group == omit:
            # The group DIRECTORY still exists -- only the value's file is
            # missing. A missing directory would be indistinguishable from
            # "this key is not a config group at all" (e.g. seed=42).
            continue
        (root / group / f"{value}.yaml").write_text("{}\n")
    return root


@pytest.mark.parametrize("omit", sorted(_EMITTED_GROUPS))
def test_expand_rejects_a_value_with_no_config_file_in_every_emitted_group(
        tmp_path, omit):
    # One parametrization per group a spec emits. Deleting the guard, or
    # narrowing it back to the predictor group alone, fails three or four of
    # these -- which is what "generalized" has to mean to be worth anything.
    root = _fake_config_root(tmp_path, omit=omit)
    manifest = {
        "arms": ["bnn_mfvi"], "systems": ["pendulum"], "tier": "reference",
        "acquisition": ["ranked"], "seeds": [42], "n_epochs": 10,
    }
    with pytest.raises(ValueError, match=f"no {omit}/{_EMITTED_GROUPS[omit]}.yaml"):
        expand_manifest(manifest, config_root=root)


def test_expand_accepts_the_same_manifest_once_every_group_file_exists(tmp_path):
    # The companion that keeps the test above honest: the four raises are
    # caused by the one missing file each time, not by the fake tree being
    # unusable in general.
    root = _fake_config_root(tmp_path, omit="")
    specs = expand_manifest({
        "arms": ["bnn_mfvi"], "systems": ["pendulum"], "tier": "reference",
        "acquisition": ["ranked"], "seeds": [42], "n_epochs": 10,
    }, config_root=root)
    assert len(specs) == 1


def test_a_non_group_override_is_not_treated_as_a_missing_config_file(tmp_path):
    # seed=42 / n_epochs=10 set VALUES inside the composed config; there is
    # no configs/adaptive_v2/seed/42.yaml and there never will be. A guard
    # that demanded one would reject every spec ever built.
    root = _fake_config_root(tmp_path, omit="")
    specs = expand_manifest({
        "arms": ["bnn_mfvi"], "systems": ["pendulum"], "tier": "reference",
        "acquisition": ["ranked"], "seeds": [42], "n_epochs": 10,
        "overrides": ["predictor.hmc.num_samples=100"],
    }, config_root=root)
    assert "seed=42" in specs[0].hydra_overrides()
    assert "predictor.hmc.num_samples=100" in specs[0].hydra_overrides()


def test_an_extra_manifest_override_naming_a_real_group_is_validated(tmp_path):
    # Extras are not exempt: `eval=` is a real group, so a typo'd value in
    # one must fail at expansion the same way `system=` does.
    root = _fake_config_root(tmp_path, omit="")
    (root / "eval").mkdir()
    (root / "eval" / "full_roa.yaml").write_text("{}\n")
    with pytest.raises(ValueError, match="no eval/typo.yaml"):
        expand_manifest({
            "arms": ["bnn_mfvi"], "systems": ["pendulum"], "tier": "reference",
            "acquisition": ["ranked"], "seeds": [42], "n_epochs": 10,
            "overrides": ["eval=typo"],
        }, config_root=root)


@pytest.mark.parametrize(
    "manifest_path", sorted(BENCHMARK_MANIFEST_DIR.glob("*.yaml")),
    ids=lambda p: p.stem)
def test_every_shipped_manifest_expands_against_the_real_config_tree(manifest_path):
    # The regression test for the shipped pilot.yaml itself. Runs against
    # configs/adaptive_v2/ as it exists on disk, so renaming or deleting a
    # config group that a manifest names breaks this immediately.
    specs = expand_manifest(yaml.safe_load(manifest_path.read_text()))
    assert specs


@pytest.mark.parametrize(
    "manifest_path", sorted(BENCHMARK_MANIFEST_DIR.glob("*.yaml")),
    ids=lambda p: p.stem)
def test_every_shipped_manifest_actually_composes_under_hydra(manifest_path):
    # File existence is necessary, not sufficient: composition can still fail
    # on an override SET (two +experiment= values, a group that exists but
    # conflicts). Deduped on the override shape -- seed/n_epochs cannot
    # affect composability -- so this is ~42 compose() calls for the pilot's
    # 126 runs rather than 126.
    specs = expand_manifest(yaml.safe_load(manifest_path.read_text()))
    shapes = {tuple(o for o in spec.hydra_overrides()
                    if not o.startswith(("seed=", "n_epochs=")))
              : spec.hydra_overrides() for spec in specs}
    assert shapes
    for overrides in shapes.values():
        with initialize_config_dir(config_dir=ADAPTIVE_V2_CONFIG_DIR,
                                   version_base=None):
            assert compose(config_name="default", overrides=list(overrides))


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
