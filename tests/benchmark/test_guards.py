import pandas as pd
import pytest
from adaptive_roa.benchmark.guards import (
    assert_comparable,
    assert_distinct_seeds,
    assert_matched_budget,
    assert_post_fix,
    INVALIDATING_COMMITS,
    NON_COMPARABLE_COLUMNS,
    SEEDING_FIX_COMMIT,
    validate_frame,
)
from adaptive_roa.benchmark.provenance import git_sha

# A real commit from this repository's early history: older than every entry
# in INVALIDATING_COMMITS and older than SEEDING_FIX_COMMIT, so it is usable
# as a "definitely predates the fix" fixture without inventing a fake sha
# that assert_post_fix's git subprocess call could not resolve either way.
_PRE_FIX_COMMIT = "2b78b44"


def _df(**cols):
    base = dict(arm=["bnn_mfvi", "bnn_laplace"], n_epochs=[10, 10],
                seed=[42, 42], accuracy=[0.8, 0.9])
    base.update(cols)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Step 1 tests, reproduced verbatim from the task brief.
# ---------------------------------------------------------------------------

def test_matched_budget_passes_when_budgets_agree():
    assert_matched_budget(_df())


def test_matched_budget_raises_on_mismatch():
    with pytest.raises(ValueError, match="budget"):
        assert_matched_budget(_df(n_epochs=[10, 20]))


def test_val_nll_is_refused_as_a_cross_arm_column():
    # Same criterion CLASS, different quantity: the BNN arms use beta-NLL in
    # manifold-normalized coords summed over dims; gp_reg uses a plain Gaussian
    # NLL in embedded space meaned over tasks. Selection is valid within an arm.
    assert "best_val_nll" in NON_COMPARABLE_COLUMNS
    with pytest.raises(ValueError, match="not comparable across arms"):
        assert_comparable(_df(best_val_nll=[1.0, 2.0]), "best_val_nll")


def test_a_normal_metric_is_allowed():
    assert_comparable(_df(), "accuracy")


def test_identical_results_across_nominal_seeds_are_refused():
    # Three trainers ignored the run seed until the _seeding fix: replicates at
    # seeds 42/43/44 were bit-identical. A variance claim over them is fiction.
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [0.8, 0.8, 0.8], "n_epochs": [10] * 3})
    with pytest.raises(ValueError, match="identical"):
        assert_distinct_seeds(df, "accuracy")


def test_genuinely_varying_seeds_pass():
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [0.80, 0.83, 0.79], "n_epochs": [10] * 3})
    assert_distinct_seeds(df, "accuracy")


def test_single_seed_is_not_flagged():
    df = pd.DataFrame({"arm": ["bnn_mfvi"], "seed": [42],
                       "accuracy": [0.8], "n_epochs": [10]})
    assert_distinct_seeds(df, "accuracy")


# ---------------------------------------------------------------------------
# assert_distinct_seeds: false-positive classes it must NOT flag.
#
# A naive "all values equal" check also fires on ties that have nothing to do
# with the seed bug: a saturated accuracy, an integer count, a degenerate
# early epoch. These confirm the boundary/integer carve-out in
# _is_structurally_tied is doing its job, without becoming so permissive that
# it would also swallow the actual bug (test_identical_results_... above, and
# test_a_non_boundary_non_integer_tie_still_raises_even_near_a_boundary
# below, both stay strict).
# ---------------------------------------------------------------------------

def test_saturated_accuracy_across_seeds_is_not_flagged():
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [1.0, 1.0, 1.0], "n_epochs": [10] * 3})
    assert_distinct_seeds(df, "accuracy")


def test_zero_accuracy_across_seeds_is_not_flagged():
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [0.0, 0.0, 0.0], "n_epochs": [10] * 3})
    assert_distinct_seeds(df, "accuracy")


def test_identical_integer_valued_count_across_seeds_is_not_flagged():
    # e.g. a raw hit count, or an epoch index at which two runs both stopped:
    # discrete by construction, so a tie across genuinely different seeds is
    # unremarkable and must not be read as "the seed never reached the model".
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "n_success": [7.0, 7.0, 7.0], "n_epochs": [10] * 3})
    assert_distinct_seeds(df, "n_success")


def test_a_non_boundary_non_integer_tie_still_raises_even_near_a_boundary():
    # 0.999999 is close to the 1.0 saturation point but is NOT itself a
    # boundary or integer value -- exempting "close to 0/1" as well as "equal
    # to 0/1" would reopen the hole the boundary carve-out exists to close
    # without widening: an almost-saturated metric that ties exactly across
    # seeds is exactly as suspicious as one that ties at 0.8.
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [0.999999, 0.999999, 0.999999],
                       "n_epochs": [10] * 3})
    with pytest.raises(ValueError, match="identical"):
        assert_distinct_seeds(df, "accuracy")


def test_only_the_tied_arm_is_reported_not_the_whole_frame():
    # A frame with one honestly-varying arm and one suspiciously-tied arm
    # must name the offending arm, and must still raise despite the other
    # arm being fine (guards must not average good and bad rows together).
    df = pd.DataFrame({
        "arm": ["bnn_mfvi", "bnn_mfvi", "bnn_mfvi",
                "bnn_laplace", "bnn_laplace", "bnn_laplace"],
        "seed": [42, 43, 44, 42, 43, 44],
        "accuracy": [0.80, 0.83, 0.79, 0.8, 0.8, 0.8],
        "n_epochs": [10] * 6,
    })
    with pytest.raises(ValueError, match=r"bnn_laplace.*identical|identical.*bnn_laplace"):
        assert_distinct_seeds(df, "accuracy")


# ---------------------------------------------------------------------------
# INVALIDATING_COMMITS: content and wiring against the real assert_post_fix.
# ---------------------------------------------------------------------------

def test_invalidating_commits_maps_the_documented_arm_families():
    assert INVALIDATING_COMMITS["bnn_mfvi"] == "597db9f"
    assert INVALIDATING_COMMITS["bnn_ensemble"] == "597db9f"
    assert INVALIDATING_COMMITS["bnn_laplace"] == "597db9f"
    assert INVALIDATING_COMMITS["mlp_det"] == "18ea67e"
    assert INVALIDATING_COMMITS["bnn_mfvi_reg"] == "18ea67e"
    assert INVALIDATING_COMMITS["bnn_ensemble_reg"] == "18ea67e"
    assert INVALIDATING_COMMITS["bnn_laplace_reg"] == "18ea67e"
    assert INVALIDATING_COMMITS["gp_reg"] == "e642259"


def test_fm_is_not_gated_by_the_final_state_head_fix():
    # `fm` never constructs a FinalStateHead (it has its own FlowMatchingTrainer),
    # so it must not be pinned to 18ea67e -- doing so would refuse perfectly
    # valid fm runs that predate a fix that never touched fm's code path.
    assert "fm" not in INVALIDATING_COMMITS


def test_outcome_arms_without_a_bnn_posterior_are_not_gated():
    # mlp / gp / gp_optdelta never route through BayesianMLPTrainer.
    assert "mlp" not in INVALIDATING_COMMITS
    assert "gp" not in INVALIDATING_COMMITS
    assert "gp_optdelta" not in INVALIDATING_COMMITS


def test_seeding_fix_commit_is_the_real_seeding_commit():
    assert SEEDING_FIX_COMMIT == "a012c7e"


@pytest.mark.parametrize("arm,fix_sha", sorted(INVALIDATING_COMMITS.items()))
def test_every_invalidating_commit_rejects_a_pre_fix_run(arm, fix_sha):
    df = pd.DataFrame({"arm": [arm], "commit": [_PRE_FIX_COMMIT]})
    with pytest.raises(ValueError, match="predates|not found|unknown"):
        assert_post_fix(df, fix_sha=fix_sha)


@pytest.mark.parametrize("arm,fix_sha", sorted(INVALIDATING_COMMITS.items()))
def test_every_invalidating_commit_accepts_a_run_at_head(arm, fix_sha):
    head = git_sha()
    df = pd.DataFrame({"arm": [arm], "commit": [head]})
    assert_post_fix(df, fix_sha=fix_sha)


def test_seeding_fix_commit_rejects_a_pre_fix_run():
    df = pd.DataFrame({"arm": ["mlp"], "commit": [_PRE_FIX_COMMIT]})
    with pytest.raises(ValueError, match="predates|not found|unknown"):
        assert_post_fix(df, fix_sha=SEEDING_FIX_COMMIT)


def test_seeding_fix_commit_accepts_a_run_at_head():
    df = pd.DataFrame({"arm": ["mlp"], "commit": [git_sha()]})
    assert_post_fix(df, fix_sha=SEEDING_FIX_COMMIT)


# ---------------------------------------------------------------------------
# validate_frame: the whole-frame entry point the reporting path calls.
# ---------------------------------------------------------------------------

def _full_df(**overrides):
    head = git_sha()
    base = dict(
        run_id=["r1", "r2"],
        arm=["bnn_mfvi", "bnn_mfvi"],
        system=["pendulum", "pendulum"],
        tier=["production", "production"],
        acquisition=["random", "random"],
        seed=[42, 43],
        epoch=[9, 9],
        n_epochs=[10, 10],
        run_complete=[True, True],
        commit=[head, head],
        accuracy=[0.80, 0.83],
    )
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_frame_passes_on_a_well_formed_frame():
    validate_frame(_full_df())


def test_validate_frame_raises_on_budget_mismatch():
    with pytest.raises(ValueError, match="budget"):
        validate_frame(_full_df(n_epochs=[10, 20]))


def test_validate_frame_raises_on_a_non_comparable_column_across_arms():
    df = _full_df(
        arm=["mlp", "gp"], seed=[42, 42],
        commit=[git_sha(), git_sha()], best_val_nll=[1.0, 2.0],
    )
    with pytest.raises(ValueError, match="not comparable across arms"):
        validate_frame(df)


def test_validate_frame_raises_when_an_arm_predates_its_invalidating_commit():
    df = _full_df(commit=[_PRE_FIX_COMMIT, _PRE_FIX_COMMIT], seed=[42, 42])
    with pytest.raises(ValueError, match="predates|not found|unknown"):
        validate_frame(df)


def test_validate_frame_passes_when_an_arm_postdates_its_invalidating_commit():
    # Single seed: isolates the provenance check from the seed-distinctness
    # and seeding-fix-provenance checks, which need >= 2 distinct seeds.
    df = _full_df(
        run_id=["r1"], arm=["bnn_mfvi"], system=["pendulum"],
        tier=["production"], acquisition=["random"], seed=[42], epoch=[9],
        n_epochs=[10], run_complete=[True], commit=[git_sha()], accuracy=[0.8],
    )
    validate_frame(df)


def test_validate_frame_raises_when_a_multiseed_arm_predates_the_seeding_fix():
    # `mlp` carries no INVALIDATING_COMMITS entry of its own, isolating this
    # to the universal seeding-fix provenance check. 597db9f postdates every
    # INVALIDATING_COMMITS entry `mlp` could ever need (it needs none) but
    # predates SEEDING_FIX_COMMIT (a012c7e), and accuracy genuinely varies so
    # assert_distinct_seeds itself would not be the one raising here.
    df = _full_df(
        arm=["mlp", "mlp"], commit=["597db9f", "597db9f"], accuracy=[0.80, 0.83],
    )
    with pytest.raises(ValueError, match="predates|not found|unknown"):
        validate_frame(df)


def test_validate_frame_raises_when_a_multiseed_arm_reports_identical_metric():
    df = _full_df(arm=["mlp", "mlp"], commit=[git_sha(), git_sha()],
                   accuracy=[0.8, 0.8])
    with pytest.raises(ValueError, match="identical"):
        validate_frame(df)


def test_validate_frame_does_not_flag_a_saturated_metric_across_real_seeds():
    df = _full_df(arm=["mlp", "mlp"], commit=[git_sha(), git_sha()],
                   accuracy=[1.0, 1.0])
    validate_frame(df)
