import pandas as pd
import pytest
from adaptive_roa.benchmark.aggregate import PROVENANCE_COLUMNS
from adaptive_roa.benchmark.guards import (
    assert_all_complete,
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
# Step 1 tests, reproduced verbatim from the task brief. Still pass unchanged
# under the review-round redesign: none of these fixtures carry
# system/acquisition/tier/epoch, so assert_distinct_seeds falls back to
# grouping by arm alone, and none of them carry a second metric column, so
# the whole-metric-vector check collapses to exactly the single-column check
# these tests were written against.
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
# F1 (Critical, review round 2): assert_distinct_seeds must not pool systems,
# epochs, acquisition modes and tiers into one comparison, and must raise on
# the ENTIRE metric vector tying, not a single column.
# ---------------------------------------------------------------------------

def test_scoping_catches_a_broken_seed_that_pooling_by_arm_alone_would_mask():
    # Reproduces the reviewer's demonstration on a smaller frame: every seed
    # WITHIN one (system, epoch) cell reports the identical value, but the
    # value differs from cell to cell -- which is exactly what would make a
    # naive "pool everything with this arm" grouping see 4 different values
    # and conclude "not identical", even though every cell individually is a
    # complete seeding failure.
    rows = []
    cell_value = {("pendulum", 0): 0.5, ("pendulum", 1): 0.6,
                  ("quad2d", 0): 0.7, ("quad2d", 1): 0.8}
    for (system, epoch), value in cell_value.items():
        for seed in (42, 43, 44):
            rows.append(dict(arm="bnn_mfvi", system=system, epoch=epoch,
                              seed=seed, accuracy=value, tp=10, n_epochs=10))
    df = pd.DataFrame(rows)

    # Sanity: pooled by arm alone, the 4 distinct values would NOT look tied.
    assert df["accuracy"].nunique() == len(cell_value)

    with pytest.raises(ValueError, match="identical"):
        assert_distinct_seeds(df, "accuracy")


def test_the_same_data_sliced_to_one_cell_also_raises():
    df = pd.DataFrame({
        "arm": ["bnn_mfvi"] * 3, "system": ["pendulum"] * 3, "epoch": [0] * 3,
        "seed": [42, 43, 44], "accuracy": [0.5, 0.5, 0.5], "tp": [10, 10, 10],
        "n_epochs": [10] * 3,
    })
    with pytest.raises(ValueError, match="identical"):
        assert_distinct_seeds(df, "accuracy")


def test_saturated_metric_does_not_abort_when_a_sibling_metric_still_varies():
    # accuracy/F1/precision are rationals over a fixed, finite evaluation
    # grid, so honest replicate seeds land on the SAME accuracy routinely,
    # even while disagreeing on WHICH points they got right. tp genuinely
    # differs here -- real variance, just not visible in accuracy -- so this
    # must NOT abort.
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [1.0, 1.0, 1.0], "tp": [10, 12, 9],
                       "n_epochs": [10] * 3})
    assert_distinct_seeds(df, "accuracy")


def test_a_lone_tied_column_with_no_corroboration_still_raises():
    # Without any other metric column to check, there is no way to tell
    # "genuine saturation" from "the seed never reached the model" -- kept
    # from the prior round (the 0.999999-near-a-boundary judgment was
    # confirmed correct), generalized: this is no longer about boundary
    # values specifically, it's that an unaccompanied tie has no
    # corroborating evidence either way, and the guard raises rather than
    # assume it's benign.
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [0.999999, 0.999999, 0.999999],
                       "n_epochs": [10] * 3})
    with pytest.raises(ValueError, match="identical"):
        assert_distinct_seeds(df, "accuracy")


def test_an_integer_valued_metric_tying_alone_now_raises():
    # This is the adversarial case the OLD (pre-review) integer/boundary
    # exemption let through silently: an integer-valued count tying across
    # seeds, with nothing else present to check, used to be waved through
    # as "just an integer coincidence". A broken-seed arm reporting
    # lambda_delta.n_uncertain = 4173.0 at seeds 42/43/44 is exactly this
    # shape, and with no other column to corroborate real variance, it must
    # now raise.
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "n_uncertain": [4173.0, 4173.0, 4173.0],
                       "n_epochs": [10] * 3})
    with pytest.raises(ValueError, match="identical"):
        assert_distinct_seeds(df, "n_uncertain")


def test_an_integer_valued_metric_tying_with_real_variance_elsewhere_passes():
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "n_uncertain": [4173.0, 4173.0, 4173.0],
                       "accuracy": [0.80, 0.83, 0.79], "n_epochs": [10] * 3})
    assert_distinct_seeds(df, "n_uncertain")


def test_only_the_tied_arm_is_reported_not_the_whole_frame():
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
# F5: NaN must not produce a false "identical" raise, and inf must not crash.
# ---------------------------------------------------------------------------

def test_a_missing_value_for_one_seed_does_not_false_raise():
    # Only 2 of 3 seeds have a value at all -- that is missing data, not
    # evidence of an identical result, and must not be read as either.
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [None, 0.8, 0.8], "n_epochs": [10] * 3})
    assert_distinct_seeds(df, "accuracy")


def test_an_infinite_value_ties_correctly_without_crashing():
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "loss": [float("inf")] * 3, "n_epochs": [10] * 3})
    with pytest.raises(ValueError, match="identical"):
        assert_distinct_seeds(df, "loss")


# ---------------------------------------------------------------------------
# F2 (Important): assert_matched_budget must not pool systems either.
# ---------------------------------------------------------------------------

def test_matched_budget_allows_legitimately_different_budgets_per_system():
    df = pd.DataFrame({
        "arm": ["fm", "fm", "gp_reg", "gp_reg"],
        "system": ["pendulum", "quad3d", "pendulum", "quad3d"],
        "n_epochs": [19, 30, 19, 30],
    })
    assert_matched_budget(df)


def test_matched_budget_still_raises_when_arms_differ_within_a_system():
    df = pd.DataFrame({
        "arm": ["fm", "gp_reg"], "system": ["pendulum", "pendulum"],
        "n_epochs": [19, 25],
    })
    with pytest.raises(ValueError, match="budget"):
        assert_matched_budget(df)


def test_matched_budget_still_raises_when_one_arm_spans_two_budgets_in_one_system():
    df = pd.DataFrame({
        "arm": ["fm", "fm"], "system": ["pendulum", "pendulum"],
        "n_epochs": [19, 25],
    })
    with pytest.raises(ValueError, match="budget"):
        assert_matched_budget(df)


# ---------------------------------------------------------------------------
# F3 (Important): a null arm/n_epochs/seed must raise, not be silently
# dropped by pandas groupby.
# ---------------------------------------------------------------------------

def test_matched_budget_raises_on_a_null_arm():
    df = pd.DataFrame({"arm": ["bnn_mfvi", None], "n_epochs": [10, 10]})
    with pytest.raises(ValueError, match="null"):
        assert_matched_budget(df)


def test_matched_budget_raises_on_a_null_n_epochs():
    df = pd.DataFrame({"arm": ["bnn_mfvi", "bnn_mfvi"], "n_epochs": [10, None]})
    with pytest.raises(ValueError, match="null"):
        assert_matched_budget(df)


def test_assert_distinct_seeds_raises_on_a_null_arm():
    df = pd.DataFrame({"arm": ["bnn_mfvi", None, None], "seed": [42, 43, 44],
                       "accuracy": [0.8, 0.8, 0.8]})
    with pytest.raises(ValueError, match="null"):
        assert_distinct_seeds(df, "accuracy")


def test_assert_distinct_seeds_raises_on_a_null_seed():
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, None, 44],
                       "accuracy": [0.8, 0.8, 0.8]})
    with pytest.raises(ValueError, match="null"):
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
# F6: NON_COMPARABLE_COLUMNS is dormant against the real schema -- keep that
# claim honest instead of just asserted in a comment.
# ---------------------------------------------------------------------------

def test_non_comparable_columns_do_not_occur_in_a_real_aggregated_frame():
    # best_val_nll / val_nll / train_loss are Lightning `self.log()` MONITOR
    # names (see `_MONITOR = "val_nll"` in bayesian_mlp_trainer.py and
    # final_state_trainer.py): they live in the PL CSVLogger's own
    # metrics.csv, never in artifacts_v2.json's eval_metrics, so
    # aggregate.collect_runs() has no path that produces a column with these
    # names today. This is a representative sample of the columns a real
    # frame DOES carry (aggregate's provenance columns plus the
    # full_roa.py-emitted metric names actually observed in
    # adaptive_v2/eval/full_roa.py), not an exhaustive schema.
    real_ish_columns = set(PROVENANCE_COLUMNS) | {
        "accuracy", "tp", "tn", "fp", "fn", "n_confident", "n_uncertain",
        "uncertain_pct", "separatrix_pct", "q_hat_training", "q_hat_eval",
        "n_cal_eval",
    }
    assert not (NON_COMPARABLE_COLUMNS & real_ish_columns)
    # The mechanism itself is still live and tested against a synthetic
    # column -- see test_val_nll_is_refused_as_a_cross_arm_column above.


# ---------------------------------------------------------------------------
# F7: assert_all_complete -- a NEW, separate guard, deliberately NOT called
# by validate_frame. See its docstring / validate_frame's docstring for why.
# ---------------------------------------------------------------------------

def test_assert_all_complete_passes_when_every_row_is_complete():
    df = pd.DataFrame({"run_id": ["r1", "r2"], "run_complete": [True, True]})
    assert_all_complete(df)


def test_assert_all_complete_raises_on_any_incomplete_row():
    df = pd.DataFrame({"run_id": ["r1", "r2"], "run_complete": [True, False]})
    with pytest.raises(ValueError, match="run_complete|final_results"):
        assert_all_complete(df)


# ---------------------------------------------------------------------------
# F11: empty frame vs. missing column are different failure modes, and both
# are now deliberate, tested decisions rather than accidents.
# ---------------------------------------------------------------------------

def test_validate_frame_passes_vacuously_on_an_empty_frame():
    # Deliberate: an empty frame has nothing to compare, so nothing can be
    # WRONG about it -- every guard here checks a RELATION between rows, and
    # a relation over zero rows is vacuously satisfied (the same way an
    # empty test suite passes). This is different from a MISSING column,
    # which is refused below: "no data" and "we don't know what data would
    # even mean here" are not the same failure.
    empty = pd.DataFrame(columns=list(PROVENANCE_COLUMNS) + ["accuracy"])
    validate_frame(empty)


def test_validate_frame_raises_on_a_missing_required_column():
    df = _full_df_for_validate().drop(columns=["system"])
    with pytest.raises(ValueError, match="missing"):
        validate_frame(df)


def test_assert_matched_budget_raises_a_clear_error_not_a_keyerror_on_a_missing_column():
    df = pd.DataFrame({"n_epochs": [10, 10]})  # no "arm" column at all
    with pytest.raises(ValueError, match="missing"):
        assert_matched_budget(df)


def test_assert_comparable_raises_a_clear_error_not_a_keyerror_on_a_missing_column():
    df = pd.DataFrame({"best_val_nll": [1.0, 2.0]})  # no "arm" column at all
    with pytest.raises(ValueError, match="missing"):
        assert_comparable(df, "best_val_nll")


# ---------------------------------------------------------------------------
# validate_frame: the whole-frame entry point the reporting path calls.
# ---------------------------------------------------------------------------

def _full_df_for_validate(**overrides):
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
    validate_frame(_full_df_for_validate())


def test_validate_frame_raises_on_budget_mismatch():
    with pytest.raises(ValueError, match="budget"):
        validate_frame(_full_df_for_validate(n_epochs=[10, 20]))


def test_validate_frame_raises_on_a_non_comparable_column_across_arms():
    df = _full_df_for_validate(
        arm=["mlp", "gp"], seed=[42, 42],
        commit=[git_sha(), git_sha()], best_val_nll=[1.0, 2.0],
    )
    with pytest.raises(ValueError, match="not comparable across arms"):
        validate_frame(df)


def test_validate_frame_raises_when_an_arm_predates_its_invalidating_commit():
    df = _full_df_for_validate(commit=[_PRE_FIX_COMMIT, _PRE_FIX_COMMIT], seed=[42, 42])
    with pytest.raises(ValueError, match="predates|not found|unknown"):
        validate_frame(df)


def test_validate_frame_passes_when_an_arm_postdates_its_invalidating_commit():
    # Single row: isolates the provenance check from the seed-distinctness
    # and seeding-fix-provenance checks, which need >= 2 distinct seeds.
    df = _full_df_for_validate(
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
    df = _full_df_for_validate(
        arm=["mlp", "mlp"], commit=["597db9f", "597db9f"], accuracy=[0.80, 0.83],
    )
    with pytest.raises(ValueError, match="predates|not found|unknown"):
        validate_frame(df)


def test_validate_frame_raises_when_a_multiseed_arm_reports_identical_metric():
    df = _full_df_for_validate(arm=["mlp", "mlp"], commit=[git_sha(), git_sha()],
                                accuracy=[0.8, 0.8])
    with pytest.raises(ValueError, match="identical"):
        validate_frame(df)


def test_validate_frame_does_not_flag_a_saturated_metric_when_a_sibling_metric_varies():
    df = _full_df_for_validate(arm=["mlp", "mlp"], commit=[git_sha(), git_sha()],
                                accuracy=[1.0, 1.0], tp=[10, 12])
    validate_frame(df)


def test_validate_frame_raises_on_a_lone_saturated_metric_with_no_corroboration():
    # Companion to the test above: with nothing else to check, a tie has no
    # corroborating evidence of real variance and must raise (see the F1
    # redesign section above for the full reasoning).
    df = _full_df_for_validate(arm=["mlp", "mlp"], commit=[git_sha(), git_sha()],
                                accuracy=[1.0, 1.0])
    with pytest.raises(ValueError, match="identical"):
        validate_frame(df)


def test_validate_frame_raises_when_arm_is_null_even_if_the_row_otherwise_looks_fine():
    # F3/F4: a row with unknown arm identity cannot be checked against ANY
    # INVALIDATING_COMMITS entry -- `df[df["arm"] == "bnn_mfvi"]` never
    # matches a null arm -- so a pre-fix commit on a corrupted-identity row
    # would silently sail through every per-arm provenance check if nothing
    # upstream refused it first. Caught here in DEPTH: assert_matched_budget
    # (called first) and assert_distinct_seeds (called last, over the metric
    # columns) each independently null-check "arm" for their own reasons, so
    # this specific frame is refused twice over. A separate, explicit
    # null-arm check written directly in validate_frame itself (a natural
    # first instinct) was mutation-tested against assert_matched_budget's
    # check alone and proved dead -- removed rather than kept as decoration;
    # this test was then re-verified to require BOTH of the real checks
    # disabled at once before it stops catching the bug (see task-4-report.md).
    df = _full_df_for_validate(arm=[None, None], commit=[_PRE_FIX_COMMIT, _PRE_FIX_COMMIT])
    with pytest.raises(ValueError, match="null"):
        validate_frame(df)


def test_validate_frame_raises_when_commit_column_is_missing():
    df = _full_df_for_validate().drop(columns=["commit"])
    with pytest.raises(ValueError, match="missing"):
        validate_frame(df)


def test_validate_frame_raises_when_seed_column_is_missing():
    df = _full_df_for_validate().drop(columns=["seed"])
    with pytest.raises(ValueError, match="missing"):
        validate_frame(df)


def test_validate_frame_allows_legitimately_different_budgets_per_system():
    df = _full_df_for_validate(
        run_id=["r1", "r2"], arm=["fm", "fm"], system=["pendulum", "quad3d"],
        tier=["production", "production"], acquisition=["random", "random"],
        seed=[42, 42], epoch=[9, 9], n_epochs=[19, 30], run_complete=[True, True],
        commit=[git_sha(), git_sha()], accuracy=[0.8, 0.7],
    )
    validate_frame(df)


def test_validate_frame_does_not_require_run_complete_true():
    # Deliberate (F7): run_complete=False is the ROUTINE state for an
    # in-flight ~450-run campaign (aggregate.py's own module docstring), not
    # evidence of a problem by itself. Requiring every row to be finished
    # before validate_frame passes would make it fail on essentially every
    # query against a live campaign. A caller building a FINAL report, where
    # every row really must be settled, calls assert_all_complete explicitly
    # in addition to validate_frame -- see that function's docstring.
    df = _full_df_for_validate(run_complete=[False, False])
    validate_frame(df)
