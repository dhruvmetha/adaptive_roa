import pandas as pd
import pytest

from adaptive_roa.benchmark.aggregate import PROVENANCE_COLUMNS
from adaptive_roa.benchmark.fidelity import FidelityResult
from adaptive_roa.benchmark.guards import validate_frame
from adaptive_roa.benchmark.report import build_report


def _df(**kw):
    # NOTE: every arm carries the SAME seed here. The brief's original
    # fixture gave bnn_mfvi seed 42 and gp_reg seed 43, which
    # assert_matched_coverage (C3) now correctly refuses: two arms whose
    # means come from different seeds were evaluated on different
    # populations, which is the same category error as one arm skipping the
    # hard system, one level down. The fixture was wrong, not the guard.
    base = dict(
        arm=["bnn_mfvi", "bnn_mfvi", "gp_reg", "gp_reg"],
        tier=["production"] * 4, acquisition=["ranked", "random"] * 2,
        seed=[42] * 4, n_epochs=[10] * 4, accuracy=[0.91, 0.86, 0.88, 0.84],
    )
    base.update(kw)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Step 1 tests, reproduced verbatim from the task brief.
# ---------------------------------------------------------------------------

def test_report_contains_every_arm_and_both_acquisition_modes():
    out = build_report(_df())
    assert "bnn_mfvi" in out and "gp_reg" in out
    assert "ranked" in out and "random" in out


def test_report_runs_the_budget_guard():
    with pytest.raises(ValueError, match="budget"):
        build_report(_df(n_epochs=[10, 10, 20, 20]))


def test_report_refuses_to_mix_tiers_in_one_table():
    with pytest.raises(ValueError, match="tier"):
        build_report(_df(tier=["production", "production", "reference", "reference"]))


def test_withheld_fidelity_prints_the_reason_not_a_number():
    fid = {"bnn_mfvi_reg": FidelityResult(
        available=False, reason="reference did not converge: rhat_max=91.4")}
    out = build_report(_df(), fidelity=fid)
    assert "91.4" in out
    assert "did not converge" in out


def test_available_fidelity_prints_the_number():
    fid = {"bnn_mfvi": FidelityResult(available=True, agreement=0.93,
                                      total_variation=0.04)}
    out = build_report(_df(), fidelity=fid)
    assert "0.93" in out and "0.04" in out


def test_paired_control_delta_is_reported():
    # Foong et al. found MFVI-driven active learning LOSING to random. Without
    # the paired delta surfaced, that finding is invisible.
    out = build_report(_df())
    assert "delta" in out.lower()


# ---------------------------------------------------------------------------
# Beyond the brief: the given 6 tests exercise the tier-mixing refusal and
# assert_matched_budget directly, but NOT assert_comparable or
# assert_distinct_seeds -- with the given fixture, both run as pure no-ops
# (assert_comparable never sees a NON_COMPARABLE_COLUMNS name; every cell
# assert_distinct_seeds inspects has exactly one seed). A guard that is
# CALLED but never actually TRIGGERED by any test is caught by the same
# mutation check the brief asks for ("each guard invocation removed... must
# fail a test") only if some test actually depends on it firing. These close
# that gap.
# ---------------------------------------------------------------------------

def test_report_runs_the_comparability_guard():
    df = _df()
    df["best_val_nll"] = [1.0, 2.0, 3.0, 4.0]  # NON_COMPARABLE_COLUMNS member
    with pytest.raises(ValueError, match="comparable"):
        build_report(df, metric="best_val_nll")


def test_report_runs_the_seed_distinctness_guard():
    # Same arm/acquisition/tier cell, two nominally distinct seeds, but an
    # IDENTICAL metric vector across them -- the exact fingerprint of the
    # seed never reaching the model (see guards.SEEDING_FIX_COMMIT).
    df = pd.DataFrame(dict(
        arm=["bnn_mfvi", "bnn_mfvi"],
        tier=["production", "production"],
        acquisition=["ranked", "ranked"],
        seed=[42, 43],
        n_epochs=[10, 10],
        accuracy=[0.91, 0.91],
    ))
    with pytest.raises(ValueError, match="seed"):
        build_report(df)


def test_mixed_fidelity_dict_keeps_withheld_rows_visibly_withheld():
    # A dict mixing an available result with a withheld one is the ROUTINE
    # shape for a real call against the hmc_reg reference (fidelity.py's
    # module docstring). The withheld row must stay visibly withheld -- not
    # a blank cell that reads as "not measured" rather than "refused".
    fid = {
        "bnn_mfvi": FidelityResult(available=True, agreement=0.93,
                                   total_variation=0.04),
        "bnn_mfvi_reg": FidelityResult(
            available=False,
            reason="reference did not converge: rhat_max=91.4"),
    }
    out = build_report(_df(), fidelity=fid)
    fidelity_lines = out.split("## Posterior fidelity", 1)[1].splitlines()

    available_line = next(l for l in fidelity_lines if l.startswith("| bnn_mfvi "))
    assert "0.93" in available_line and "0.04" in available_line

    withheld_line = next(l for l in fidelity_lines if l.startswith("| bnn_mfvi_reg"))
    assert "withheld" in withheld_line
    assert "91.4" in withheld_line and "did not converge" in withheld_line


def test_missing_metric_column_fails_loudly_with_the_dotted_column_hint():
    df = _df().drop(columns=["accuracy"])
    df["lambda_delta.accuracy"] = [0.91, 0.86, 0.88, 0.84]
    with pytest.raises(ValueError, match="lambda_delta.accuracy"):
        build_report(df)  # default metric="accuracy" is absent from this frame


def test_metric_keyword_selects_a_real_dotted_column():
    df = _df().drop(columns=["accuracy"])
    df["lambda_delta.accuracy"] = [0.91, 0.86, 0.88, 0.84]
    out = build_report(df, metric="lambda_delta.accuracy")
    assert "bnn_mfvi" in out and "0.910" in out


# ---------------------------------------------------------------------------
# C2: never pool across systems. One table per system, never one row.
# ---------------------------------------------------------------------------

def _multi_system_df(**kw):
    base = dict(
        run_id=["r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4"],
        arm=["mlp"] * 4 + ["gp"] * 4,
        system=["pendulum", "pendulum", "quadrotor3d", "quadrotor3d"] * 2,
        tier=["production"] * 8,
        acquisition=["ranked", "direct"] * 4,
        seed=[42] * 8, n_epochs=[10] * 8, epoch=[9] * 8,
        accuracy=[0.95, 0.85, 0.55, 0.50, 0.93, 0.88, 0.60, 0.52],
    )
    base.update(kw)
    return pd.DataFrame(base)


def test_systems_get_their_own_table_and_are_never_averaged_together():
    # pendulum ranked=0.95 and quadrotor3d ranked=0.55 used to be reported as
    # a single 0.802 per arm, with no guard firing: assert_matched_budget is
    # scoped BY system, so a multi-system frame passes it happily.
    out = build_report(_multi_system_df())
    assert "system: pendulum" in out and "system: quadrotor3d" in out
    assert "0.950" in out and "0.550" in out
    assert "0.802" not in out


def test_each_system_section_holds_only_that_system_numbers():
    out = build_report(_multi_system_df())
    pendulum, quad = out.split("system: quadrotor3d", 1)
    assert "0.950" in pendulum and "0.950" not in quad
    assert "0.550" in quad and "0.550" not in pendulum


def test_a_single_system_frame_still_renders_one_table():
    df = _multi_system_df()
    out = build_report(df[df["system"] == "pendulum"])
    assert "system: pendulum" in out
    assert "never averaged together" not in out


# ---------------------------------------------------------------------------
# C3: unequal arm coverage inverts the ranking, and --require-complete cannot
# see it (every run present IS complete; the missing ones have no rows).
# ---------------------------------------------------------------------------

def test_report_refuses_arms_that_did_not_run_on_the_same_systems():
    # gp finished on pendulum only; its mean would "win" against an mlp
    # averaged over pendulum plus the hard system.
    df = _multi_system_df()
    df = df[~((df["arm"] == "gp") & (df["system"] == "quadrotor3d"))]
    with pytest.raises(ValueError, match="do not cover the same"):
        build_report(df)


def test_report_refuses_arms_that_did_not_run_on_the_same_seeds():
    df = _df(seed=[42, 42, 43, 43])
    with pytest.raises(ValueError, match="do not cover the same"):
        build_report(df)


# ---------------------------------------------------------------------------
# I1: acquisition levels are derived from the frame, never hardcoded.
# ---------------------------------------------------------------------------

def test_a_third_acquisition_level_is_rendered_not_discarded():
    # 'entropy' rows used to be aggregated, pass every guard, and then vanish
    # from the output with no mention at all.
    df = pd.DataFrame(dict(
        arm=["mlp"] * 3 + ["gp"] * 3, tier=["production"] * 6,
        acquisition=["ranked", "direct", "entropy"] * 2,
        seed=[42] * 6, n_epochs=[10] * 6,
        accuracy=[0.95, 0.85, 0.90, 0.93, 0.88, 0.91],
    ))
    out = build_report(df)
    assert "entropy" in out
    assert "0.900" in out and "0.910" in out


def test_the_control_column_is_the_real_group_name_not_a_hardcoded_random():
    # The manifest's control group is `direct`; a hardcoded "random" column
    # rendered nan while discarding the control data actually collected.
    out = build_report(_df(acquisition=["ranked", "direct"] * 2))
    assert "direct" in out
    assert "nan" not in out.lower()
    assert "delta (ranked − direct)" in out


def test_no_delta_is_reported_and_said_so_when_no_control_level_exists():
    df = _df(acquisition=["ranked", "entropy"] * 2)
    out = build_report(df)
    assert "No paired delta is reported" in out
    assert "entropy" in out


def test_an_ambiguous_control_is_reported_as_no_delta_not_guessed():
    df = pd.DataFrame(dict(
        arm=["mlp"] * 3 + ["gp"] * 3, tier=["production"] * 6,
        acquisition=["ranked", "direct", "random"] * 2,
        seed=[42] * 6, n_epochs=[10] * 6,
        accuracy=[0.95, 0.85, 0.84, 0.93, 0.88, 0.87],
    ))
    out = build_report(df)
    assert "No paired delta is reported" in out
    # ...and both controls are still tabulated.
    assert "0.850" in out and "0.840" in out


def test_an_explicit_control_disambiguates():
    df = pd.DataFrame(dict(
        arm=["mlp"] * 3 + ["gp"] * 3, tier=["production"] * 6,
        acquisition=["ranked", "direct", "random"] * 2,
        seed=[42] * 6, n_epochs=[10] * 6,
        accuracy=[0.95, 0.85, 0.84, 0.93, 0.88, 0.87],
    ))
    out = build_report(df, control="direct")
    assert "delta (ranked − direct)" in out
    assert "+0.100" in out


def test_an_explicit_control_that_was_never_collected_is_refused():
    with pytest.raises(ValueError, match="never collected|not an acquisition level"):
        build_report(_df(), control="entropy")


# ---------------------------------------------------------------------------
# I2: the epoch reduction is explicit, selectable, and stated in the output.
# ---------------------------------------------------------------------------

def _curve_df():
    # Ranked and the control are near-identical at epoch 0 (adaptive
    # acquisition has not selected a point yet) and separate by the last
    # epoch -- the shape that makes a curve-mean dilute the headline delta.
    rows = []
    for epoch in range(3):
        for arm in ("mlp", "gp"):
            rows.append(dict(run_id=f"{arm}_ranked", arm=arm, tier="production",
                             system="pendulum", acquisition="ranked", seed=42,
                             n_epochs=3, epoch=epoch,
                             accuracy=0.80 + 0.05 * epoch))
            rows.append(dict(run_id=f"{arm}_direct", arm=arm, tier="production",
                             system="pendulum", acquisition="direct", seed=42,
                             n_epochs=3, epoch=epoch,
                             accuracy=0.80 + 0.01 * epoch))
    return pd.DataFrame(rows)


def test_the_default_report_uses_the_final_epoch_of_each_run():
    out = build_report(_curve_df())
    assert "FINAL epoch of each run" in out
    assert "+0.080" in out          # 0.90 - 0.82, the final-epoch delta


def test_pooling_the_whole_curve_is_available_and_labelled_as_such():
    out = build_report(_curve_df(), epochs="all")
    assert "ALL epochs" in out
    assert "epoch 0" in out
    assert "+0.040" in out          # 0.85 - 0.81, the diluted delta
    assert "+0.080" not in out


def test_a_single_epoch_can_be_selected():
    out = build_report(_curve_df(), epochs=0)
    assert "epoch 0 only" in out
    assert "+0.000" in out


def test_the_epoch_selection_is_always_stated_in_the_output():
    for epochs in ("final", "all", 1):
        assert "Epoch selection" in build_report(_curve_df(), epochs=epochs)


def test_selecting_an_epoch_no_run_reached_is_refused():
    with pytest.raises(ValueError, match="selects no rows"):
        build_report(_curve_df(), epochs=99)


def test_an_unknown_epoch_selection_is_refused():
    with pytest.raises(ValueError, match="not a valid epoch selection"):
        build_report(_curve_df(), epochs="last")


def test_the_final_epoch_is_taken_per_run_not_frame_wide():
    # A preempted run that stopped at epoch 1 must contribute its OWN last
    # epoch, not be dropped because another run reached epoch 2.
    df = _curve_df()
    df = df[~((df["run_id"] == "gp_ranked") & (df["epoch"] == 2))]
    out = build_report(df)
    assert "epochs contributing: [1, 2]" in out
    assert "| gp |" in out


def test_a_frame_without_an_epoch_column_says_no_reduction_was_applied():
    out = build_report(_df())
    assert "no `epoch` column" in out


# ---------------------------------------------------------------------------
# n_epochs_collected: a run that lost epochs is named, not quietly averaged.
# ---------------------------------------------------------------------------

def test_a_run_short_of_its_budget_is_named_in_the_report():
    df = _multi_system_df(n_epochs_collected=[10, 10, 10, 10, 10, 10, 8, 8])
    out = build_report(df)
    assert "fewer epochs than configured" in out
    assert "`r4` 8/10" in out


def test_a_campaign_with_no_lost_epochs_says_so():
    df = _multi_system_df(n_epochs_collected=[10] * 8,
                          n_epochs_evaluated=[10] * 8)
    out = build_report(df)
    assert "full configured number of epochs" in out


def test_epochs_that_produced_no_metrics_are_reported_separately():
    # NEW-3: n_epochs_collected alone cannot see this. The rows exist and
    # parse, so the run looks complete, while a third of its metric
    # population is NaN and silently skipped by the aggregation.
    df = _multi_system_df(n_epochs_collected=[10] * 8,
                          n_epochs_evaluated=[10, 10, 10, 10, 10, 10, 6, 6])
    out = build_report(df)
    assert "produced NO metrics" in out
    assert "`r4` 6 of 10 epochs evaluated" in out
    assert "full configured number of epochs" not in out


def test_both_kinds_of_lost_epoch_are_reported_together():
    df = _multi_system_df(n_epochs_collected=[10, 10, 10, 10, 10, 10, 8, 8],
                          n_epochs_evaluated=[10, 10, 10, 10, 6, 6, 8, 8])
    out = build_report(df)
    assert "fewer epochs than configured" in out
    assert "produced NO metrics" in out


# ---------------------------------------------------------------------------
# NEW-2: a report generated with validation disabled must say so IN THE
# ARTIFACT. The artifact is what gets saved, read weeks later and pasted into
# a thread; a warning that lives only in a terminal scrollback is not
# attached to it.
# ---------------------------------------------------------------------------

def test_a_report_built_without_provenance_checks_says_so_in_its_own_text():
    out = build_report(_df(), provenance_checked=False)
    assert "PROVENANCE NOT CHECKED" in out
    assert "--no-validate" in out
    assert "INVALIDATING_COMMITS" in out


def test_the_disclosure_sits_above_every_number():
    out = build_report(_df(), provenance_checked=False)
    banner = out.index("PROVENANCE NOT CHECKED")
    assert banner < out.index("Downstream task")


def test_a_validated_report_carries_no_such_banner():
    assert "PROVENANCE NOT CHECKED" not in build_report(_df())


# ---------------------------------------------------------------------------
# m2: a missing identity column is diagnosed, not a bare KeyError.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("column", ["tier", "acquisition", "arm"])
def test_a_missing_identity_column_is_refused_with_a_diagnosis(column):
    with pytest.raises(ValueError, match="missing required column"):
        build_report(_df().drop(columns=[column]))


# ---------------------------------------------------------------------------
# I5: the near-boundary section build_report renders from pointwise rows.
# ---------------------------------------------------------------------------

def _conditioned_rows(refused=None):
    return [dict(system="pendulum", arm="bnn_mfvi", acquisition="ranked", k=5,
                 n_runs=3, overall=0.91, near_boundary=0.64, interior=0.99,
                 n_near=120, n_interior=880, refused=refused)]


def test_the_conditioned_section_reports_the_boundary_split():
    out = build_report(_df(), conditioned=_conditioned_rows())
    assert "Near-boundary conditioned accuracy" in out
    assert "0.640" in out and "0.990" in out


def test_the_conditioned_section_states_it_is_not_the_calibrated_rule():
    # These accuracies are thresholded at 0.5, not at lambda/delta, so they
    # are a different quantity from the headline table and must say so.
    out = build_report(_df(), conditioned=_conditioned_rows())
    assert "0.5" in out and "not a decomposition of it" in out


def test_a_refused_conditioned_row_is_printed_verbatim():
    out = build_report(_df(), conditioned=_conditioned_rows(
        refused="r7: no full_roa_per_point.npz at /x/r7/epoch_009"))
    assert "Runs refused while computing the band" in out
    assert "no full_roa_per_point.npz" in out


def test_no_conditioned_rows_means_no_conditioned_section():
    assert "Near-boundary" not in build_report(_df())


def test_build_report_does_not_require_full_provenance_columns():
    # validate_frame (Task 4) hard-requires every PROVENANCE_COLUMNS entry
    # (run_id, system, epoch, run_complete, commit, ...). This fixture --
    # the same shape every test in this module uses -- is missing several of
    # them, on purpose: build_report calling validate_frame INSTEAD of the
    # three named guards would break every test in this file, not just this
    # one, and would refuse to report on a still-launching campaign slice
    # (Task 8, step 4) before every row's provenance is fully resolved.
    df = _df()
    missing = [c for c in PROVENANCE_COLUMNS if c not in df.columns]
    assert missing, "fixture is expected to be short of full provenance"
    with pytest.raises(ValueError, match="missing required column"):
        validate_frame(df)
    build_report(df)  # must still succeed despite the identical gap
