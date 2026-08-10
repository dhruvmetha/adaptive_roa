import pandas as pd
import pytest

from adaptive_roa.benchmark.aggregate import PROVENANCE_COLUMNS
from adaptive_roa.benchmark.fidelity import FidelityResult
from adaptive_roa.benchmark.guards import validate_frame
from adaptive_roa.benchmark.report import build_report


def _df(**kw):
    base = dict(
        arm=["bnn_mfvi", "bnn_mfvi", "gp_reg", "gp_reg"],
        tier=["production"] * 4, acquisition=["ranked", "random"] * 2,
        seed=[42, 42, 43, 43], n_epochs=[10] * 4, accuracy=[0.91, 0.86, 0.88, 0.84],
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
