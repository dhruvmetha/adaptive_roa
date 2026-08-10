#!/usr/bin/env python
"""Score the outcome-FM ablation arm against the two arms it sits between.

The point of the arm is attribution, so this script exists to make the
attribution rule mechanical rather than a judgement call made after seeing
numbers. The rule was fixed before any run started (see
docs/superpowers/specs/2026-08-10-outcome-fm-ablation-design.md):

    outcome-FM calibration ~ endpoint-FM  =>  the edge is FM MACHINERY
    outcome-FM calibration ~ classifier   =>  the edge is the FULL-STATE TARGET
    between                                =>  both contribute

"~" is judged against the run-to-run noise floor, never against a paired test
over the ~40k eval cells: that test has a tiny standard error and returns |z|>40
for effects an order of magnitude below run-to-run variation. Without seed
replicates for the outcome arm the verdict is reported as PROVISIONAL, because a
single run cannot distinguish a real gap from the spread the campaign already
measured on this exact quantity.

Usage:
    python scripts/outcome_fm_ablation_report.py                 # all available levels
    python scripts/outcome_fm_ablation_report.py --level low
    python scripts/outcome_fm_ablation_report.py --level low --with-mc-readout
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stoch_prob_metrics import (  # noqa: E402
    epoch_dirs, load_ground_truth, match_to_truth, score_epoch,
)

EXP = Path("/common/users/shared/pracsys/adaptive_roa_experiments")
DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr")
OUTCOME_ROOT = EXP / "outcome_fm"
LEVELS = ("low", "med", "high", "xhigh")

# Headline metric for the attribution. Debiased Brier is the quantity the
# campaign's 6-43x claim was made in, so the ablation has to answer in the same
# units or it is not addressing the same question.
KEY = "brier_debiased"


def resolve_run(root: Path) -> Path | None:
    """Find the directory holding epoch_* dirs.

    The stoch_compare arms keep epoch dirs directly under the run dir, while a
    fresh Hydra run nests them under output_dir/<template>/<timestamp>/. Both
    layouts are in play, so search rather than assume.
    """
    if not root.exists():
        return None
    if epoch_dirs(root):
        return root
    cands = [d for d in root.rglob("epoch_000") if d.is_dir()]
    runs = sorted({d.parent for d in cands}, key=lambda p: (-len(epoch_dirs(p)), str(p)))
    return runs[0] if runs else None


ROOTS = ("stoch_compare_amarel", "stoch_compare", "stoch_compare_seeds")


def _best_copy(name: str) -> Path | None:
    """Deepest copy of a run across all roots, Amarel breaking ties.

    The same run name exists under several roots; taking the first match found
    silently discards deeper data when a preempted Amarel copy is shallower.
    """
    cands = []
    for sub in ROOTS:
        d = resolve_run(EXP / sub / name)
        if d is not None:
            cands.append((d, sub != "stoch_compare_amarel"))
    cands.sort(key=lambda t: (-len(epoch_dirs(t[0])), t[1]))
    return cands[0][0] if cands else None


def seed_runs(level: str, kind: str) -> list[Path]:
    """All seed replicates of an arm: base (s42) plus s43/s44.

    The outcome arm lives under its own root but follows the same naming, so the
    verdict can stop being provisional once it has three seeds of its own.
    """
    if kind == "outcome":
        names = [f"fm_outcome_{level}"] + [f"fm_outcome_{level}_s{s}" for s in (43, 44)]
        return [r for r in (resolve_run(OUTCOME_ROOT / n) for n in names) if r is not None]
    names = [f"{kind}_{level}_dir00"] + [f"{kind}_{level}_dir00_s{s}" for s in (43, 44)]
    return [r for r in (_best_copy(n) for n in names) if r is not None]


def find_arm(level: str, kind: str) -> Path | None:
    """kind: 'outcome' | 'clf' | 'fm'. Deepest copy wins, Amarel breaks ties."""
    if kind == "outcome":
        return resolve_run(OUTCOME_ROOT / f"fm_outcome_{level}")
    return _best_copy(f"{kind}_{level}_dir00")


def matched_epoch(runs: dict[str, Path]) -> int | None:
    """Deepest epoch index present in EVERY arm.

    Comparing arms at different data budgets would confound the ablation with
    training-set size, which is the one thing this design holds fixed.
    """
    common = None
    for d in runs.values():
        idx = {int(p.name.split("_")[1]) for p in epoch_dirs(d)}
        common = idx if common is None else (common & idx)
    return max(common) if common else None


def verdict(outcome: float, clf: float, fm: float, floor: float | None) -> str:
    """Which reference is outcome-FM closer to, and is the gap resolvable?"""
    d_fm, d_clf = abs(outcome - fm), abs(outcome - clf)
    span = abs(clf - fm)
    if span <= 0:
        return "INDETERMINATE: the two reference arms did not separate at this epoch"
    if floor is not None and span < floor:
        return (f"INDETERMINATE: reference arms differ by {span:.5f}, inside the "
                f"{floor:.5f} run-to-run floor -- nothing to attribute")

    lo, hi = min(fm, clf), max(fm, clf)
    weak = " [no noise floor available -- weak]" if floor is None else ""

    # Outside the bracket the "% of span" framing is meaningless (it goes
    # negative), and the finding is different in kind: the arm is not
    # interpolating between the two mechanisms at all.
    if outcome < lo - (floor or 0.0):
        who = "endpoint-FM" if fm < clf else "the classifier"
        return (f"OUTSIDE THE BRACKET (provisional): better calibrated than BOTH references "
                f"(by {lo - outcome:.5f} vs the better one, {who}) -- not an interpolation, "
                f"so the two-factor framing does not explain it{weak}")
    if outcome > hi + (floor or 0.0):
        who = "the classifier" if clf > fm else "endpoint-FM"
        return (f"OUTSIDE THE BRACKET (provisional): worse calibrated than BOTH references "
                f"(by {outcome - hi:.5f} vs the worse one, {who}) -- the scalar target costs "
                f"more than either mechanism explains{weak}")

    frac = d_fm / span  # 0 = sits on endpoint-FM, 1 = sits on the classifier
    if floor is not None and min(d_fm, d_clf) > floor and 0.25 < frac < 0.75:
        return (f"BOTH CONTRIBUTE (provisional): sits {frac:.0%} of the way from "
                f"endpoint-FM toward the classifier, resolvably far from each")
    if d_fm < d_clf:
        near = "" if floor is None or d_fm > floor else " (within the noise floor of it)"
        return f"MACHINERY (provisional): tracks endpoint-FM{near}, {frac:.0%} of the span from it{weak}"
    near = "" if floor is None or d_clf > floor else " (within the noise floor of it)"
    return f"TARGET (provisional): tracks the classifier{near}, {1 - frac:.0%} of the span from it{weak}"


def seed_scores(level: str, kind: str, epoch: int) -> list[float]:
    """Score every seed replicate of a reference arm that reached `epoch`."""
    gt = load_ground_truth(DATA / level)
    out = []
    for r in seed_runs(level, kind):
        d = r / f"epoch_{epoch:03d}"
        if (d / "full_roa_per_point.npz").exists():
            out.append(float(score_epoch(d, gt, None, None)[KEY]))
    return out


def reference_value(level: str, kind: str, epoch: int) -> tuple[float, int, float]:
    """(median across seeds, n seeds, spread) for a reference arm.

    The MEDIAN, not a single arbitrarily-chosen run. Individual runs spike: at
    med epoch 18 the base fm seed reads 0.00318 against siblings at 0.00098 and
    0.00057, and anchoring 'endpoint-FM' on that would bias the attribution
    purely from which copy the discovery happened to pick.
    """
    vals = seed_scores(level, kind, epoch)
    if not vals:
        return float("nan"), 0, float("nan")
    spread = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
    return float(np.median(vals)), len(vals), spread


def noise_floor(level: str, epoch: int) -> tuple[float | None, str]:
    """(2xSD across fm seed replicates at `epoch`, reason if unavailable).

    Returns the reason rather than a bare None: 'fewer than 3 seeds' and 'seeds
    exist but none reached this epoch' call for different responses, and
    collapsing them into one message previously reported '<3 seeds' for levels
    that in fact had all three.

    Strictly within-campaign. `low` and `xhigh` have same-named dir00 seeds under
    ensemble_epistemic/, but that is a DIFFERENT campaign whose configuration
    could not be confirmed equivalent from the stored artifacts. A floor is the
    denominator of every verdict here, so an unverified one would turn a guess
    into a confident claim. See `cross_campaign_floor` for a labelled estimate
    that is reported but never used to decide.
    """
    runs = seed_runs(level, "fm")
    if len(runs) < 3:
        return None, f"only {len(runs)} within-campaign seed run(s) found"
    vals = seed_scores(level, "fm", epoch)
    if len(vals) < 3:
        return None, f"{len(runs)} seeds exist but only {len(vals)} reached epoch {epoch}"
    return float(2.0 * np.std(vals, ddof=1)), ""


def cross_campaign_floor(level: str, epoch: int) -> float | None:
    """Floor estimated from ensemble_epistemic's dir00 seeds. CONTEXT ONLY.

    Never feeds a verdict: those runs come from a different campaign and their
    equivalence to the stoch_compare arms is unverified.
    """
    gt = load_ground_truth(DATA / level)
    vals = []
    for name in (f"fm_{level}_dir00_s43", f"fm_{level}_dir00_s44"):
        r = resolve_run(EXP / "ensemble_epistemic" / name)
        if r is None:
            continue
        d = r / f"epoch_{epoch:03d}"
        if (d / "full_roa_per_point.npz").exists():
            vals.append(float(score_epoch(d, gt, None, None)[KEY]))
    base = seed_scores(level, "fm", epoch)
    vals.extend(base[:1])
    return float(2.0 * np.std(vals, ddof=1)) if len(vals) >= 3 else None


def mc_vs_exact(level: str, epoch: int) -> str | None:
    """Re-score the outcome arm with the MC readout to bound sampling noise.

    Requires rebuilding the model from the epoch checkpoint: OutcomeFlowMatcher
    holds a `system` and a net that Lightning cannot reconstruct on its own.
    """
    import torch
    from adaptive_roa.model.outcome_flow_matcher import OutcomeFlowMatcher
    from adaptive_roa.systems.pendulum import PendulumSystem

    run = find_arm(level, "outcome")
    if run is None:
        return None
    ep_dir = run / f"epoch_{epoch:03d}"
    ckpts = sorted((ep_dir / "checkpoints").glob("best*.ckpt"))
    if not ckpts:
        return None

    # Shapes are read from the checkpoint and the load is strict -- hard-coding
    # dims here would silently analyse an untrained net if the config ever moved.
    model = OutcomeFlowMatcher.from_checkpoint(
        ckpts[0], PendulumSystem(), num_ode_steps=50
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    with np.load(ep_dir / "full_roa_per_point.npz") as z:
        states = torch.as_tensor(z["start_states"].astype(np.float32), device=dev)
        p_pipeline = z["p_success"].astype(np.float64)

    p_mc = model.p_success_mc(states, num_samples=100, num_steps=50).cpu().numpy()
    p_ex, mono = model.p_success_exact(states, num_steps=50)
    p_ex = p_ex.cpu().numpy()

    return (f"    MC vs exact: mean|d|={np.abs(p_mc - p_ex).mean():.5f}  "
            f"max|d|={np.abs(p_mc - p_ex).max():.5f}  "
            f"non-monotone={1 - float(mono.double().mean()):.4%}  "
            f"(exact vs pipeline p: {np.abs(p_ex - p_pipeline).max():.2e})")


def report_level(level: str, with_mc: bool) -> None:
    runs = {k: find_arm(level, k) for k in ("outcome", "clf", "fm")}
    missing = [k for k, v in runs.items() if v is None]
    if missing:
        print(f"\n## {level}: SKIPPED (no runs for {', '.join(missing)})")
        return

    ep = matched_epoch(runs)
    if ep is None:
        print(f"\n## {level}: SKIPPED (arms share no epoch)")
        return

    gt = load_ground_truth(DATA / level)
    scored = {k: score_epoch(v / f"epoch_{ep:03d}", gt, None, None) for k, v in runs.items()}
    floor, floor_why = noise_floor(level, ep)

    # All three arms use the seed median where seeds exist.
    ref = {k: reference_value(level, k, ep) for k in ("fm", "clf", "outcome")}
    n_outcome_seeds = ref["outcome"][1]

    # Mean signed bias is reported alongside Brier because on these arms the
    # classifier's deficit is dominated by systematic over-prediction of success
    # (+0.026/+0.038/+0.138 at low/med/high on dir00) rather than random error,
    # while endpoint-FM is near-unbiased. Brier alone conflates the two, and the
    # ablation's whole question is which mechanism carries that bias.
    gt_starts, gt_p, _, _ = load_ground_truth(DATA / level)
    bias = {}
    for k, v in runs.items():
        with np.load(v / f"epoch_{ep:03d}" / "full_roa_per_point.npz") as z:
            st, ph = z["start_states"], z["p_success"].astype(np.float64)
        bias[k] = float(np.mean(ph - gt_p[match_to_truth(st, gt_starts)]))

    print(f"\n## {level} — matched epoch {ep}")
    print("| arm | debiased Brier | mean bias | seeds | seed spread | skill | sAUROC | mean p_invalid |")
    print("|---|---|---|---|---|---|---|---|")
    for k, label in (("fm", "endpoint FM (generative x state)"),
                     ("outcome", "**outcome FM (generative x binary)**"),
                     ("clf", "classifier (discriminative x binary)")):
        s = scored[k]
        if k in ref and ref[k][1] > 0:
            val, n, spread = ref[k]
            n_s, sp_s = str(n), f"{spread:.5f}" if np.isfinite(spread) else "—"
        else:
            val, n_s, sp_s = s[KEY], "1", "—"
        print(f"| {label} | {val:.5f} | {bias[k]:+.4f} | {n_s} | {sp_s} | "
              f"{s.get('skill_score', float('nan')):.4f} | "
              f"{s.get('soft_auroc', float('nan')):.4f} | {s.get('mean_p_invalid', 0.0):.4f} |")

    print(f"\n    noise floor (2xSD, fm seeds @ep{ep}): "
          f"{f'{floor:.5f}' if floor is not None else f'unavailable — {floor_why}'}")
    if floor is None:
        xc = cross_campaign_floor(level, ep)
        if xc is not None:
            print(f"    (context only, NOT used for the verdict: ensemble_epistemic dir00 "
                  f"seeds give ~{xc:.5f}; different campaign, config equivalence unverified)")
    fm_ref = ref["fm"][0] if ref["fm"][1] > 0 else scored["fm"][KEY]
    clf_ref = ref["clf"][0] if ref["clf"][1] > 0 else scored["clf"][KEY]
    out_ref = ref["outcome"][0] if n_outcome_seeds > 0 else scored["outcome"][KEY]

    v = verdict(out_ref, clf_ref, fm_ref, floor)
    # "provisional" is about the OUTCOME arm's own replication, not the
    # references'. With three of its own seeds the verdict stands on the same
    # footing as the campaign's other n=3 claims, so stop hedging.
    if n_outcome_seeds >= 3 and floor is not None:
        v = v.replace(" (provisional)", "").replace("(provisional): ", "")
        v += f"  [n={n_outcome_seeds} outcome seeds, spread {ref['outcome'][2]:.5f}]"
    else:
        v += f"  [outcome arm has {n_outcome_seeds or 1} seed(s); 3 needed to drop 'provisional']"
    print(f"    VERDICT: {v}")

    # The campaign's separation claim: FM and CLF tie on ranking, differ on
    # calibration. If outcome-FM breaks the tie, that claim is less clean than
    # believed -- worth stating loudly because it was a recorded prediction.
    aurocs = {k: scored[k].get("soft_auroc", float("nan")) for k in scored}
    spread = np.nanmax(list(aurocs.values())) - np.nanmin(list(aurocs.values()))
    print(f"    sAUROC spread across all three arms: {spread:.4f}"
          f"{'  <-- PREDICTION HELD (ranking unaffected)' if spread < 5e-3 else '  <-- PREDICTION BROKEN: ranking moved'}")

    if with_mc:
        line = mc_vs_exact(level, ep)
        print(line if line else "    MC vs exact: unavailable (no checkpoint)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=LEVELS, help="default: every level with runs")
    ap.add_argument("--with-mc-readout", action="store_true",
                    help="rebuild the model and compare MC vs exact readouts (slow)")
    args = ap.parse_args()

    print("# Outcome-FM ablation — target vs machinery")
    print("\nAttribution rule fixed before running; verdicts are PROVISIONAL until the")
    print("outcome arm has seed replicates, since one run cannot beat the run-to-run floor.")
    for lvl in ([args.level] if args.level else LEVELS):
        report_level(lvl, args.with_mc_readout)


if __name__ == "__main__":
    main()
