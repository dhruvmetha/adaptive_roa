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

from stoch_prob_metrics import epoch_dirs, load_ground_truth, score_epoch  # noqa: E402

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


def find_arm(level: str, kind: str) -> Path | None:
    """kind: 'outcome' | 'clf' | 'fm'. Deepest copy wins, Amarel breaks ties."""
    if kind == "outcome":
        return resolve_run(OUTCOME_ROOT / f"fm_outcome_{level}")
    cands = []
    for sub in ("stoch_compare_amarel", "stoch_compare"):
        d = resolve_run(EXP / sub / f"{kind}_{level}_dir00")
        if d is not None:
            cands.append((d, sub != "stoch_compare_amarel"))
    cands.sort(key=lambda t: (-len(epoch_dirs(t[0])), t[1]))
    return cands[0][0] if cands else None


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

    frac = d_fm / span  # 0 = sits on endpoint-FM, 1 = sits on the classifier
    if floor is not None and min(d_fm, d_clf) > floor and 0.25 < frac < 0.75:
        return (f"BOTH CONTRIBUTE (provisional): sits {frac:.0%} of the way from "
                f"endpoint-FM toward the classifier, resolvably far from each")
    if d_fm < d_clf:
        near = "" if floor is None or d_fm > floor else " (within the noise floor of it)"
        return f"MACHINERY (provisional): tracks endpoint-FM{near}, {frac:.0%} of the span from it"
    near = "" if floor is None or d_clf > floor else " (within the noise floor of it)"
    return f"TARGET (provisional): tracks the classifier{near}, {1 - frac:.0%} of the span from it"


def noise_floor(level: str) -> float | None:
    """Run-to-run spread on this metric, from the campaign's seed replicates.

    Uses the fm_*_dir00 seed family at its deepest shared epoch. Returns None
    when fewer than 3 seeds exist, rather than inventing a denominator.
    """
    seeds = [EXP / "stoch_compare_seeds" / f"fm_{level}_dir00_s{s}" for s in (43, 44)]
    base = find_arm(level, "fm")
    runs = [r for r in ([base] + [resolve_run(s) for s in seeds]) if r is not None]
    if len(runs) < 3:
        return None
    ep = matched_epoch({str(i): r for i, r in enumerate(runs)})
    if ep is None:
        return None
    gt = load_ground_truth(DATA / level)
    vals = [score_epoch(r / f"epoch_{ep:03d}", gt, None, None)[KEY] for r in runs]
    return float(2.0 * np.std(vals, ddof=1))


def mc_vs_exact(level: str, epoch: int) -> str | None:
    """Re-score the outcome arm with the MC readout to bound sampling noise.

    Requires rebuilding the model from the epoch checkpoint: OutcomeFlowMatcher
    holds a `system` and a net that Lightning cannot reconstruct on its own.
    """
    import torch
    from adaptive_roa.model.outcome_flow_matcher import OutcomeFlowMatcher, OutcomeVelocityMLP
    from adaptive_roa.systems.pendulum import PendulumSystem

    run = find_arm(level, "outcome")
    if run is None:
        return None
    ep_dir = run / f"epoch_{epoch:03d}"
    ckpts = sorted((ep_dir / "checkpoints").glob("best*.ckpt"))
    if not ckpts:
        return None

    system = PendulumSystem()
    dummy = torch.zeros(1, int(system.state_dim))
    cond_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])

    model = OutcomeFlowMatcher(
        velocity_net=OutcomeVelocityMLP(cond_dim, [256, 512, 256]),
        system=system,
        num_ode_steps=50,
    )
    state = torch.load(ckpts[0], map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=False)
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
    floor = noise_floor(level)

    print(f"\n## {level} — matched epoch {ep}")
    print(f"| arm | debiased Brier | skill | sAUROC | mean p_invalid |")
    print(f"|---|---|---|---|---|")
    for k, label in (("fm", "endpoint FM (generative x state)"),
                     ("outcome", "**outcome FM (generative x binary)**"),
                     ("clf", "classifier (discriminative x binary)")):
        s = scored[k]
        print(f"| {label} | {s[KEY]:.5f} | {s.get('skill_score', float('nan')):.4f} | "
              f"{s.get('soft_auroc', float('nan')):.4f} | {s.get('mean_p_invalid', 0.0):.4f} |")

    print(f"\n    noise floor (2xSD, fm dir00 seeds): "
          f"{'unavailable (<3 seeds)' if floor is None else f'{floor:.5f}'}")
    print(f"    VERDICT: {verdict(scored['outcome'][KEY], scored['clf'][KEY], scored['fm'][KEY], floor)}")

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
