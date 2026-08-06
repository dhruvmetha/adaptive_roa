#!/usr/bin/env python
"""Discover every arm of the stochastic-pendulum comparison and report it.

Arms live in two places: the four FM non-adaptive baselines predate this
campaign and sit under their original per-level output trees, while everything
launched for the comparison sits under stoch_compare/{pred}_{level}_{arm}.

Because the runs are still training when this is called, comparisons are made
at the deepest epoch every arm of a given (predictor, level) has reached — epoch
index is the acquisition budget, so a matched epoch is a matched amount of data
regardless of which GPU an arm landed on.

Usage:
    python scripts/stoch_compare_report.py --out docs/stoch_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stoch_prob_metrics import (
    epoch_dirs,
    load_ground_truth,
    match_to_truth,
    paired_delta,
    score_epoch,
)

EXP = Path("/common/users/shared/pracsys/adaptive_roa_experiments")
DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr")
LEVELS = ["low", "med", "high", "xhigh"]
ARM_LABEL = {
    "dir00": "non-adaptive (d2=0)",
    "ent05": "entropy d2=0.5",
    "ent10": "entropy d2=1.0",
    "tb05": "entropy+modesep d2=0.5",
    "tb10": "entropy+modesep d2=1.0",
}
ARM_ORDER = ["dir00", "ent05", "ent10", "tb05", "tb10"]

# Headline columns: error, then what drives it, then ranking, then the
# threshold view kept for continuity with the earlier tables.
COLS = [
    ("brier_debiased", "Brier_deb", "{:+.5f}"),
    ("skill_score", "SS", "{:.4f}"),
    ("REL_debiased", "REL_deb", "{:.5f}"),
    ("RES", "RES", "{:.5f}"),
    ("UNC_debiased", "UNC_deb", "{:.5f}"),
    ("sAUROC", "sAUROC", "{:.4f}"),
    ("SHARP", "SHARP", "{:.4f}"),
    ("SHARP_star", "SHARP*", "{:.4f}"),
    ("AURC", "AURC", "{:+.5f}"),
    ("KL", "KL", "{:.4f}"),
    ("f1@0.5", "F1@.5", "{:.4f}"),
    ("mean_p_invalid", "p_inv", "{:.3f}"),
]

# Second table: the threshold sweep and the selective-risk curve, which only
# make sense read across several levels of beta / coverage.
COLS2 = [
    ("f1@0.25", "F1@.25", "{:.4f}"),
    ("f1@0.5", "F1@.5", "{:.4f}"),
    ("f1@0.75", "F1@.75", "{:.4f}"),
    ("acc@0.5", "acc@.5", "{:.4f}"),
    ("roa_frac_true@0.5", "RoA_true", "{:.4f}"),
    ("roa_frac_pred@0.5", "RoA_pred", "{:.4f}"),
    ("risk@0.2", "risk@20%", "{:+.5f}"),
    ("risk@0.5", "risk@50%", "{:+.5f}"),
    ("risk@1", "risk@100%", "{:+.5f}"),
    ("log_score", "logS", "{:.4f}"),
    ("log_score_oracle", "logS_orac", "{:.4f}"),
    ("MAE", "MAE", "{:.4f}"),
]


def july_baseline(lvl: str):
    """The pre-campaign FM non-adaptive run, used only if no fresh control exists.

    It was trained months earlier on different GPUs, so it does NOT reproduce the
    campaign's epoch-0 model and is not a matched control -- see the control
    check in the report.
    """
    base = EXP / f"adaptive_pendulum_stoch_{lvl}" / "outputs"
    cands = [d for d in base.glob("*d2_ratio_0_*sampling_mode_direct/2026-*")
             if epoch_dirs(d)]
    return max(cands, key=lambda d: len(epoch_dirs(d))) if cands else None


def discover() -> list[dict]:
    """Locate one run per (predictor, level, arm), matched on hardware where possible.

    Training is not bit-reproducible across GPU architectures, so arms are only
    comparable within one hardware family. Amarel (L40S) hosts a copy of every
    arm, so it is preferred; the iLab/direct copies are a fallback, and the
    pre-campaign July baseline is a last resort that the control check will flag.
    """
    arms: list[dict] = []
    for lvl in LEVELS:
        for pred in ("fm", "clf"):
            for arm in ARM_ORDER:
                # Take the deeper copy, tie-breaking to Amarel. Amarel is faster
                # but its jobs are preemptible, so its copy is sometimes the
                # shallower one; preferring it unconditionally would throw away
                # better data. Hardware is no longer a tie-breaker worth paying
                # depth for, because run-to-run variation is measured directly
                # by the epoch-0 noise floor rather than assumed away.
                cands = [(root / f"{pred}_{lvl}_{arm}", src) for root, src in (
                    (EXP / "stoch_compare_amarel", "amarel"),
                    (EXP / "stoch_compare", "ilab/direct"))]
                cands = [(d, s) for d, s in cands if d.exists() and epoch_dirs(d)]
                cands.sort(key=lambda t: (-len(epoch_dirs(t[0])), t[1] != "amarel"))
                if not cands and pred == "fm" and arm == "dir00":
                    b = july_baseline(lvl)
                    if b is not None:
                        cands = [(b, "july-baseline (UNMATCHED hardware)")]
                if cands:
                    d, src = cands[0]
                    arms.append({"predictor": pred, "level": lvl, "arm": arm,
                                 "run_dir": d, "source": src})
    return arms


def fmt(v, spec: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return spec.format(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/stoch_compare"))
    ap.add_argument("--all-epochs", action="store_true",
                    help="score every epoch (default: matched epoch + final only)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arms = discover()
    print(f"discovered {len(arms)} arms")
    gt_cache: dict[str, tuple] = {}
    depth = {}
    for a in arms:
        a["epochs"] = [int(d.name.split("_")[1]) for d in epoch_dirs(a["run_dir"])]
        depth[(a["predictor"], a["level"], a["arm"])] = max(a["epochs"]) if a["epochs"] else -1

    rows = []
    for a in arms:
        lvl = a["level"]
        if lvl not in gt_cache:
            gt_cache[lvl] = load_ground_truth(DATA / lvl)
        eds = epoch_dirs(a["run_dir"])
        if not args.all_epochs:
            # matched epoch for this (predictor, level) plus each arm's own tip
            peers = [d for (p, l, _), d in depth.items() if p == a["predictor"] and l == lvl]
            matched = min(peers) if peers else -1
            keep = {matched, max(a["epochs"])}
            eds = [d for d in eds if int(d.name.split("_")[1]) in keep]
        for ed in eds:
            try:
                r = score_epoch(ed, gt_cache[lvl], None, None)
            except Exception as exc:
                print(f"  !! {a['predictor']}/{lvl}/{a['arm']}/{ed.name}: {exc}", flush=True)
                continue
            r.update(predictor=a["predictor"], level=lvl, arm=a["arm"],
                     epoch=int(ed.name.split("_")[1]), run_dir=str(a["run_dir"]))
            rows.append(r)
    print(f"scored {len(rows)} epochs")

    (args.out / "metrics.json").write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        keys = ["predictor", "level", "arm", "epoch"] + sorted(
            {k for r in rows for k in r} - {"predictor", "level", "arm", "epoch"})
        with (args.out / "metrics.csv").open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    # ---- markdown ----
    md = ["# Stochastic pendulum: adaptive vs non-adaptive, FM vs classifier", ""]
    md.append("Ground truth is the 158x315 eval grid, M=90 rollouts per cell. Model "
              "probabilities are the raw p(success) (never renormalised against "
              "p_invalid, which would destroy calibration). Every squared-error "
              "quantity is debiased on both sides: the model's K-sample noise and "
              "the grid's M-sample noise are subtracted, so a perfect model scores 0.")
    md.append("")
    md.append("`Brier_deb` lower is better; `SS` (skill vs the climatology of the true "
              "field) and `sAUROC` higher is better. `REL_deb` is miscalibration, `RES` "
              "resolution, `UNC_deb` the irreducible spread of the true field. `AURC` "
              "is the area under the debiased risk-coverage curve (selective risk where "
              "the model is most confident).")
    md.append("")

    md.extend(headline(rows, arms, depth))
    md.append("## Depth reached (epoch index = acquisition budget)")
    md.append("")
    md.append("| predictor | level | " + " | ".join(ARM_LABEL[a] for a in ARM_ORDER) + " |")
    md.append("|---|---|" + "---|" * len(ARM_ORDER))
    for pred in ("fm", "clf"):
        for lvl in LEVELS:
            cells = []
            for arm in ARM_ORDER:
                d = depth.get((pred, lvl, arm))
                cells.append(str(d) if d is not None and d >= 0 else "-")
            md.append(f"| {pred} | {lvl} | " + " | ".join(cells) + " |")
    md.append("")

    md.extend(noise_floor_section(rows, arms))
    md.extend(predictor_comparison_section(rows))
    md.extend(dose_response_section(rows))

    for pred in ("fm", "clf"):
        md.append(f"## {'Flow matching' if pred == 'fm' else 'Classifier'}")
        md.append("")
        for lvl in LEVELS:
            sub = [r for r in rows if r["predictor"] == pred and r["level"] == lvl]
            if not sub:
                continue
            peers = [d for (p, l, _), d in depth.items() if p == pred and l == lvl]
            matched = min(peers) if peers else -1
            at = [r for r in sub if r["epoch"] == matched]
            if not at:
                continue
            n = at[0]["n_points"]
            md.append(f"### {lvl} — matched epoch {matched} (n={n})")
            md.append("")
            if matched == 0:
                md.append("> Epoch 0 precedes the first acquisition, so every arm here is "
                          "the same model trained on the same random initial pool. Identical "
                          "rows are the expected result and confirm the arms share "
                          "initialisation; arms only diverge from epoch 1 onward.")
                md.append("")
            md.append("| arm | " + " | ".join(c[1] for c in COLS) + " |")
            md.append("|---|" + "---|" * len(COLS))
            for arm in ARM_ORDER:
                r = next((x for x in at if x["arm"] == arm), None)
                if r is None:
                    continue
                md.append(f"| {ARM_LABEL[arm]} | "
                          + " | ".join(fmt(r.get(k), s) for k, _, s in COLS) + " |")
            md.append("")
            md.append("| arm | " + " | ".join(c[1] for c in COLS2) + " |")
            md.append("|---|" + "---|" * len(COLS2))
            for arm in ARM_ORDER:
                r = next((x for x in at if x["arm"] == arm), None)
                if r is None:
                    continue
                md.append(f"| {ARM_LABEL[arm]} | "
                          + " | ".join(fmt(r.get(k), s) for k, _, s in COLS2) + " |")
            md.append("")
            md.extend(paired_table(arms, pred, lvl, matched, gt_cache[lvl], rows))
            md.extend(pairwise_deep_table(arms, pred, lvl, gt_cache[lvl], rows, depth))
    (args.out / "report.md").write_text("\n".join(md))
    print(f"wrote {args.out}/report.md")


def headline(rows: list[dict], arms: list[dict], depth: dict) -> list[str]:
    """State the answer, and whether it is allowed to be believed yet.

    Ordering arms by score is easy and usually meaningless here: the arms sit
    close together, so the summary leads with whether the comparison is even
    valid (matched epoch-0 control) before naming a winner.
    """
    floor = epoch0_noise_floor(rows)
    out = ["## Summary", "",
           "Each line asks one question: does the best adaptive arm beat the "
           "non-adaptive arm by more than this system's own run-to-run noise?", ""]
    for pred in ("fm", "clf"):
        name = "Flow matching" if pred == "fm" else "Classifier"
        lines = []
        for lvl in LEVELS:
            peers = [d for (p, l, _), d in depth.items() if p == pred and l == lvl]
            if not peers:
                continue
            ep = min(peers)
            if ep < 1:
                continue
            at = {r["arm"]: r["brier_debiased"] for r in rows
                  if r["predictor"] == pred and r["level"] == lvl and r["epoch"] == ep}
            if "dir00" not in at or len(at) < 2:
                continue
            adaptive = {a: v for a, v in at.items() if a != "dir00"}
            if not adaptive:
                continue
            best = min(adaptive, key=adaptive.get)
            gap = adaptive[best] - at["dir00"]
            f = floor.get((pred, lvl))
            sd = f["sd"] if f else 0.0
            n = f["n"] if f else 0
            if sd < 1e-9:
                verdict = "floor from seed replicates (see seed_variance.md)"
            elif abs(gap) > 2 * sd:
                verdict = (f"**exceeds** the noise floor (2×SD={2 * sd:.5f}, n={n}) — "
                           + ("adaptive better" if gap < 0 else "adaptive WORSE"))
            else:
                verdict = (f"within the noise floor (2×SD={2 * sd:.5f}, n={n}) — "
                           f"not distinguishable")
            lines.append(f"    - {lvl} ep{ep}: best adaptive = {ARM_LABEL[best]}, "
                         f"gap vs non-adaptive {gap:+.5f} → {verdict}")
        if lines:
            out += [f"- **{name}**", ""] + lines + [""]
    out += ["Rankings alone are not evidence here: the arms sit close together, so a gap "
            "must clear both the paired test (eval-grid noise) and the run-to-run floor "
            "above / `seed_variance.md` before it means anything.", ""]
    return out


ARM_D2 = {"dir00": 0.0, "ent05": 0.5, "tb05": 0.5, "ent10": 1.0, "tb10": 1.0}
# Minimum number of shared epochs before a monotone-in-d2 ordering is asserted.
_MIN_TAIL_EPOCHS = 3
# Every consecutive d2 step must be at least this fraction of the total span,
# so a 'monotone' chain cannot rest on a step that is effectively zero.
_MIN_STEP_FRAC = 0.15


def dose_response_section(rows: list[dict]) -> list[str]:
    """Does the effect scale with how much of the budget acquisition controls?

    A single-epoch gap can be luck. A gap that grows monotonically with the
    adaptive fraction d2, and keeps growing as more data arrives, cannot easily
    be: run-to-run noise has no reason to order itself by d2 at two noise levels
    across many epochs. `slope` is the least-squares trend of debiased Brier over
    epochs >= 1, so positive means the arm degrades as it acquires more.
    """
    floor = epoch0_noise_floor(rows)
    out = ["## Dose-response: does more acquisition make it worse (or better)?", "",
           "Grouped by the fraction of each epoch's budget chosen by the acquisition "
           "rule. A monotone ordering in d2 that persists over many epochs is much "
           "stronger evidence than any single-epoch gap, because seed noise has no "
           "reason to sort itself by d2. An ordering is only asserted when every step "
           "carries real weight (each ≥15% of the span) and the whole span clears this "
           "level's run-to-run noise floor.", "",
           "| predictor | level | d2 | arms | mean Brier (last 3 common epochs) | slope/epoch |",
           "|---|---|---|---|---|---|"]
    for pred in ("fm", "clf"):
        for lvl in LEVELS:
            sub = [r for r in rows if r["predictor"] == pred and r["level"] == lvl]
            if not sub:
                continue
            by_arm: dict[str, dict[int, float]] = {}
            for r in sub:
                by_arm.setdefault(r["arm"], {})[r["epoch"]] = r["brier_debiased"]
            if len(by_arm) < 2:
                continue
            common = sorted(set.intersection(*[set(v) for v in by_arm.values()]))
            tail = [e for e in common if e >= 1][-3:]
            if not tail:
                continue
            groups: dict[float, list[str]] = {}
            for arm in by_arm:
                groups.setdefault(ARM_D2.get(arm, -1.0), []).append(arm)
            means, ok_rows = {}, []
            for d2 in sorted(groups):
                arms_here = groups[d2]
                vals = [np.mean([by_arm[a][e] for e in tail]) for a in arms_here]
                slopes = []
                for a in arms_here:
                    eps = sorted(e for e in by_arm[a] if e >= 1)
                    if len(eps) >= 3:
                        slopes.append(float(np.polyfit(eps, [by_arm[a][e] for e in eps], 1)[0]))
                means[d2] = float(np.mean(vals))
                sl = f"{np.mean(slopes):+.5f}" if slopes else "n/a"
                ok_rows.append(f"| {pred} | {lvl} | {d2:.1f} | {','.join(sorted(arms_here))} | "
                               f"{means[d2]:+.5f} | {sl} |")
            out += ok_rows
            ordered = sorted(means)
            # A monotone ordering read off a single shared epoch is precisely the
            # fragile one-point claim this section exists to replace, so require
            # the ordering to hold over a span of epochs before asserting it.
            if len(ordered) >= 3 and len(tail) >= _MIN_TAIL_EPOCHS:
                vals = [means[d] for d in ordered]
                steps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
                span = abs(vals[-1] - vals[0])
                # Ordering alone is too weak: a chain can be "monotone" while one
                # step is a rounding error, which just restates the d2=0 vs d2>0
                # gap. Demand every step carry real weight, and demand the whole
                # span clear this level's run-to-run noise.
                balanced = span > 0 and all(abs(s) >= _MIN_STEP_FRAC * span for s in steps)
                f = floor.get((pred, lvl))
                thr = 2 * f["sd"] if f and f["sd"] > 0 else None
                clears = thr is None or span > thr
                why = []
                if not balanced:
                    why.append(f"smallest step is {min(abs(s) for s in steps) / span:.0%} "
                               f"of the span")
                if not clears:
                    why.append(f"span {span:.5f} < noise floor {thr:.5f}")
                if all(s > 0 for s in steps) and balanced and clears:
                    out.append(f"| | | | **monotone ↑ in d2** | more acquisition = worse "
                               f"| epochs {tail[0]}–{tail[-1]} |")
                elif all(s < 0 for s in steps) and balanced and clears:
                    out.append(f"| | | | **monotone ↓ in d2** | more acquisition = better "
                               f"| epochs {tail[0]}–{tail[-1]} |")
                elif why:
                    out.append(f"| | | | _ordering not asserted_ | {'; '.join(why)} "
                               f"| epochs {tail[0]}–{tail[-1]} |")
            elif len(ordered) >= 3:
                out.append(f"| | | | _ordering not asserted_ | only {len(tail)} shared "
                           f"epoch(s), need {_MIN_TAIL_EPOCHS} | epochs {tail[0]}–{tail[-1]} |")
    return out + [""]


def predictor_comparison_section(rows: list[dict]) -> list[str]:
    """Flow matching vs classifier, at equal data budget.

    Compared on the non-adaptive arm so the acquisition strategy is held fixed
    and only the predictor differs. Both are scored against the same grid with
    the same debiasing, so the numbers are directly comparable despite FM's
    probability coming from K MC samples and the classifier's from one forward
    pass.
    """
    out = ["## Flow matching vs classifier (non-adaptive arm, equal budget)", "",
           "| level | epoch | FM Brier_deb | CLF Brier_deb | FM SS | CLF SS | "
           "FM sAUROC | CLF sAUROC | winner |", "|---|---|---|---|---|---|---|---|---|"]
    any_row = False
    for lvl in LEVELS:
        f = {r["epoch"]: r for r in rows
             if r["predictor"] == "fm" and r["level"] == lvl and r["arm"] == "dir00"}
        c = {r["epoch"]: r for r in rows
             if r["predictor"] == "clf" and r["level"] == lvl and r["arm"] == "dir00"}
        common = sorted(set(f) & set(c))
        if not common:
            continue
        ep = max(common)
        fr, cr = f[ep], c[ep]
        win = "FM" if fr["brier_debiased"] < cr["brier_debiased"] else "classifier"
        ratio = (cr["brier_debiased"] / fr["brier_debiased"]
                 if fr["brier_debiased"] > 0 else float("nan"))
        tag = f"**{win}**" + (f" ({ratio:.1f}× lower)" if np.isfinite(ratio) and ratio > 1
                              else "")
        out.append(f"| {lvl} | {ep} | {fr['brier_debiased']:+.5f} | "
                   f"{cr['brier_debiased']:+.5f} | {fr['skill_score']:.4f} | "
                   f"{cr['skill_score']:.4f} | {fr['sAUROC']:.4f} | {cr['sAUROC']:.4f} "
                   f"| {tag} |")
        any_row = True
    return out + [""] if any_row else []


def epoch0_noise_floor(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Run-to-run noise floor, measured for free at epoch 0.

    Epoch 0 precedes the first acquisition, so every arm of a (predictor, level)
    is the SAME configuration trained on the SAME data. Their spread is
    therefore a direct estimate of run-to-run variability, and it is not zero for
    flow matching: PyTorch training and the K-sample MC evaluation are not
    bit-reproducible across runs or GPU architectures. That makes a bit-identical
    control unattainable rather than merely missing, so the honest test is
    whether a later arm gap exceeds this floor.

    The classifier is deterministic here (spread exactly 0), so its floor has to
    come from the seed replicates instead.
    """
    floor: dict[tuple[str, str], dict] = {}
    for pred in ("fm", "clf"):
        for lvl in LEVELS:
            vals = np.array([r["brier_debiased"] for r in rows
                             if r["predictor"] == pred and r["level"] == lvl
                             and r["epoch"] == 0])
            if len(vals) >= 2:
                floor[(pred, lvl)] = {
                    "n": int(len(vals)),
                    "sd": float(vals.std(ddof=1)),
                    "spread": float(vals.max() - vals.min()),
                }
    return floor


def noise_floor_section(rows: list[dict], arms: list[dict]) -> list[str]:
    floor = epoch0_noise_floor(rows)
    out = ["## Run-to-run noise floor (measured at epoch 0)", "",
           "Epoch 0 is pre-acquisition, so every arm of a row below is the *same* "
           "configuration on the *same* data. Any spread between them is pure "
           "run-to-run variability. Flow matching is not bit-reproducible (training "
           "kernels and the K-sample MC evaluation both vary), so a bit-identical "
           "control cannot be built — instead, an arm gap later on is only "
           "meaningful if it exceeds this floor. The classifier is deterministic "
           "here, so its floor comes from the seed replicates instead.", "",
           "Treat these SDs as rough: they come from only 4-5 runs, and the flow-matching "
           "estimate varies by an order of magnitude across levels, which is itself a sign "
           "the sample is small. The seed replicates are the better-powered floor.", "",
           "| predictor | level | runs | SD at epoch 0 | max spread | 2×SD threshold |",
           "|---|---|---|---|---|---|"]
    for pred in ("fm", "clf"):
        for lvl in LEVELS:
            f = floor.get((pred, lvl))
            if not f:
                continue
            thr = ("deterministic — use seed replicates"
                   if f["sd"] < 1e-9 else f"{2 * f['sd']:.5f}")
            out.append(f"| {pred} | {lvl} | {f['n']} | {f['sd']:.5f} | "
                       f"{f['spread']:.5f} | {thr} |")
    return out + [""]


def pairwise_deep_table(arms: list[dict], pred: str, lvl: str, gt: tuple,
                        rows: list[dict], depth: dict) -> list[str]:
    """Compare each arm to the baseline at ITS OWN deepest shared epoch.

    The global matched epoch is the minimum across all arms, so one arm that
    started late (a rescue after preemption, say) costs every other arm depth.
    Comparing pairwise instead lets each arm be judged as deep as it and the
    baseline both go, at the cost of different rows using different epochs --
    which is why the epoch is printed per row.
    """
    base = next((a for a in arms if a["predictor"] == pred and a["level"] == lvl
                 and a["arm"] == "dir00"), None)
    if base is None:
        return []
    d_base = depth.get((pred, lvl, "dir00"), -1)
    k = 100.0 if pred == "fm" else None

    floor = epoch0_noise_floor(rows).get((pred, lvl))
    thr = 2 * floor["sd"] if floor else None
    lines = ["Pairwise deepest comparison (each row at its own deepest shared epoch):", ""]
    if thr and thr > 0:
        lines += [f"> The `z` column counts **only** eval-grid noise, so it is badly "
                  f"overconfident: it treats one training run as the whole story. This "
                  f"level's run-to-run floor is 2×SD = {thr:.5f}, and a Δ smaller than that "
                  f"is indistinguishable from rerunning the same arm with a different seed, "
                  f"no matter how large |z| looks. The last column applies that test.", ""]
    lines += ["| arm | epoch | Δ debiased Brier | SE | z | beats run-to-run floor? |",
              "|---|---|---|---|---|---|"]
    any_row = False
    for arm in ARM_ORDER:
        if arm == "dir00":
            continue
        d_arm = depth.get((pred, lvl, arm), -1)
        ep = min(d_arm, d_base)
        if ep < 1:
            continue
        a = next((x for x in arms if x["predictor"] == pred and x["level"] == lvl
                  and x["arm"] == arm), None)
        if a is None:
            continue
        fb = base["run_dir"] / f"epoch_{ep:03d}" / "full_roa_per_point.npz"
        fa = a["run_dir"] / f"epoch_{ep:03d}" / "full_roa_per_point.npz"
        if not (fb.exists() and fa.exists()):
            continue
        with np.load(fb) as z:
            ib = match_to_truth(z["start_states"], gt[0])
            pb = z["p_success"].astype(np.float64)
        with np.load(fa) as z:
            ia = match_to_truth(z["start_states"], gt[0])
            pa = z["p_success"].astype(np.float64)
        if not np.array_equal(ia, ib):
            continue
        d = paired_delta(pa, pb, gt[1][ib], k, k, 90.0)
        mark = " **" if abs(d["z"]) > 2 else ""
        if thr is None or thr <= 0:
            beats = "no floor yet"
        elif abs(d["delta"]) > thr:
            beats = "**yes**"
        else:
            beats = f"no (|Δ| < {thr:.5f})"
        lines.append(f"| {ARM_LABEL[arm]} | {ep} | {d['delta']:+.6f}{mark} | "
                     f"{d['se']:.6f} | {d['z']:+.2f} | {beats} |")
        any_row = True
    return lines + [""] if any_row else []


def paired_table(arms: list[dict], pred: str, lvl: str, epoch: int, gt: tuple,
                 rows: list[dict]) -> list[str]:
    """Adaptive-vs-non-adaptive on identical eval cells.

    At the plateau the arms differ by far less than the spread across cells, so
    an unpaired table cannot tell a real gain from noise. Pairing on the cell
    removes that variance; the ground-truth noise term cancels exactly.
    """
    base = next((a for a in arms if a["predictor"] == pred and a["level"] == lvl
                 and a["arm"] == "dir00"), None)
    if base is None:
        return []
    bf = base["run_dir"] / f"epoch_{epoch:03d}" / "full_roa_per_point.npz"
    if not bf.exists():
        return []
    k = 100.0 if pred == "fm" else None
    with np.load(bf) as z:
        idx = match_to_truth(z["start_states"], gt[0])
        p_base = z["p_success"].astype(np.float64)
    p_true = gt[1][idx]

    lines = [f"Paired vs non-adaptive at epoch {epoch} "
             f"(negative = adaptive better; |z| > 2 is significant):", ""]
    if "UNMATCHED" in str(base.get("source", "")):
        lines += ["> **These deltas are confounded.** The non-adaptive run used here was "
                  "trained on different GPU hardware, so it does not reproduce the shared "
                  "epoch-0 model. Part of every delta below is that offset rather than the "
                  "effect of acquisition. The `Δ vs own epoch 0` column removes the offset "
                  "by measuring each arm's own improvement since acquisition began; compare "
                  "those instead until a matched control finishes.", ""]
    def own(a_arm: str, ep: int):
        return next((r["brier_debiased"] for r in rows
                     if r["predictor"] == pred and r["level"] == lvl
                     and r["arm"] == a_arm and r["epoch"] == ep), None)

    b0, bN = own("dir00", 0), own("dir00", epoch)
    base_own = f"{bN - b0:+.5f}" if (b0 is not None and bN is not None) else "n/a"
    lines += ["| arm | Δ debiased Brier | SE | z | Δ vs own epoch 0 |",
              "|---|---|---|---|---|",
              f"| _non-adaptive (reference)_ | – | – | – | {base_own} |"]
    any_row = False
    for arm in ARM_ORDER:
        if arm == "dir00":
            continue
        a = next((x for x in arms if x["predictor"] == pred and x["level"] == lvl
                  and x["arm"] == arm), None)
        if a is None:
            continue
        f = a["run_dir"] / f"epoch_{epoch:03d}" / "full_roa_per_point.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            i2 = match_to_truth(z["start_states"], gt[0])
            p_arm = z["p_success"].astype(np.float64)
        if not np.array_equal(i2, idx):
            order = np.argsort(i2)
            p_arm, i2 = p_arm[order], i2[order]
            o2 = np.argsort(idx)
            if not np.array_equal(i2, idx[o2]):
                continue
            p_base_use, p_true_use = p_base[o2], p_true[o2]
        else:
            p_base_use, p_true_use = p_base, p_true
        e0, eN = own(arm, 0), own(arm, epoch)
        own_txt = f"{eN - e0:+.5f}" if (e0 is not None and eN is not None) else "n/a"

        d = paired_delta(p_arm, p_base_use, p_true_use, k, k, 90.0)
        if d["se"] == 0.0 and d["delta"] == 0.0:
            # Identical predictions: before the first acquisition every arm is the
            # same model trained on the same random initial pool.
            lines.append(f"| {ARM_LABEL[arm]} | identical | – | – | {own_txt} |")
        else:
            mark = " **" if abs(d["z"]) > 2 else ""
            lines.append(f"| {ARM_LABEL[arm]} | {d['delta']:+.6f}{mark} | {d['se']:.6f} "
                         f"| {d['z']:+.2f} | {own_txt} |")
        any_row = True
    return lines + [""] if any_row else []


if __name__ == "__main__":
    main()
