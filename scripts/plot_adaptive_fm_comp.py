#!/usr/bin/env python
"""Level-set figures restricted to the four FM acquisition scores.

Same panels, same style contract and same oracle/floor conventions as
plot_stoch_levelsets.py, but the arm table is cut to the four flow-matching
acquisition variants under comparison plus the 3-seed non-adaptive control,
which is drawn as a min-max band and a mean line rather than as three arms.

The four:
    epi_bald           BALD alone, no yield term
    yield_mlp          epistemic variance x L_hat   (the epi-var yield rule)
    yield_mlp_bald     BALD x L_hat
    yield_mlp_bald_db  debiased BALD x L_hat

Classifier, BNN and Part-X arms are deliberately absent: this figure answers
"which acquisition score for the FM predictor", so mixing in other predictor
families would invite reading an arm against the wrong baseline. With no
exact-family arm present only the FM oracle ceiling is drawn, which is the
correct ceiling for every line here.

INPUT is unchanged -- the scored CSVs under docs/stochastic. Only the figures
are written elsewhere, to docs/adaptive-fm-comp, mirroring the campaign
subdirectory layout so a level's figure sits beside its siblings.

Usage:
    python scripts/plot_adaptive_fm_comp.py                    # every campaign
    python scripts/plot_adaptive_fm_comp.py pendulum cartpole
    python scripts/plot_adaptive_fm_comp.py --out-root /some/other/dir
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_stoch_all_levels import CAMPAIGNS, DOCS, SEEDS  # noqa: E402
from plot_stoch_levelsets import (  # noqa: E402
    draw_level, final_epoch_rows, level_rows_within_budget,
)
from score_stoch_incremental import levelsets_path  # noqa: E402

OUT_ROOT = Path("/common/users/shared/pracsys/genMoPlan/docs/adaptive-fm-comp")

# Colours for epi_bald and yield_mlp are kept at their canonical values so a
# reader moving between this figure and the full one does not have to relearn
# them. The two BALD-yield arms are new here and take unused hues.
FM_ARMS = [
    ("epi_bald",          "FM  BALD  (no yield)",        "#2ca02c", "-", "^"),
    ("yield_mlp",         "FM  epi-var x yield",         "#ff7f0e", "-", "s"),
    ("yield_mlp_bald",    "FM  BALD x yield",            "#1f77b4", "-", "o"),
    ("yield_mlp_bald_db", "FM  BALD debiased x yield",   "#d62728", "-", "D"),
]
KEEP = {a for a, *_ in FM_ARMS} | set(SEEDS)

SUBTITLE = ("truth and prediction both thresholded at β · "
            "all four lines are flow matching, so they share one baseline and one ceiling · "
            "grey band = 3-seed non-adaptive range, black = its mean · "
            "dashed black = what a perfect model scores through the same sampling noise")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="*", default=list(CAMPAIGNS))
    ap.add_argument("--allow-partial", action="store_true",
                    help="draw a level whose arms sit at ragged depths; the figure is stamped")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    a = ap.parse_args()

    wrote = 0
    for name in (a.campaigns or list(CAMPAIGNS)):
        c = CAMPAIGNS[name]
        src = levelsets_path(DOCS / c["csv"])
        if not src.exists():
            print(f"  SKIP {name}: {src.name} absent (run score_stoch_incremental first)")
            continue
        wide = list(csv.DictReader((DOCS / c["csv"]).open()))
        rows = level_rows_within_budget(list(csv.DictReader(src.open())), wide, c)
        cap = f"  ·  read to {c['budget']:,} training trajectories" if c.get("budget") else ""
        for lv, lab in c["panels"]:
            sub = [r for r in rows if r["level"] == lv and r["arm"] in KEEP]
            if not sub:
                print(f"  SKIP {name}/{lv}: no rows for the four FM arms")
                continue
            by_arm, depth = final_epoch_rows(sub)
            missing = sorted(KEEP - set(by_arm))
            if missing:
                print(f"  NOTE {name}/{lv}: absent {missing}")
            if len(set(depth.values())) > 1 and not a.allow_partial:
                print(f"  REFUSING {name}/{lv}: ragged depths {dict(sorted(depth.items()))} "
                      "(pass --allow-partial to override)")
                continue
            out = (a.out_root / c["out"]).with_name(f"levelsets_{lv}.png")
            draw_level(by_arm, depth, out, lab + cap, arms=FM_ARMS, subtitle=SUBTITLE)
            print(f"  wrote {out.relative_to(a.out_root)}  ({len(by_arm)} series, "
                  f"final epoch {max(depth.values())})")
            wrote += 1
    print(f"\n{wrote} figure(s) under {a.out_root}")


if __name__ == "__main__":
    main()
