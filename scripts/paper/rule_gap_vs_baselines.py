#!/usr/bin/env python
"""Which selection rule gives FM-BALD the larger margin over its comparators?

For each (system, level) and each selection rule (diverse = greedy_diverse, the
paper's default; greedy = plain top-N), read FM-BALD and every comparator under
THAT rule at the deepest epoch they all share, and report FM-BALD's advantage:

    advantage = comparator - FM-BALD   for KL          (lower is better)
    advantage = FM-BALD - comparator   for sAUROC, skill_score, auc_f05

Comparators: Part-X (no rule variant, same row under both rules), CLF-BALD and
BNN-BALD (their own greedy twins under the greedy rule). Part-X therefore only
measures FM-BALD's own rule change; CLF-BALD and BNN-BALD measure whether the
rule moves FM-BALD more or less than it moves them.

Inputs are the per-system wide CSVs (probability metrics) and the all-epoch
level-set areas store. quad3d is the 800k-pool campaign (decision 2026-09-10).
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

STOCH = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
PAPER = STOCH / "paper_draft"
AREAS = PAPER / "levelsets_greedy"
OUT = PAPER / "tables"

SYSTEMS = {
    "pendulum_lqr": (STOCH / "pendulum/lqr/gaussian_all_levels.csv", ["low", "med", "high"]),
    "cartpole_ppo": (STOCH / "cartpole/safe_explorer_ppo/gaussian_all_levels.csv",
                     ["baseline", "low", "med", "high"]),
    "quad2d_rl": (STOCH / "quadrotor2d/rl/quad2d_corridor_sine_ambient_all_levels.csv",
                  ["corridor_sine_ambient_smooth"]),
    "quad3d_ppo800k": (STOCH / "quadrotor3d/ppo_800k/quad3d800k_corridor_sine_ambient_all_levels.csv",
                       ["corridor_sine_ambient_f_0.00", "corridor_sine_ambient_f_0.12",
                        "corridor_sine_ambient_f_0.20", "corridor_sine_ambient_f_0.40"]),
}
RULES = {"diverse": "", "greedy": "_greedy"}
OURS = "epi_bald"
COMPARATORS = {"Part-X": ("partx_fix", False), "CLF-BALD": ("clf_epi_bald", True),
               "BNN-BALD": ("bnn_mfvi_a1", True)}
METRICS = {"KL": -1, "sAUROC": +1, "skill_score": +1, "auc_f05": +1}


def load(system: str) -> pd.DataFrame:
    wide_path, _ = SYSTEMS[system]
    wide = pd.read_csv(wide_path)
    cols = ["level", "arm", "epoch", "KL", "sAUROC", "skill_score"]
    wide = wide[cols].copy()
    ap = AREAS / f"{system}_levelset_areas_b50.csv"
    if ap.exists():
        areas = pd.read_csv(ap)[["level", "arm", "epoch", "auc_f05"]]
        wide = wide.merge(areas, on=["level", "arm", "epoch"], how="left")
    else:
        wide["auc_f05"] = np.nan
    for c in METRICS:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
    wide["epoch"] = wide["epoch"].astype(int)
    return wide


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    rows = []
    for system, (_, levels) in SYSTEMS.items():
        df = load(system)
        for level in levels:
            lv = df[df["level"] == level]
            for rule, suf in RULES.items():
                arms = {OURS + suf: "ours"}
                for name, (key, has_twin) in COMPARATORS.items():
                    arms[key + (suf if has_twin else "")] = name
                present = {a for a in arms if a in set(lv["arm"])}
                if OURS + suf not in present:
                    continue
                ep = int(min(lv[lv["arm"] == a]["epoch"].max() for a in present))
                at = lv[lv["epoch"] == ep].set_index("arm")
                ours = at.loc[OURS + suf]
                for a, name in arms.items():
                    if name == "ours" or a not in at.index:
                        continue
                    r = dict(system=system, level=level, rule=rule, epoch=ep, comparator=name,
                             comparator_arm=a)
                    for m, sign in METRICS.items():
                        r[f"{m}_ours"] = ours[m]
                        r[f"{m}_comp"] = at.loc[a, m]
                        r[f"{m}_adv"] = sign * (ours[m] - at.loc[a, m])
                    rows.append(r)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "rule_gap_vs_baselines.csv"
    out.to_csv(csv_path, index=False, float_format="%.6g")
    os.chmod(csv_path, 0o660)

    # Paired: same (system, level, comparator), diverse vs greedy advantage.
    piv = out.pivot_table(index=["system", "level", "comparator"], columns="rule",
                          values=[f"{m}_adv" for m in METRICS])
    lines = ["# FM-BALD's margin over each comparator, by selection rule", "",
             "Advantage is comparator minus FM-BALD for KL and FM-BALD minus comparator for the "
             "rest, so positive always means FM-BALD ahead. Both arms read at the deepest epoch "
             "shared by every arm in the cell under that rule. quad3d is the 800k-pool campaign.", ""]
    summary = []
    for m in METRICS:
        col = f"{m}_adv"
        if (col, "diverse") not in piv.columns or (col, "greedy") not in piv.columns:
            continue
        d, g = piv[(col, "diverse")], piv[(col, "greedy")]
        both = d.notna() & g.notna()
        for comp in COMPARATORS:
            mask = both & (piv.index.get_level_values("comparator") == comp)
            if not mask.any():
                continue
            dd, gg = d[mask], g[mask]
            summary.append(dict(metric=m, comparator=comp, n=int(mask.sum()),
                                mean_adv_diverse=dd.mean(), mean_adv_greedy=gg.mean(),
                                cells_diverse_margin_larger=int((dd > gg).sum()),
                                cells_greedy_margin_larger=int((gg > dd).sum())))
    s = pd.DataFrame(summary)
    lines += ["## Summary (mean advantage, and in how many cells each rule gives the larger margin)", "",
              "| metric | comparator | n | mean adv, diverse | mean adv, greedy | diverse larger | greedy larger |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.metric} | {r.comparator} | {r.n} | {r.mean_adv_diverse:+.4f} | "
                     f"{r.mean_adv_greedy:+.4f} | {r.cells_diverse_margin_larger} | {r.cells_greedy_margin_larger} |")
    lines += ["", "## Per cell (KL advantage; F0.5-area advantage)", "",
              "| system | level | comparator | epoch d/g | KL adv diverse | KL adv greedy | F0.5 adv diverse | F0.5 adv greedy |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for (system, level, comp), r in piv.iterrows():
        ep = out[(out.system == system) & (out.level == level) & (out.comparator == comp)]
        epd = ep[ep.rule == "diverse"]["epoch"]; epg = ep[ep.rule == "greedy"]["epoch"]
        f = lambda v: "" if pd.isna(v) else f"{v:+.4f}"
        lines.append(f"| {system} | {level} | {comp} | {int(epd.iloc[0]) if len(epd) else '-'}/"
                     f"{int(epg.iloc[0]) if len(epg) else '-'} | {f(r[('KL_adv','diverse')])} | "
                     f"{f(r[('KL_adv','greedy')])} | {f(r[('auc_f05_adv','diverse')])} | "
                     f"{f(r[('auc_f05_adv','greedy')])} |")
    md_path = OUT / "rule_gap_vs_baselines.md"
    md_path.write_text("\n".join(lines) + "\n")
    os.chmod(md_path, 0o660)
    print("\n".join(lines[:4 + 3 + len(s)]))
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
