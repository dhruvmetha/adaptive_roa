#!/usr/bin/env python3
"""
Classification (CLF) vs Generative flow-matching (FM) for ROA.

Systems: pendulum, cartpole, quad2d, quad3d, humanoid (standup reach).
Metrics (band / lambda_delta operating point): F1, Abstain% (uncertain+invalid),
FPR=FP/(FP+TN), FNR=FN/(FN+TP). x = # training trajectories, capped per system to
the common max budget across the arms shown.

Arms per system (auto-skipped where a run doesn't exist):
  clf_adaptive, clf_nonadapt           -- all systems
  fm_ranked, fm_direct, fm_nonadapt    -- d2=1.0 systems (pendulum/cartpole/quad2d[no ranked]/quad3d)
  fm_adapt_mfF/mfT, fm_nonadapt_mfF/mfT-- humanoid only (d2=0.5 direct; manifold False/True)

NOTE: humanoid is structurally different — adaptive = direct d2=0.5 (not ranked d2=1.0),
and FM comes in Euclidean (manifold=False) and manifold=True flavours; CLF is manifold=True.
FM metrics: <run>/evaluations/<eval>/epoch_XXX/artifacts_v2.json -> eval_metrics.lambda_delta.
"""
import json, glob, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/common/users/shared/pracsys/adaptive_roa_experiments"
DHRUV = f"{EXP}/dhruv"
LOCAL = "/common/home/st1122/Projects/adaptive_roa/outputs"
OUT = "/common/home/st1122/Projects/adaptive_roa/docs/plots"
os.makedirs(OUT, exist_ok=True)
SYSTEMS = ["pendulum", "cartpole", "quad2d", "quad3d", "humanoid"]

FM_ROOT = {"cartpole": "dhruv/adaptive_cartpole_pybullet", "pendulum": "dhruv/adaptive_pendulum_dhruv",
           "quad2d": "dhruv/adaptive_quadrotor2d", "quad3d": "dhruv/adaptive_quadrotor3d",
           "humanoid": "adaptive_humanoid_standup_reach"}

# (system, arm) -> (d2_ratio token, sampling_mode, timestamp, manifold|None)
FM_SPEC = {
 ("cartpole","fm_ranked"): ("1.0","ranked","2026-02-18_19-22-14",None),
 ("cartpole","fm_direct"): ("1.0","direct","2026-02-18_14-26-32",None),
 ("cartpole","fm_nonadapt"): ("0.0","ranked","2026-02-18_14-26-34",None),
 ("pendulum","fm_ranked"): ("1.0","ranked","2026-02-20_01-55-55",None),
 ("pendulum","fm_direct"): ("1.0","direct","2026-02-20_01-56-07",None),
 ("pendulum","fm_nonadapt"): ("0.0","direct","2026-02-28_13-15-25",None),
 ("quad2d","fm_direct"): ("1.0","direct","2026-02-28_10-05-32",None),
 ("quad2d","fm_nonadapt"): ("0.0","direct","2026-02-28_10-17-55",None),
 ("quad3d","fm_ranked"): ("1.0","ranked","2026-02-26_12-02-36",None),
 ("quad3d","fm_direct"): ("1.0","direct","2026-02-26_12-02-36",None),
 ("quad3d","fm_nonadapt"): ("0.0","direct","2026-02-28_11-49-54",None),
 ("humanoid","fm_adapt_mfF"): ("0.5","direct","2026-06-24_23-51-39","False"),
 ("humanoid","fm_adapt_mfT"): ("0.5","direct","2026-06-24_23-51-39","True"),
 ("humanoid","fm_nonadapt_mfF"): ("0","direct","2026-06-24_23-51-39","False"),
 ("humanoid","fm_nonadapt_mfT"): ("0","direct","2026-06-24_23-51-39","True"),
}
FORCE_EVAL = {"quad3d": "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000"}
# epoch->traj fallback (initial, per-epoch) when inline results.json lacks train_trajectories
SCHED = {"humanoid": (1000, 1000)}

# --- New-criteria (newcrit) FM runs ---------------------------------------
# ROA success/failure criteria changed (commit cf1abb5) for quad2d/quad3d/humanoid
# ONLY (pendulum/cartpole unchanged). The retrained FM arms for those 3 systems live at
# NEW run dirs with leaf `newcrit_20260708` and write per-epoch metrics INLINE at
# epoch_XXX/results.json (NO evaluations/ subdir). (system, arm) -> glob pattern; the
# resolver globs {EXP}/{root}/outputs/<pattern> and returns the run dir (or None).
NEWCRIT_SYSTEMS = {"quad2d", "quad3d", "humanoid"}
# newcrit runs live under these roots (NOT the dhruv/ FM_ROOT ones)
NEWCRIT_ROOT = {"quad2d": "adaptive_quadrotor2d", "quad3d": "adaptive_quadrotor3d",
                "humanoid": "adaptive_humanoid_standup_reach"}
FM_NEWCRIT = {
 ("quad2d","fm_ranked"):   "*d2_ratio_1.0_*sampling_mode_ranked*/newcrit_20260708",
 ("quad2d","fm_direct"):   "*d2_ratio_1.0_*sampling_mode_direct*/newcrit_20260708",
 ("quad2d","fm_nonadapt"): "*d2_ratio_0.0_*adapt_iter_10_*sampling_mode_direct*/newcrit_20260708",
 ("quad3d","fm_ranked"):   "*d2_ratio_1.0_*adapt_iter_20_*sampling_mode_ranked*/newcrit_20260708",
 ("quad3d","fm_direct"):   "*d2_ratio_1.0_*adapt_iter_20_*sampling_mode_direct*/newcrit_20260708",
 ("quad3d","fm_nonadapt"): "*d2_ratio_0.0_*adapt_iter_15_*sampling_mode_direct*/newcrit_20260708",
 ("humanoid","fm_adapt_mfF"):    "*d2_ratio_0.5_*manifold_False_*sampling_mode_direct*/newcrit_20260708",
 ("humanoid","fm_adapt_mfT"):    "*d2_ratio_0.5_*manifold_True_*sampling_mode_direct*/newcrit_20260708",
 ("humanoid","fm_ranked_mfF"):   "*d2_ratio_0.5_*manifold_False_*sampling_mode_ranked*/newcrit_20260708",
 ("humanoid","fm_ranked_mfT"):   "*d2_ratio_0.5_*manifold_True_*sampling_mode_ranked*/newcrit_20260708",
 ("humanoid","fm_nonadapt_mfF"): "*d2_ratio_0.0_*manifold_False_*sampling_mode_direct*/newcrit_20260708",
 ("humanoid","fm_nonadapt_mfT"): "*d2_ratio_0.0_*manifold_True_*sampling_mode_direct*/newcrit_20260708",
}

CLF_DIR = {
 ("pendulum","clf_adaptive"): f"{DHRUV}/adaptive_classification/pendulum/adaptive",
 ("pendulum","clf_nonadapt"): f"{DHRUV}/adaptive_classification/pendulum/random",
 ("cartpole","clf_adaptive"): f"{DHRUV}/adaptive_classification/cartpole/adaptive",
 ("cartpole","clf_nonadapt"): f"{DHRUV}/adaptive_classification/cartpole/random",
 ("quad2d","clf_adaptive"): f"{DHRUV}/adaptive_classification/quad2d/adaptive",
 ("quad2d","clf_nonadapt"): f"{DHRUV}/adaptive_classification/quad2d/random",
 ("quad3d","clf_adaptive"): f"{LOCAL}/clf_quad3d_match_fm/adaptive",
 ("quad3d","clf_nonadapt"): f"{LOCAL}/clf_quad3d_match_fm_slurm/random",
 ("humanoid","clf_adaptive"): f"{EXP}/adaptive_humanoid_standup_reach_classifier/outputs/*d2_ratio_0.5*/2026-06-24_23-58-56",
 ("humanoid","clf_nonadapt"): f"{EXP}/adaptive_humanoid_standup_reach_classifier/outputs/*d2_ratio_0_warm*/2026-06-24_23-58-56",
}

# --- part-X runs -----------------------------------------------------------
# part-X writes per-epoch metrics INLINE at epoch_XXX/results.json in the SAME
# on-disk format as the newcrit FM runs (full_roa.lambda_delta + full_roa.n_total,
# train_trajectories recorded). system -> run dir (or None if not ready yet).
PARTX_DIR = {
 "pendulum": f"{EXP}/adaptive_pendulum_dhruv/outputs/training_index_0_warm_start_False_adapt_iter_10/2026-07-09_17-37-51",
 "cartpole": f"{EXP}/adaptive_cartpole_pybullet/outputs/training_index_0_warm_start_False_manifold_False_adapt_iter_15/2026-07-09_17-37-51",
 "quad2d":   f"{EXP}/adaptive_quadrotor2d/outputs/training_index_0_warm_start_False_manifold_False_adapt_iter_10/2026-07-09_23-13-21",
 "quad3d":   f"{EXP}/adaptive_quadrotor3d/outputs/training_index_0_warm_start_False_manifold_False_adapt_iter_15/2026-07-09_23-13-21",
 "humanoid": f"{EXP}/adaptive_humanoid_standup_reach/outputs/training_index_0_warm_start_False_manifold_True_adapt_iter_30/2026-07-09_23-13-21",
}

# --- Dispersion FM runs (2026-07-26 campaign) -----------------------------
# Acquisition scores candidates by the spread of the predicted final-state cloud
# instead of by label counts. Run on Amarel, results synced into EXP. Same inline
# epoch_XXX/results.json format as newcrit, so load_fm_inline handles them.
# arm -> (d2_ratio, selection_rule)
DISPERSION_ROOT = {"pendulum": "adaptive_pendulum_lqr", "cartpole": "adaptive_cartpole_pybullet",
                   "quad2d": "adaptive_quadrotor2d", "quad3d": "adaptive_quadrotor3d"}
FM_DISPERSION = {
 "fm_disp_greedy_d05":   ("0.5", "greedy"),
 "fm_disp_gdiv_d05":     ("0.5", "greedy_diverse"),
 "fm_disp_prop_d05":     ("0.5", "proportional"),
 "fm_disp_greedy_d10":   ("1.0", "greedy"),
 "fm_disp_gdiv_d10":     ("1.0", "greedy_diverse"),
 "fm_disp_prop_d10":     ("1.0", "proportional"),
}

ARMS = ["clf_adaptive","clf_nonadapt","partx","fm_ranked","fm_direct","fm_nonadapt",
        "fm_adapt_mfF","fm_adapt_mfT","fm_ranked_mfF","fm_ranked_mfT",
        "fm_nonadapt_mfF","fm_nonadapt_mfT",
        "fm_disp_greedy_d05","fm_disp_gdiv_d05","fm_disp_prop_d05",
        "fm_disp_greedy_d10","fm_disp_gdiv_d10","fm_disp_prop_d10"]
STYLE = {
 "clf_adaptive":   dict(color="#08519c", marker="o", ls="-",  label="CLF adaptive"),
 "clf_nonadapt":   dict(color="#6baed6", marker="o", ls="--", label="CLF non-adaptive"),
 "partx":          dict(color="#238b45", marker="P", ls="-",  label="part-X"),
 "fm_ranked":      dict(color="#a50f15", marker="^", ls="-",  label="FM adaptive ranked (d2=1.0)"),
 "fm_direct":      dict(color="#d94801", marker="s", ls="-",  label="FM adaptive direct (d2=1.0)"),
 "fm_nonadapt":    dict(color="#fdae6b", marker="s", ls="--", label="FM non-adaptive (d2=0)"),
 "fm_adapt_mfF":   dict(color="#a50f15", marker="s", ls="-",  label="FM adaptive direct Euclid (d2=0.5)"),
 "fm_adapt_mfT":   dict(color="#d94801", marker="^", ls="-",  label="FM adaptive direct manifold (d2=0.5)"),
 "fm_ranked_mfF":  dict(color="#54278f", marker="D", ls="-",  label="FM adaptive ranked Euclid (d2=0.5)"),
 "fm_ranked_mfT":  dict(color="#3f007d", marker="v", ls="-",  label="FM adaptive ranked manifold (d2=0.5)"),
 "fm_nonadapt_mfF":dict(color="#fdae6b", marker="s", ls="--", label="FM non-adapt Euclid (d2=0)"),
 "fm_nonadapt_mfT":dict(color="#fa9fb5", marker="^", ls="--", label="FM non-adapt manifold (d2=0)"),
 # Dispersion: colour = selection rule, solid = d2 1.0, dashed = d2 0.5
 "fm_disp_greedy_d10": dict(color="#6a51a3", marker="*", ls="-",  label="FM dispersion greedy (d2=1.0)"),
 "fm_disp_gdiv_d10":   dict(color="#c51b8a", marker="X", ls="-",  label="FM dispersion greedy-diverse (d2=1.0)"),
 "fm_disp_prop_d10":   dict(color="#017a4a", marker="d", ls="-",  label="FM dispersion proportional (d2=1.0)"),
 "fm_disp_greedy_d05": dict(color="#6a51a3", marker="*", ls=":",  label="FM dispersion greedy (d2=0.5)"),
 "fm_disp_gdiv_d05":   dict(color="#c51b8a", marker="X", ls=":",  label="FM dispersion greedy-diverse (d2=0.5)"),
 "fm_disp_prop_d05":   dict(color="#017a4a", marker="d", ls=":",  label="FM dispersion proportional (d2=0.5)"),
}

def resolve_fm(sys, arm):
    sp = FM_SPEC.get((sys, arm))
    if sp is None: return None
    d2, mode, ts, mf = sp
    mid = f"*manifold_{mf}_" if mf is not None else "*"
    hits = glob.glob(f"{EXP}/{FM_ROOT[sys]}/outputs/*d2_ratio_{d2}_{mid}*sampling_mode_{mode}*/{ts}")
    return hits[0] if hits else None

def resolve_fm_newcrit(sys, arm):
    pat = FM_NEWCRIT.get((sys, arm))
    if pat is None: return None
    hits = glob.glob(f"{EXP}/{NEWCRIT_ROOT[sys]}/outputs/{pat}")
    return hits[0] if hits else None

def resolve_fm_dispersion(sys, arm):
    """Dispersion arms: outputs/<config>/<timestamp>/epoch_XXX/results.json.

    Same inline on-disk format as the newcrit runs, so load_fm_inline reads them —
    only the run-dir resolution differs. The config segment carries d2_ratio,
    sampling_mode and selection_rule, so the glob pins all three; the trailing /*
    steps into the timestamp dir that holds the epoch_* dirs."""
    spec = FM_DISPERSION.get(arm)
    if spec is None or sys not in DISPERSION_ROOT: return None
    d2, rule = spec
    hits = sorted(glob.glob(
        f"{EXP}/{DISPERSION_ROOT[sys]}/outputs"
        f"/*d2_ratio_{d2}_*sampling_mode_dispersion_selection_{rule}_exp_*/*"))
    return hits[-1] if hits else None

def resolve_clf(sys, arm):
    d = CLF_DIR.get((sys, arm))
    if d and "*" in d:
        h = glob.glob(d); return h[0] if h else None
    return d

def metrics_from_ld(ld, n_total):
    fp, tn, fn, tp = ld.get("fp"), ld.get("tn"), ld.get("fn"), ld.get("tp")
    n_conf = ld.get("n_confident")
    return dict(
        F1=ld.get("f1"),
        FPR=(fp/(fp+tn) if (fp+tn) else np.nan),
        FNR=(fn/(fn+tp) if (fn+tp) else np.nan),
        Abstain=(100.0*(n_total-n_conf)/n_total if (n_total and n_conf is not None) else np.nan),
    )

def fm_eval_base(sys):
    if sys in NEWCRIT_SYSTEMS: return None  # newcrit FM writes metrics inline, no evaluations/
    if sys in FORCE_EVAL: return FORCE_EVAL[sys]
    sets = []
    for arm in [a for a in ARMS if a.startswith("fm")]:
        d = resolve_fm(sys, arm)
        if d: sets.append({os.path.basename(p.rstrip("/")) for p in glob.glob(f"{d}/evaluations/*/")})
    if not sets: return None
    common = set.intersection(*sets)
    pref = sorted(common, key=lambda b: ("mc_20" not in b, "calk" in b, b))
    return pref[0] if pref else None

FM_EVAL = {s: fm_eval_base(s) for s in SYSTEMS}

def load_fm(sys, arm):
    d = resolve_fm(sys, arm)
    if not d: return []
    init_spe = SCHED.get(sys)
    traj_of = {}
    for e in sorted(glob.glob(f"{d}/epoch_*")):
        ep = int(os.path.basename(e).split("_")[1])
        t = None
        try: t = json.load(open(f"{e}/results.json")).get("train_trajectories")
        except: pass
        if (not t) and init_spe: t = init_spe[0] + ep * init_spe[1]
        traj_of[ep] = t
    ev = f"{d}/evaluations/{FM_EVAL[sys]}"
    rows = []
    if not FM_EVAL[sys] or not os.path.isdir(ev): return rows
    for e in sorted(glob.glob(f"{ev}/epoch_*")):
        ep = int(os.path.basename(e).split("_")[1])
        fr = (json.load(open(f"{e}/artifacts_v2.json")).get("eval_metrics") or {})
        ld = fr.get("lambda_delta") or {}
        if not ld: continue
        m = metrics_from_ld(ld, fr.get("n_total")); m.update(epoch=ep, traj=traj_of.get(ep))
        rows.append(m)
    rows.sort(key=lambda r: (r["traj"] is None, r["traj"]))
    return [r for r in rows if r["traj"] is not None]

def load_fm_inline(sys, arm):
    """Newcrit FM loader: metrics written INLINE at epoch_XXX/results.json (no evaluations/).
    Uses full_roa['n_total'] for Abstain and full_roa['lambda_delta'] for F1/FPR/FNR.
    Falls back to artifacts_v2.json (eval_metrics) if full_roa is missing/skipped.
    Returns the same row-dict shape as load_fm/load_clf (traj/F1/FPR/FNR/Abstain/epoch)."""
    d = resolve_fm_dispersion(sys, arm) if arm.startswith("fm_disp") else resolve_fm_newcrit(sys, arm)
    if not d: return []
    init_spe = SCHED.get(sys)
    rows = []
    for e in sorted(glob.glob(f"{d}/epoch_*")):
        ep = int(os.path.basename(e).split("_")[1])
        ld, n_total = {}, None
        try: r = json.load(open(f"{e}/results.json"))
        except: r = {}
        fr = r.get("full_roa")
        if isinstance(fr, dict) and not fr.get("skipped"):
            ld = fr.get("lambda_delta") or {}
            n_total = fr.get("n_total")
        if not ld:  # fallback to artifacts_v2.json
            try:
                em = json.load(open(f"{e}/artifacts_v2.json")).get("eval_metrics") or {}
                ld = em.get("lambda_delta") or {}
                n_total = em.get("n_total")
            except: pass
        if not ld: continue
        t = r.get("train_trajectories")
        if (not t) and init_spe: t = init_spe[0] + ep * init_spe[1]
        m = metrics_from_ld(ld, n_total); m.update(epoch=ep, traj=t)
        rows.append(m)
    rows.sort(key=lambda r: (r["traj"] is None, r["traj"]))
    return [r for r in rows if r["traj"] is not None]

def load_partx(sys):
    """part-X loader: metrics written INLINE at epoch_XXX/results.json (same on-disk
    format as newcrit FM). Reads full_roa['lambda_delta'] + full_roa['n_total'] and
    train_trajectories per epoch. Skips epochs whose results.json is missing or whose
    full_roa is absent/skipped. Returns the same row-dict shape as the other loaders."""
    d = PARTX_DIR.get(sys)
    if not d or not os.path.isdir(d): return []
    rows = []
    for e in sorted(glob.glob(f"{d}/epoch_*")):
        ep = int(os.path.basename(e).split("_")[1])
        try: r = json.load(open(f"{e}/results.json"))
        except: continue
        fr = r.get("full_roa")
        if not isinstance(fr, dict) or fr.get("skipped"): continue
        ld = fr.get("lambda_delta") or {}
        if not ld: continue
        m = metrics_from_ld(ld, fr.get("n_total"))
        m.update(epoch=ep, traj=r.get("train_trajectories"))
        rows.append(m)
    rows.sort(key=lambda r: (r["traj"] is None, r["traj"]))
    return [r for r in rows if r["traj"] is not None]

def load_clf(sys, arm):
    d = resolve_clf(sys, arm)
    rows = []
    if not d: return rows
    for e in sorted(glob.glob(f"{d}/epoch_*")):
        try: r = json.load(open(f"{e}/results.json"))
        except: continue
        fr = r.get("full_roa") or {}; ld = fr.get("lambda_delta") or {}
        if not ld: continue
        m = metrics_from_ld(ld, fr.get("n_total")); m.update(epoch=r.get("epoch"), traj=r.get("train_trajectories"))
        rows.append(m)
    return rows

def load_arm(sys, arm):
    if arm == "partx":
        return load_partx(sys)             # part-X: inline results.json (green arm)
    if arm.startswith("clf"):
        return load_clf(sys, arm)          # CLF: unchanged old paths
    if arm.startswith("fm_disp"):
        return load_fm_inline(sys, arm)    # dispersion: inline, own resolver
    if sys in NEWCRIT_SYSTEMS:
        return load_fm_inline(sys, arm)    # quad2d/quad3d/humanoid FM: newcrit inline
    return load_fm(sys, arm)               # pendulum/cartpole FM: unchanged old paths

if __name__ == "__main__":
    ALL = {}
    for s in SYSTEMS:
        for a in ARMS:
            ALL[(s,a)] = load_arm(s, a)

    CAP = {}
    for s in SYSTEMS:
        # exclude part-X from the common-budget cap: it self-limits to a much smaller
        # trajectory budget, and letting it drag the cap would clip the CLF/FM curves.
        # part-X still plots to its own (shorter) extent; the _full version is uncapped.
        maxes = [max(r["traj"] for r in ALL[(s,a)]) for a in ARMS if ALL[(s,a)] and a != "partx"]
        CAP[s] = min(maxes) if maxes else None
    print("FM eval dirs:", FM_EVAL); print("caps:", CAP)
    for s in SYSTEMS:
        present = [a for a in ARMS if ALL[(s,a)]]
        print(f"  {s}: {len(present)} arms -> {present}")

    # newcrit FM arm status: epochs loaded vs empty (in-progress) vs missing (not started)
    print("\nnewcrit FM arm status (system, arm -> #epochs loaded | dir):")
    for (s, a) in FM_NEWCRIT:
        d = resolve_fm_newcrit(s, a)
        n = len(ALL[(s, a)])
        if d is None:
            print(f"  {s:9}|{a:16}: MISSING (no run dir)")
        elif n == 0:
            print(f"  {s:9}|{a:16}: EMPTY / in-progress (0 usable epochs)  [{d}]")
        else:
            print(f"  {s:9}|{a:16}: {n} epochs loaded")

    # CSV
    csv_path = f"{OUT}/clf_vs_fm_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system","arm","train_traj","F1","FPR","FNR","abstain_pct"])
        for s in SYSTEMS:
            for a in ARMS:
                for r in ALL[(s,a)]:
                    w.writerow([s,a,r["traj"],f"{r['F1']:.4f}",f"{r['FPR']:.4f}",f"{r['FNR']:.4f}",f"{r['Abstain']:.3f}"])
    print("wrote", csv_path)

    METRICS = [("F1","F1 (band / committed points)"),("Abstain","Abstain%  (uncertain + invalid)"),
               ("FPR","FPR = FP/(FP+TN)"),("FNR","FNR = FN/(FN+TP)")]

    def panel(ax, s, key, cap):
        for a in ARMS:
            rows = [r for r in ALL[(s,a)] if cap is None or r["traj"] <= cap]
            if not rows: continue
            st = STYLE[a]
            ax.plot([r["traj"] for r in rows], [r[key] for r in rows],
                    color=st["color"], marker=st["marker"], ls=st["ls"], ms=4, lw=1.6, label=st["label"])

    # Two renderings: capped (common budget, existing filenames) and full (uncapped x-axis, `_full`).
    MODES = [("",      {s: CAP[s] for s in SYSTEMS}, "common-budget capped"),
             ("_full", {s: None   for s in SYSTEMS}, "full x-axis (uncapped)")]

    for suffix, capmap, desc in MODES:
        # combined grid (per-row legend in the F1 column)
        fig, axes = plt.subplots(len(SYSTEMS), len(METRICS), figsize=(19, 4.0*len(SYSTEMS)))
        for i, s in enumerate(SYSTEMS):
            for j,(key,title) in enumerate(METRICS):
                ax = axes[i,j]; panel(ax, s, key, capmap[s])
                if i == 0: ax.set_title(title, fontsize=11)
                if j == 0:
                    ax.set_ylabel(f"{s}\n", fontsize=12, fontweight="bold")
                    ax.legend(fontsize=6.5, loc="best")
                ax.set_xlabel("# training trajectories", fontsize=8); ax.grid(True, alpha=0.3)
                if key == "F1": ax.set_ylim(0,1.02)
                if key in ("FPR","FNR","Abstain"): ax.set_ylim(bottom=0)
        fig.suptitle(f"Classification vs Generative FM for ROA — band mode, {desc} "
                     "(humanoid: d2=0.5 direct/ranked, manifold F/T)", fontsize=13)
        fig.tight_layout(rect=[0,0,1,0.99])
        fig.savefig(f"{OUT}/clf_vs_fm_grid{suffix}.png", dpi=130, bbox_inches="tight"); print(f"wrote grid{suffix}")
        plt.close(fig)

        for s in SYSTEMS:
            fig, axes = plt.subplots(2,2, figsize=(11,8))
            for ax,(key,title) in zip(axes.ravel(), METRICS):
                panel(ax, s, key, capmap[s]); ax.set_title(title); ax.set_xlabel("# training trajectories"); ax.grid(True, alpha=0.3)
                if key == "F1": ax.set_ylim(0,1.02)
                if key in ("FPR","FNR","Abstain"): ax.set_ylim(bottom=0)
            axes[0,0].legend(fontsize=8, loc="best")
            cap_txt = f"capped @ {CAP[s]} traj" if suffix == "" else "full x-axis (uncapped)"
            fig.suptitle(f"{s}: CLF vs FM ROA, {cap_txt}", fontsize=13)
            fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{OUT}/clf_vs_fm_{s}{suffix}.png", dpi=130, bbox_inches="tight"); print(f"wrote {s}{suffix}")
            plt.close(fig)

    print("\nFinal capped-epoch values:")
    for s in SYSTEMS:
        for a in ARMS:
            rows = [r for r in ALL[(s,a)] if r["traj"] <= CAP[s]]
            if not rows: continue
            r = rows[-1]
            print(f"  {s:9}|{a:16}: traj={r['traj']:>6} F1={r['F1']:.3f} abst={r['Abstain']:5.1f}% FPR={r['FPR']:.3f} FNR={r['FNR']:.3f}")
