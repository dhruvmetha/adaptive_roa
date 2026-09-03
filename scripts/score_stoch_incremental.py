#!/usr/bin/env python
"""Score only the epochs that are not yet in the committed per-family CSV.

The adaptive run already evaluates every epoch and writes full_roa_per_point.npz,
but its own threshold_free metrics score against a 0.5-dichotomised label field.
This driver runs the projection in scripts/stoch_prob_metrics.py, which joins that
written p_success to the CONTINUOUS eval_success_prob.npz ground truth. No model
inference happens here -- it is a re-projection of results already on disk, so it
is cheap and safe to run every monitoring cycle.

Incremental by construction: for each campaign it diffs the epochs present on disk
against the epochs already in the CSV and scores only the difference. Arms are
grouped by identical missing-epoch sets because --epochs applies to a whole spec.

Merge key is (level, predictor, arm, epoch) with the NEW row winning. The pass is
deterministic, so a collision is a no-op; preferring new means a re-run after a
ground-truth update actually takes effect.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPROOT = Path("/common/users/shared/pracsys/adaptive_roa_experiments")
EXP = EXPROOT / "quadrotor_stoch"
GT = EXPROOT / "gaussian_torque"
DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic")
# Doc roots are per CONTROLLER, matching the data tree's
# <system>/<family>/<controller>/<level>. A system with two controllers (cartpole
# has lqr and safe_explorer_ppo) would otherwise write both into one CSV, and the
# level keys collide -- both controllers have a `med`.
DOCS = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
D2D = DOCS / "quadrotor2d/rl"
D3D = DOCS / "quadrotor3d/lqr"
DPEN = DOCS / "pendulum/lqr"
DCP = DOCS / "cartpole/lqr"
DCPRL = DOCS / "cartpole/safe_explorer_ppo"
PY = str(ROOT / "env/bin/python")

# "partx" (the pre-fix Part-X arm) is deliberately ABSENT. Its partition root was
# built from component-collapsed bounds, so acquisition could not reach 91% of the
# quadrotor2D state space or 43% of quadrotor3D's and the arms spent ~40-45% of
# their budget. Only "partx_fix" is a Part-X result. Re-adding the name here would
# silently repopulate the level CSVs it was purged from.
ARMS = ["dir00_s42", "dir00_s43", "dir00_s44", "epi_var", "epi_var_anch", "epi_bald",
        "yield_a1", "yield_mlp", "partx_fix", "clf_dir00", "clf_yield", "clf_epi_var",
        "clf_epi_bald", "clf_epi_var_anch",
        "bnn_ens", "bnn_lap", "bnn_mfvi", "bnn_mfvi_bald",
        # H_bb-a: the same BNN acquiring on total predictive entropy. This is the
        # baseline Depeweg et al. (2018) Table 1 argues against; without it the
        # decomposition-vs-total-entropy claim cannot be tested either way.
        "bnn_mfvi_total"]

# campaign -> run-dir prefix, ground-truth root, dataset level fragment, CSV level key
CAMPAIGNS = {
    "q2d_nd":    dict(prefix="q2d_nd",    root=DATA / "quadrotor2D",
                      level="noisy_dynamics/rl/f_0.150",
                      key="noisy_dynamics_f_0.150",
                      csv=D2D / "quad2d_noisy_dynamics_all_levels.csv"),
    "q2d_cs":    dict(prefix="q2d_cs",    root=DATA / "quadrotor2D",
                      level="corridor_sine_ambient/rl/smooth",
                      key="corridor_sine_ambient_smooth",
                      csv=D2D / "quad2d_corridor_sine_ambient_all_levels.csv"),
    "q3d_nd048": dict(prefix="q3d_nd048", root=DATA / "quadrotor3D",
                      level="noisy_dynamics/lqr/f_0.048",
                      key="noisy_dynamics_f_0.048",
                      csv=D3D / "quad3d_noisy_dynamics_all_levels.csv"),
    "q3d_nd060": dict(prefix="q3d_nd060", root=DATA / "quadrotor3D",
                      level="noisy_dynamics/lqr/f_0.060",
                      key="noisy_dynamics_f_0.060",
                      csv=D3D / "quad3d_noisy_dynamics_all_levels.csv"),
    # corridor_sine_ambient is a SEPARATE family from noisy_dynamics: ~1x body
    # weight vs the nd sweep's 0.12-0.27, so it gets its own csv. Never pool.
    "q3d_cs030": dict(prefix="q3d_cs030", root=DATA / "quadrotor3D",
                      level="corridor_sine_ambient/lqr/f_0.30",
                      key="corridor_sine_ambient_f_0.30",
                      csv=D3D / "quad3d_corridor_sine_ambient_all_levels.csv"),

    "pend_low":  dict(exp=GT, prefix="pen_low", root=DATA / "pendulum",
                      level="gaussian_signal/lqr/low", key="low",
                      csv=DPEN / "gaussian_all_levels.csv"),
    "pend_med":  dict(exp=GT, prefix="pen", root=DATA / "pendulum",
                      level="gaussian_signal/lqr/med", key="med",
                      csv=DPEN / "gaussian_all_levels.csv"),
    "pend_high": dict(exp=GT, prefix="pen_high", root=DATA / "pendulum",
                      level="gaussian_signal/lqr/high", key="high",
                      csv=DPEN / "gaussian_all_levels.csv"),
    "cp_low":    dict(exp=GT, prefix="cp_low_v2", root=DATA / "cartpole",
                      level="gaussian_signal/lqr/low", key="low",
                      csv=DCP / "gaussian_all_levels.csv"),
    "cp_med":    dict(exp=GT, prefix="cp_med_v2", root=DATA / "cartpole",
                      level="gaussian_signal/lqr/med", key="med",
                      csv=DCP / "gaussian_all_levels.csv"),
    "cp_high":   dict(exp=GT, prefix="cp_high_v2", root=DATA / "cartpole",
                      level="gaussian_signal/lqr/high", key="high",
                      csv=DCP / "gaussian_all_levels.csv"),

    # safe_explorer_ppo (RL) cartpole. SEPARATE csv from lqr: `med` exists under
    # both controllers and would silently overwrite in a shared file. `baseline`
    # is the ZERO-noise regime and has no lqr counterpart.
    "cprl_base": dict(exp=GT, prefix="cprl_base", root=DATA / "cartpole",
                      level="gaussian_signal/safe_explorer_ppo/baseline", key="baseline",
                      csv=DCPRL / "gaussian_all_levels.csv"),
    "cprl_med":  dict(exp=GT, prefix="cprl_med", root=DATA / "cartpole",
                      level="gaussian_signal/safe_explorer_ppo/med", key="med",
                      csv=DCPRL / "gaussian_all_levels.csv"),
}


def predictor_of(arm: str) -> str:
    return "gp" if arm.startswith("partx") else ("clf" if arm.startswith("clf") else "fm")


def on_disk(prefix: str, exp: Path = None) -> dict[str, set[int]]:
    """Epochs with BOTH artifacts_v2.json (epoch finished) and the per-point npz."""
    out = {}
    for arm in ARMS:
        rd = (exp or EXP) / f"{prefix}_{arm}"
        eps = {int(d.name.split("_")[1]) for d in rd.glob("epoch_*")
               if (d / "artifacts_v2.json").exists() and (d / "full_roa_per_point.npz").exists()}
        if eps:
            out[arm] = eps
    return out


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def score_campaign(name: str, dry: bool) -> int:
    c = CAMPAIGNS[name]
    # A level whose ground-truth grid has been deleted can never be scored again.
    # The noisy_dynamics and noisy_action families were removed from the data tree
    # on 2026-08-26; their run dirs survive, so without this guard the differ sees
    # an empty CSV, decides every epoch is unscored, and dies on the missing npz.
    gt = c["root"] / c["level"] / "eval_success_prob.npz"
    if not gt.exists():
        print(f"  {name}: SKIP -- ground truth gone ({gt.parent.name})")
        return 0
    disk = on_disk(c["prefix"], c.get("exp"))
    if not disk:
        print(f"  {name}: nothing on disk yet")
        return 0
    rows = read_csv(c["csv"])
    have = defaultdict(set)
    for r in rows:
        if r["level"] == c["key"]:
            have[r["arm"]].add(int(r["epoch"]))

    missing = {a: sorted(e - have[a]) for a, e in disk.items()}
    missing = {a: e for a, e in missing.items() if e}
    if not missing:
        print(f"  {name}: up to date ({sum(len(v) for v in disk.values())} epochs scored)")
        return 0

    groups = defaultdict(list)
    for arm, eps in missing.items():
        groups[tuple(eps)].append(arm)
    n_new = sum(len(e) * len(a) for e, a in groups.items())
    print(f"  {name}: {n_new} epoch(s) to score across {len(groups)} group(s)")
    for eps, arms in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"    epochs {eps[0]}..{eps[-1]} ({len(eps)}) <- {' '.join(sorted(arms))}")
    if dry:
        return n_new

    new_rows = []
    with tempfile.TemporaryDirectory(prefix=f"score_{name}_") as td:
        td = Path(td)
        for i, (eps, arms) in enumerate(groups.items()):
            spec = [dict(predictor=predictor_of(a), level=c["level"], arm=a,
                         run_dir=str((c.get("exp") or EXP) / f"{c['prefix']}_{a}")) for a in sorted(arms)]
            sp = td / f"spec_{i}.json"
            sp.write_text(json.dumps(spec, indent=1))
            out = td / f"out_{i}"
            cmd = [PY, str(ROOT / "scripts/stoch_prob_metrics.py"), "--spec", str(sp),
                   "--data-root", str(c["root"]), "--epochs", ",".join(map(str, eps)),
                   "--out", str(out)]
            r = subprocess.run(cmd, env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin",
                                         "HOME": str(Path.home())},
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:]); print(r.stderr[-2000:], file=sys.stderr)
                raise SystemExit(f"scorer failed on {name} group {i}")
            got = out / "metrics.csv"
            if got.exists():
                new_rows += list(csv.DictReader(got.open()))
    print(f"    scored {len(new_rows)} row(s)")

    fields = list(rows[0].keys()) if rows else (
        ["predictor", "level", "arm", "epoch"] +
        [k for k in sorted(new_rows[0]) if k not in ("predictor", "level", "arm", "epoch")])
    merged = {(r["level"], r["predictor"], r["arm"], int(r["epoch"])): r for r in rows}
    clash = 0
    for r in new_rows:
        r["level"] = c["key"]
        k = (r["level"], r["predictor"], r["arm"], int(r["epoch"]))
        clash += k in merged
        merged[k] = r
    ordered = sorted(merged.values(),
                     key=lambda r: (r["level"], r["predictor"], r["arm"], int(r["epoch"])))
    c["csv"].parent.mkdir(parents=True, exist_ok=True)
    with c["csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in fields})

    depths = defaultdict(int)
    for r in ordered:
        if r["level"] == c["key"]:
            depths[r["arm"]] += 1
    dv = sorted(set(depths.values()))
    print(f"    wrote {c['csv'].name}: {len(ordered)} rows total, "
          f"{len(depths)} arms at this level, depths {dv} "
          f"{'RAGGED' if len(dv) > 1 else 'UNIFORM'} (collisions overwritten: {clash})")
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="*", default=list(CAMPAIGNS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    total = 0
    for name in (a.campaigns or list(CAMPAIGNS)):
        total += score_campaign(name, a.dry_run)
    print(f"TOTAL {'would score' if a.dry_run else 'scored'} {total} epoch(s)")


if __name__ == "__main__":
    main()
