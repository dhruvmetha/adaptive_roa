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
import argparse, csv, json, os, subprocess, sys, tempfile
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
# Separate root, and therefore a separate CSV, from D3D. The ppo campaign shares
# the `corridor_sine_ambient` family name with lqr but is a different experiment:
# no ambient term, a different level ladder (f_0.00/0.12/0.20/0.40 vs 0.25/0.30),
# a different start box, half the control rate, and eval states that are a
# farthest-point sample of mid-flight rows rather than the shipped start grid.
# The dataset json records index_aligned: false. Pooling them would be wrong.
D3DP = DOCS / "quadrotor3d/ppo"
# Same PPO controller, success criterion and ground truth as D3DP; the ONLY
# difference is the trainable pool (800,000 flights vs 100,000, see
# configs/adaptive_v2/controller/ppo_800k.yaml). A separate root and CSV so the
# pool-size ablation never collides with the 100k rows: the level keys are
# identical on both sides and would overwrite each other in one file.
D3DP8 = DOCS / "quadrotor3d/ppo_800k"
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
        # decomp_yield_mlp_bald: the yield-MLP arms rescored on BALD's epistemic
        # entropy instead of the debiased between-member variance. Same length
        # model and same alpha=1 pricing as yield_mlp / clf_yield, so read each
        # against ITS OWN variance twin, not against epi_bald.
        "yield_mlp_bald", "clf_yield_bald",
        # decomp_yield_mlp_bald_db: same arm with the K-sample MC bias removed
        # from BALD. FM only -- the classifier backend has member_sample_size
        # None, so the correction is a no-op there and no clf twin was run.
        "yield_mlp_bald_db",
        "bnn_ens", "bnn_lap", "bnn_mfvi", "bnn_mfvi_bald",
        # H_bb-a: the same BNN acquiring on total predictive entropy. This is the
        # baseline Depeweg et al. (2018) Table 1 argues against; without it the
        # decomposition-vs-total-entropy claim cannot be tested either way.
        "bnn_mfvi_total",
        # I_bb-a: BNN fitted by black-box alpha-divergence at alpha=1, the
        # posterior Depeweg et al. (2018) actually use. Also carries the
        # untempered likelihood and per-system network width, so it is NOT
        # comparable to the bnn_mfvi_* arms above -- those are alpha=0 (ELBO)
        # with a class-tempered loss and one pendulum-sized net everywhere.
        "bnn_mfvi_a1",
        # The matched uniform control for bnn_mfvi_a1: SAME predictor, acquisition=direct,
        # d2_ratio=0. It is not optional and it is not interchangeable with bnn_mfvi.
        # Reading bnn_mfvi_a1 against bnn_mfvi would vary four things at once (alpha,
        # likelihood tempering, net width, acquisition); this varies only acquisition.
        # Omitting it from this list does not error -- the differ finds the arm in no
        # registry and skips the run silently, so the contrast would simply have no
        # control and nobody would be told.
        "bnn_a1_dir00",
        # Seed replicates carry the seed in the ARM slot, matching dir00_s42/43/44.
        # Putting it in the prefix instead (pen_low_s43_*) makes the differ read the
        # arm as "s43_bnn_mfvi_a1", which is in no registry, and the run is silently
        # skipped rather than erroring.
        "bnn_mfvi_a1_s43", "bnn_mfvi_a1_s44",
        # greedy (non-diverse) selection_rule twins. Same arm, same budget, same
        # seed as the entry above them; the ONLY difference is
        # acquisition.selection_rule=greedy instead of greedy_diverse. They are
        # separate arms rather than a flag because each is read against its own
        # greedy_diverse twin, and a missing name here is skipped SILENTLY --
        # the contrast would just quietly have no rows.
        "epi_bald_greedy", "epi_var_greedy",
        "clf_epi_bald_greedy", "bnn_mfvi_a1_greedy",
        # q2d_csbase only: the LIVE FM runs carry a `_fast` suffix (Q3D-style
        # cached-dataset path); the bare-name dirs at that prefix are cancelled
        # 2-3 epoch twins. Scored under their own names; the paper scripts alias
        # `<arm>_fast` -> `<arm>` at that level and drop the bare rows.
        "dir00_s42_fast", "epi_bald_greedy_fast", "epi_var_greedy_fast",
        # timeout-fix seeds 43 and 44 (2026-09-12). The s42 campaign ran every
        # cell at one seed; these are the other two, in the SAME experiment root,
        # with the seed carried in the arm name. The uniform control needs no
        # entry because dir00_s43/s44 are already listed above -- that asymmetry
        # is the whole reason the adaptive arms need explicit names here, and a
        # name missing from this list is skipped SILENTLY, so an unregistered
        # seed would look like a run that never banked anything.
        "epi_bald_greedy_s43", "epi_bald_greedy_s44",
        "bnn_mfvi_a1_greedy_s43", "bnn_mfvi_a1_greedy_s44",
        "clf_epi_bald_greedy_s43", "clf_epi_bald_greedy_s44"]

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
    # Prefixes must never be an underscore-extension of another campaign's, or the
    # shorter one globs the longer one's dirs and the differ reads the arm as
    # "<leftover>_<arm>", finds it in no registry, and skips the run silently.
    # Hence q2d_csbase / q2d_cssharp rather than q2d_cs_base / q2d_cs_sharp.
    "q2d_csbase": dict(prefix="q2d_csbase", root=DATA / "quadrotor2D",
                       level="corridor_sine_ambient/rl/baseline",
                       key="corridor_sine_ambient_baseline",
                       csv=D2D / "quad2d_corridor_sine_ambient_all_levels.csv"),
    "q2d_cssharp": dict(prefix="q2d_cssharp", root=DATA / "quadrotor2D",
                        level="corridor_sine_ambient/rl/sharp",
                        key="corridor_sine_ambient_sharp",
                        csv=D2D / "quad2d_corridor_sine_ambient_all_levels.csv"),
    # "loud" (built 2026-09-11) is the high-noise member of this family, but it is
    # NOT a drop-in fourth column next to smooth/sharp, for two measured reasons:
    #   1. Its oracle uses 20 MC trials per eval point, against 50 for smooth and
    #      sharp, so p_success is quantised to 0.05 instead of 0.02. KL against a
    #      coarser oracle carries more quantisation noise and is not comparable
    #      term-for-term with the 50-trial levels.
    #   2. Base rate 0.0358, against smooth 0.0593 and sharp 0.0602. smooth/sharp
    #      were matched to 0.001 BY CONSTRUCTION to kill the base-rate confound;
    #      loud breaks that pairing, so any loud-vs-smooth gap is confounded by
    #      base rate the way the nd/cs pair was in base-rate-ordering-does-not-survive.
    "q2d_csloud": dict(prefix="q2d_csloud", root=DATA / "quadrotor2D",
                       level="corridor_sine_ambient/rl/loud",
                       key="corridor_sine_ambient_loud",
                       csv=D2D / "quad2d_corridor_sine_ambient_all_levels.csv"),
    "q3d_cs025": dict(prefix="q3d_cs025", root=DATA / "quadrotor3D",
                      level="corridor_sine_ambient/lqr/f_0.25",
                      key="corridor_sine_ambient_f_0.25",
                      csv=D3D / "quad3d_corridor_sine_ambient_all_levels.csv"),
    "q3d_cs030": dict(prefix="q3d_cs030", root=DATA / "quadrotor3D",
                      level="corridor_sine_ambient/lqr/f_0.30",
                      key="corridor_sine_ambient_f_0.30",
                      csv=D3D / "quad3d_corridor_sine_ambient_all_levels.csv"),
    # PPO controller, wave 1. f_0.00 is DETERMINISTIC -- no wind and one eval trial
    # per state -- so its KL is a clipped log-loss against binary truth and is not
    # comparable to the noisy levels, exactly like cprl `baseline`.
    "q3dppo_cs000": dict(prefix="q3dppo_cs000", root=DATA / "quadrotor3D",
                         level="corridor_sine_ambient/ppo/f_0.00",
                         key="corridor_sine_ambient_f_0.00",
                         csv=D3DP / "quad3dppo_corridor_sine_ambient_all_levels.csv"),
    "q3dppo_cs020": dict(prefix="q3dppo_cs020", root=DATA / "quadrotor3D",
                         level="corridor_sine_ambient/ppo/f_0.20",
                         key="corridor_sine_ambient_f_0.20",
                         csv=D3DP / "quad3dppo_corridor_sine_ambient_all_levels.csv"),
    "q3dppo_cs012": dict(prefix="q3dppo_cs012", root=DATA / "quadrotor3D",
                         level="corridor_sine_ambient/ppo/f_0.12",
                         key="corridor_sine_ambient_f_0.12",
                         csv=D3DP / "quad3dppo_corridor_sine_ambient_all_levels.csv"),
    "q3dppo_cs040": dict(prefix="q3dppo_cs040", root=DATA / "quadrotor3D",
                         level="corridor_sine_ambient/ppo/f_0.40",
                         key="corridor_sine_ambient_f_0.40",
                         csv=D3DP / "quad3dppo_corridor_sine_ambient_all_levels.csv"),

    # 800k-pool twin of the q3dppo campaign (pool-size ablation). Run dirs live
    # on Amarel /scratch AND in the shared tree, and after mid-campaign
    # relocations NEITHER side is authoritative for every arm (2026-09-09,
    # q3d-800k session): this scorer reads the shared tree only, so an arm whose
    # Amarel copy is ahead scores short until the sync lands. Incremental by
    # construction, so a later pass fills the gap; never read a depth off this
    # CSV as final while the campaign is in flight. The *_greedy arms are a
    # second cohort (selection_rule=greedy); a pool-size-only comparison must
    # exclude them rather than glob the prefix.
    "q3d800k_cs000": dict(prefix="q3d800k_cs000", root=DATA / "quadrotor3D",
                          level="corridor_sine_ambient/ppo/f_0.00",
                          key="corridor_sine_ambient_f_0.00",
                          csv=D3DP8 / "quad3d800k_corridor_sine_ambient_all_levels.csv"),
    "q3d800k_cs012": dict(prefix="q3d800k_cs012", root=DATA / "quadrotor3D",
                          level="corridor_sine_ambient/ppo/f_0.12",
                          key="corridor_sine_ambient_f_0.12",
                          csv=D3DP8 / "quad3d800k_corridor_sine_ambient_all_levels.csv"),
    "q3d800k_cs020": dict(prefix="q3d800k_cs020", root=DATA / "quadrotor3D",
                          level="corridor_sine_ambient/ppo/f_0.20",
                          key="corridor_sine_ambient_f_0.20",
                          csv=D3DP8 / "quad3d800k_corridor_sine_ambient_all_levels.csv"),
    "q3d800k_cs040": dict(prefix="q3d800k_cs040", root=DATA / "quadrotor3D",
                          level="corridor_sine_ambient/ppo/f_0.40",
                          key="corridor_sine_ambient_f_0.40",
                          csv=D3DP8 / "quad3d800k_corridor_sine_ambient_all_levels.csv"),

    # Ambient-wobble variants, added 2026-09-10. Same twin-curtain wind at the
    # same f_max as the plain level of the matching number, PLUS an ungated
    # ambient wobble -- so f_0.12 and f_0.12_a0.03 are a controlled pair (hold
    # the corridor, add ambient) and belong in the SAME csv, separated by the
    # level key rather than by file.
    #
    # Two ways these differ from every q3d800k entry above, both load-bearing:
    #   1. `level` points at ppo_800k/, NOT ppo/. The plain levels can borrow
    #      ppo/ ground truth because the 800k pool is a pool-size ablation with
    #      an identical policy, hence an identical ROA. The _a levels exist ONLY
    #      under ppo_800k/ and ship their own eval_success_prob.npz, so there is
    #      no ppo/ counterpart to borrow.
    #   2. prefix is `q3damb_*`, deliberately NOT `q3d800k_*`. A q3d800k prefix
    #      would be glob-matched by the existing `q3d800k_cs0NN*` patterns and
    #      would also pull these arms into another session's monitor and
    #      auto-resume.
    #
    # Run dirs are SPLIT: the FM bald arms for f_0.20_a0.04 and f_0.40_a0.04 ran
    # on iLab straight into the shared tree, the other 19 arms ran on Amarel
    # /scratch. This scorer reads the shared tree only, so the Amarel side scores
    # as absent until it is rsynced into EXP.
    "q3damb_f012a03": dict(prefix="q3damb_f012a03", root=DATA / "quadrotor3D",
                           level="corridor_sine_ambient/ppo_800k/f_0.12_a0.03",
                           key="corridor_sine_ambient_f_0.12_a0.03",
                           csv=D3DP8 / "quad3d800k_corridor_sine_ambient_all_levels.csv"),
    "q3damb_f020a04": dict(prefix="q3damb_f020a04", root=DATA / "quadrotor3D",
                           level="corridor_sine_ambient/ppo_800k/f_0.20_a0.04",
                           key="corridor_sine_ambient_f_0.20_a0.04",
                           csv=D3DP8 / "quad3d800k_corridor_sine_ambient_all_levels.csv"),
    "q3damb_f040a04": dict(prefix="q3damb_f040a04", root=DATA / "quadrotor3D",
                           level="corridor_sine_ambient/ppo_800k/f_0.40_a0.04",
                           key="corridor_sine_ambient_f_0.40_a0.04",
                           csv=D3DP8 / "quad3d800k_corridor_sine_ambient_all_levels.csv"),

    # The zero-noise floor of this family: alpha=0, beta=0, collected 2026-09-11
    # with the SAME generator, success rule, horizon, control rates, resolution
    # and 100k pool as low/med/high, so it is comparable to them cell-for-cell
    # (its eval grid is bit-identical -- np.array_equal on the 49,770 starts).
    #
    # TWO THINGS DIFFER FROM ITS SIBLINGS AND BOTH MATTER WHEN QUOTING IT:
    #   1. trials = 1, not 100. The dynamics are deterministic, so every rollout
    #      from a cell is identical and K>1 buys nothing. p_success is therefore
    #      exactly binary (verified: 0.0000 interior mass) and its KL is a
    #      clipped log-loss against binary truth -- NOT comparable term-for-term
    #      with the K=100 levels, the same caveat that applies to q3dppo f_0.00
    #      and q2d baseline.
    #   2. It is NOT the old deterministic/pendulum/lqr tree. That one labels
    #      success by a 0.075 radius threshold; this family uses the box rule
    #      (|theta| < BOX_TOL[0], |theta_dot| < BOX_TOL[1]) that --noise_beta
    #      switches on. Scoring the old tree here would read a different
    #      question's labels as this family's floor.
    # Mean p_success 0.3860, which reproduces the deterministic reference the
    # family README cites as noisy_torque/lqr/tau_0.00 -- a dataset that is not
    # on disk anywhere under DATA_DIR.
    "pend_base": dict(exp=GT, prefix="pen_base", root=DATA / "pendulum",
                      level="gaussian_signal/lqr/baseline", key="baseline",
                      csv=DPEN / "gaussian_all_levels.csv"),
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
    "cprl_low":  dict(exp=GT, prefix="cprl_low", root=DATA / "cartpole",
                      level="gaussian_signal/safe_explorer_ppo/low", key="low",
                      csv=DCPRL / "gaussian_all_levels.csv"),
    # cprl_high runs on AMAREL and is rsynced into gaussian_torque/, so it only
    # appears here after scripts/ sync. A "nothing on disk yet" for this campaign
    # may mean the sync has not run, not that the runs failed.
    "cprl_high": dict(exp=GT, prefix="cprl_high", root=DATA / "cartpole",
                      level="gaussian_signal/safe_explorer_ppo/high", key="high",
                      csv=DCPRL / "gaussian_all_levels.csv"),
}

# ---------------------------------------------------------------- timeout fix
# The 2026-09-11 rerun with `data_source.timeout_intermediates=drop` (commit
# f16314f): a trajectory cut at the collector horizon without succeeding keeps
# only (x_0, x_T) instead of pairing every intermediate state with x_T.
#
# SEPARATE CSVs, in their own docs subtree. These rows are NOT comparable with
# the rows above and must never land in the same file: the training pairs differ,
# the budgets differ (11 epochs at a new schedule per system), and q3d also moved
# to a different dataset (ppo_1500K, 1.5M-trajectory pool) -- so a shared file
# would let a paper script average two different experiments under one arm name.
# Run dirs live under EXPROOT/timeout_fix, hence exp=TFEXP for every entry.
TFEXP = EXPROOT / "timeout_fix"
DTF = DOCS / "timeout_fix"
CAMPAIGNS.update({
    "tf_cp_base": dict(exp=TFEXP, prefix="tf_cp_base", root=DATA / "cartpole",
                       level="gaussian_signal/safe_explorer_ppo/baseline", key="baseline",
                       csv=DTF / "cartpole_safe_explorer_ppo_all_levels.csv"),
    "tf_cp_low": dict(exp=TFEXP, prefix="tf_cp_low", root=DATA / "cartpole",
                      level="gaussian_signal/safe_explorer_ppo/low", key="low",
                      csv=DTF / "cartpole_safe_explorer_ppo_all_levels.csv"),
    "tf_cp_med": dict(exp=TFEXP, prefix="tf_cp_med", root=DATA / "cartpole",
                      level="gaussian_signal/safe_explorer_ppo/med", key="med",
                      csv=DTF / "cartpole_safe_explorer_ppo_all_levels.csv"),
    "tf_cp_high": dict(exp=TFEXP, prefix="tf_cp_high", root=DATA / "cartpole",
                       level="gaussian_signal/safe_explorer_ppo/high", key="high",
                       csv=DTF / "cartpole_safe_explorer_ppo_all_levels.csv"),
    "tf_pen_base": dict(exp=TFEXP, prefix="tf_pen_base", root=DATA / "pendulum",
                        level="gaussian_signal/lqr/baseline", key="baseline",
                        csv=DTF / "pendulum_lqr_all_levels.csv"),
    "tf_pen_low": dict(exp=TFEXP, prefix="tf_pen_low", root=DATA / "pendulum",
                       level="gaussian_signal/lqr/low", key="low",
                       csv=DTF / "pendulum_lqr_all_levels.csv"),
    "tf_pen_med": dict(exp=TFEXP, prefix="tf_pen_med", root=DATA / "pendulum",
                       level="gaussian_signal/lqr/med", key="med",
                       csv=DTF / "pendulum_lqr_all_levels.csv"),
    "tf_pen_high": dict(exp=TFEXP, prefix="tf_pen_high", root=DATA / "pendulum",
                        level="gaussian_signal/lqr/high", key="high",
                        csv=DTF / "pendulum_lqr_all_levels.csv"),
    "tf_q2d_base": dict(exp=TFEXP, prefix="tf_q2d_base", root=DATA / "quadrotor2D",
                        level="corridor_sine_ambient/rl/baseline",
                        key="corridor_sine_ambient_baseline",
                        csv=DTF / "quad2d_corridor_sine_ambient_all_levels.csv"),
    "tf_q2d_smooth": dict(exp=TFEXP, prefix="tf_q2d_smooth", root=DATA / "quadrotor2D",
                          level="corridor_sine_ambient/rl/smooth",
                          key="corridor_sine_ambient_smooth",
                          csv=DTF / "quad2d_corridor_sine_ambient_all_levels.csv"),
    "tf_q2d_loud": dict(exp=TFEXP, prefix="tf_q2d_loud", root=DATA / "quadrotor2D",
                        level="corridor_sine_ambient/rl/loud",
                        key="corridor_sine_ambient_loud",
                        csv=DTF / "quad2d_corridor_sine_ambient_all_levels.csv"),
    # q3d on ppo_1500K. The pool is the whole 1.5M collection, not the 100k
    # subset the retired runs used, so these share no arm with the q3d800k or
    # q3damb CSVs either.
    "tf_q3d_f000": dict(exp=TFEXP, prefix="tf_q3d_f000", root=DATA / "quadrotor3D",
                        level="corridor_sine_ambient/ppo_1500K/f_0.00",
                        key="corridor_sine_ambient_f_0.00",
                        csv=DTF / "quad3d_ppo1500k_corridor_sine_ambient_all_levels.csv"),
    "tf_q3d_f012a03": dict(exp=TFEXP, prefix="tf_q3d_f012a03", root=DATA / "quadrotor3D",
                           level="corridor_sine_ambient/ppo_1500K/f_0.12_a0.03",
                           key="corridor_sine_ambient_f_0.12_a0.03",
                           csv=DTF / "quad3d_ppo1500k_corridor_sine_ambient_all_levels.csv"),
    "tf_q3d_f020a035": dict(exp=TFEXP, prefix="tf_q3d_f020a035", root=DATA / "quadrotor3D",
                            level="corridor_sine_ambient/ppo_1500K/f_0.20_a0.035",
                            key="corridor_sine_ambient_f_0.20_a0.035",
                            csv=DTF / "quad3d_ppo1500k_corridor_sine_ambient_all_levels.csv"),
    "tf_q3d_f040a04": dict(exp=TFEXP, prefix="tf_q3d_f040a04", root=DATA / "quadrotor3D",
                           level="corridor_sine_ambient/ppo_1500K/f_0.40_a0.04",
                           key="corridor_sine_ambient_f_0.40_a0.04",
                           csv=DTF / "quad3d_ppo1500k_corridor_sine_ambient_all_levels.csv"),
})

# The 40k q3d campaign (10k initial + 10 x 3k). Same dataset, levels and
# timeout policy as tf_q3d_*, but a completely different budget, so it gets its
# OWN exp root, its OWN docs dir and its OWN CSVs. Sharing a CSV with the 15k
# runs would put two budgets under one arm name and let a paper script average
# them -- the same trap the tf_* block was carved out to avoid.
TF40EXP = EXPROOT / "timeout_fix_40k"
DTF40 = DOCS / "timeout_fix_40k"
CAMPAIGNS.update({
    f"tf40_q3d_{tag}": dict(exp=TF40EXP, prefix=f"tf40_q3d_{tag}", root=DATA / "quadrotor3D",
                            level=f"corridor_sine_ambient/ppo_1500K/{level}",
                            key=f"corridor_sine_ambient_{level}",
                            csv=DTF40 / "quad3d_40k_corridor_sine_ambient_all_levels.csv")
    for tag, level in [("f000", "f_0.00"), ("f012a03", "f_0.12_a0.03"),
                       ("f020a035", "f_0.20_a0.035"), ("f040a04", "f_0.40_a0.04")]
})


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


# A wide row without this column predates the level-set metrics. The differ
# treats such a row as unscored, so one ordinary monitoring pass backfills every
# campaign after a scorer upgrade -- no flag, no manual CSV deletion.
REQUIRED_COL = "auc_bal_acc"
LEAD = ["predictor", "level", "arm", "epoch"]
LEAD_LEVELS = LEAD + ["beta"]


def scored_epochs(rows: list[dict], level_key: str,
                  required_col: str = REQUIRED_COL,
                  rescore: bool = False) -> dict[str, set[int]]:
    """Epochs already scored at this level, per arm. A blank required column
    counts as missing: DictReader hands back '' for a column the row predates.
    `rescore` reports nothing as scored, for a deliberate full re-projection
    after a metric definition changes without adding a column."""
    have: dict[str, set[int]] = defaultdict(set)
    if rescore:
        return {}
    for r in rows:
        if r["level"] == level_key and r.get(required_col, "") != "":
            have[r["arm"]].add(int(r["epoch"]))
    return dict(have)


def merge_rows(old: list[dict], new: list[dict], key, lead: list[str]
               ) -> tuple[list[dict], list[str], int]:
    """Key-based merge, NEW wins. Returns (rows sorted by key, fields, clashes).

    Field order is the old file's order with any new columns appended, so a
    column a scorer upgrade adds lands in the file instead of being silently
    dropped by DictWriter's extrasaction='ignore'.
    """
    fields = list(old[0].keys()) if old else list(lead)
    fields += sorted({k for r in new for k in r} - set(fields))
    merged = {key(r): r for r in old}
    clash = 0
    for r in new:
        k = key(r)
        clash += k in merged
        merged[k] = r
    ordered = sorted(merged.values(), key=key)
    # A column no surviving row fills is a retired metric (a --rescore replaced
    # the rows that carried it); keep the key columns, drop the rest.
    fields = [f for f in fields
              if f in lead or any(r.get(f, "") != "" for r in ordered)]
    return ordered, fields, clash


def levelsets_path(csv_path: Path) -> Path:
    """The long-format companion: one row per (arm, epoch, level beta)."""
    return csv_path.with_name(csv_path.stem + "_levelsets" + csv_path.suffix)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Replace `path` atomically so a concurrent reader never sees it half-written.

    A plain open(path, "w") truncates before the first byte lands, and these
    CSVs are read by plot_stoch_all_levels.py / plot_stoch_levelsets.py, which
    the q3dppo monitor runs on its own schedule. On 2026-09-06 a scoring pass
    with 12 epochs to write overlapped the monitor's figure step: three of the
    four levels read back empty and the published all-levels PNG dropped from
    four panels to one. Nothing errored -- the figure was simply wrong until
    the next redraw.

    os.replace swaps the completed temp file in as a single step, so a reader
    gets either the whole old file or the whole new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp is 0600; the docs tree is group-shared with login-bekris, so carry
    # the destination's mode over (0o660 for a file that does not exist yet).
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o660
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def score_campaign(name: str, dry: bool, rescore: bool = False, jobs: int = 1) -> int:
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
    have = scored_epochs(rows, c["key"], rescore=rescore)

    missing = {a: sorted(e - have.get(a, set())) for a, e in disk.items()}
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

    new_rows, new_levels, group_out = [], [], {}
    with tempfile.TemporaryDirectory(prefix=f"score_{name}_") as td:
        td = Path(td)
        cmds = []
        for i, (eps, arms) in enumerate(groups.items()):
            spec = [dict(predictor=predictor_of(a), level=c["level"], arm=a,
                         run_dir=str((c.get("exp") or EXP) / f"{c['prefix']}_{a}")) for a in sorted(arms)]
            sp = td / f"spec_{i}.json"
            sp.write_text(json.dumps(spec, indent=1))
            out = td / f"out_{i}"
            cmds.append((i, out, [PY, str(ROOT / "scripts/stoch_prob_metrics.py"), "--spec", str(sp),
                                  "--data-root", str(c["root"]), "--epochs", ",".join(map(str, eps)),
                                  "--out", str(out)]))

        def _run(item):
            i, out, cmd = item
            r = subprocess.run(cmd, env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin",
                                         "HOME": str(Path.home())},
                               capture_output=True, text=True)
            return i, out, r

        # Groups are independent subprocesses with their own out dirs, so they
        # can run concurrently; the CSV merge below is still a single writer.
        # jobs=1 (the default, what the monitors call) keeps the serial order.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            results = list(ex.map(_run, cmds))
        for i, out, r in sorted(results, key=lambda t: t[0]):
            if r.returncode != 0:
                print(r.stdout[-2000:]); print(r.stderr[-2000:], file=sys.stderr)
                raise SystemExit(f"scorer failed on {name} group {i}")
            new_rows += read_csv(out / "metrics.csv")
            new_levels += read_csv(out / "level_sets.csv")
            group_out[i] = r.stdout
    print(f"    scored {len(new_rows)} row(s), {len(new_levels)} level row(s)")

    # A subprocess that exits 0 having written no rows is NOT "nothing to score":
    # we already decided there was work. It means the scorer could not use the
    # epochs it found -- on 2026-09-10 an epoch's artifacts_v2.json landed mode
    # 0600 under another user, the read raised PermissionError inside the child,
    # and this function printed "scored 0 row(s)" and rewrote the CSV, exit 0.
    # Silence on unreadable input is indistinguishable from success, so name the
    # gap, replay the child's stdout (otherwise discarded on returncode 0), and
    # fail hard when nothing at all came back.
    produced = {(r["arm"], int(r["epoch"])) for r in new_rows}
    wanted = {(a, e) for eps, arms in groups.items() for a in arms for e in eps}
    gap = sorted(wanted - produced)
    if gap:
        print(f"    *** {len(gap)} of {len(wanted)} epoch(s) FOUND BUT NOT SCORED ***")
        for arm, ep in gap:
            d = (c.get("exp") or EXP) / f"{c['prefix']}_{arm}" / f"epoch_{ep:03d}"
            why = []
            for f in ("artifacts_v2.json", "full_roa_per_point.npz"):
                p = d / f
                why.append(f"{f} {'missing' if not p.exists() else 'unreadable' if not os.access(p, os.R_OK) else 'ok'}")
            print(f"      {arm} epoch {ep}: {', '.join(why)}")
        for i, so in sorted(group_out.items()):
            if so.strip():
                print(f"      --- group {i} stdout ---\n{so[-1500:]}")
    if wanted and not new_rows:
        raise SystemExit(f"scorer produced no rows for {name} despite {len(wanted)} epoch(s) to score")

    for r in new_rows + new_levels:
        r["level"] = c["key"]
    ordered, fields, clash = merge_rows(
        rows, new_rows, lambda r: (r["level"], r["predictor"], r["arm"], int(r["epoch"])), LEAD)
    write_csv(c["csv"], fields, ordered)

    # The long-format companion is merged the same way, keyed one level deeper.
    lp = levelsets_path(c["csv"])
    lordered, lfields, _ = merge_rows(
        read_csv(lp), new_levels,
        lambda r: (r["level"], r["predictor"], r["arm"], int(r["epoch"]), float(r["beta"])),
        LEAD_LEVELS)
    write_csv(lp, lfields, lordered)

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
    ap.add_argument("--rescore", action="store_true",
                    help="re-project every epoch on disk, overwriting existing rows; "
                         "for a metric-definition change that adds no new column")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run a campaign's arm groups as this many concurrent scorer "
                         "subprocesses (default 1, serial); the CSV write stays single-writer")
    a = ap.parse_args()
    total, failed = 0, []
    # One campaign that cannot read its epochs must not cost the others their
    # pass: score every campaign, then report the failures and exit nonzero so
    # the caller retries. Letting SystemExit propagate here would have made a
    # single unreadable epoch in the first campaign skip all the rest.
    for name in (a.campaigns or list(CAMPAIGNS)):
        try:
            total += score_campaign(name, a.dry_run, a.rescore, a.jobs)
        except SystemExit as e:
            print(f"  {name}: FAILED -- {e}")
            failed.append(name)
    print(f"TOTAL {'would score' if a.dry_run else 'scored'} {total} epoch(s)")
    if failed:
        raise SystemExit(f"FAILED campaigns: {' '.join(failed)}")


if __name__ == "__main__":
    main()
