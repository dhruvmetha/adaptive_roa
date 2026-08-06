# Acquisition-arms analysis — `/common/home/st1122/Projects/adaptive_roa/docs/plots/acquisition_arms_metrics.csv`

## Methodological findings that condition everything below

**A. Effective replicates.** Since ties-at-max-entropy < budget almost everywhere, `entropy` and `entropy_tiebreak` are the same method run twice. Their paired divergence gives a *method-level* run-to-run noise floor: **~0.007 F1 on quad2d late-window averages** (0.0072 at d2=0.5, 0.0029 at d2=1.0 on clean epochs; up to 0.03 on single mid-training windows), **~0.001 on cartpole**, ~0.002-0.004 on pendulum. Per-epoch plateau wobble: σ(F1) ≈ 0.02 quad2d (max 0.04), ≈ 0.002 pendulum/cartpole. Differences below these are noise.

**B. The invalid_pct/F1 artifact.** invalid_pct is bistable across checkpoints: it spikes episodically to 0.25-0.40 and reverts the next epoch (entropy|0.5: 0.081→0.273→0.343 over three epochs; fm_ranked epoch tt=10000: 0.208 vs 0.063-0.074 around it; fm_nonadapt sustained 0.35-0.39 at tt 8-10k). Within-run corr(F1, invalid_pct) at tt≥6000 on quad2d: **median +0.80**. Mechanism: high invalid shrinks the committed set to easy points and mechanically inflates F1. Every headline "win" for high-abstention arms is this artifact — dispersion|gd|0.5's late F1=0.974 sits on invalid 0.35-0.40; gated's 0.986s on 0.34-0.37. All comparisons below therefore use **clean epochs (invalid_pct<0.15) averaged over matched train_traj windows**, with abstention reported alongside.

## 1. Matched-budget comparison

**quad2d, clean epochs, tt∈[9500,12000]** (fm_ranked/nonadapt reach 12k, so this is matched; only fm_direct stops at 7,608):

| arm | F1 | sep_pct | FNR |
|---|---|---|---|
| fm_ranked | 0.954 | 0.104 | 0.059 |
| entropy pooled replicates, d2=0.5 | 0.948 | 0.124 | 0.072 |
| gated d2=0.5 | 0.943 | 0.124 | 0.079 |
| entropy pooled, d2=1.0 | 0.940 | 0.124 | 0.081 |
| mode_sep (both d2) | 0.938 | 0.117-0.137 | 0.087 |
| fm_nonadapt | 0.914 | 0.138 | 0.127 |
| dispersion (best/worst rule) | 0.916/0.846 | 0.14-0.16 | 0.13-0.24 |

At the fm_direct-matched window [6500,8000]: fm_ranked 0.955, fm_direct 0.952, entropy|0.5 0.952, entropy_tiebreak|0.5 0.951, gated 0.934-0.937, mode_sep 0.914-0.924, fm_nonadapt 0.928, dispersion 0.81-0.87. Abstention is comparable across these (sep 0.10-0.18) once invalid-spike epochs are excluded.

**Exceeds noise:** every non-dispersion adaptive arm > fm_nonadapt (+0.02 to +0.04); dispersion < everything (−0.04 to −0.11, FNR up to 0.30); mode_sep's deficit at the 7k budget (−0.03 vs baselines). **Within noise:** the entire fm_ranked / fm_direct / entropy / gated ordering (spread ≤0.012 vs noise floor 0.007-0.015); mode_sep's residual deficit at 11.5k (−0.016 vs fm_ranked, ~2x noise floor — marginal, and it converged from −0.03 at 7k, i.e. slower, not worse asymptotically).

**pendulum [850,1100] and cartpole [1050,1300]:** saturated as warned. All non-dispersion arms span 0.9922-0.9978 (pendulum) and 0.9911-0.9957 (cartpole). These spreads are 2-5x the tiny replicate noise but ≤0.005 absolute in a regime where sep_pct swings of 0.02-0.32 (invalid excursions hit here too, e.g. pendulum entropy|1.0 invalid=0.32 in the matched window) dwarf them. **These two systems cannot rank the non-dispersion arms; I decline to.** They do resolve: dispersion|greedy|1.0 **collapses on cartpole** (F1=0.000 from epoch 8 onward, 7 consecutive epochs, invalid_pct→0.68-0.87, TPR=0) and lags on pendulum at matched budget (0.909); fm_nonadapt is consistently bottom of the non-dispersion pack on both (marginally, ≤0.006).

## 2. Does threshold-free cost anything? **No — within measurement resolution.**

entropy (d2=0.5) vs fm_ranked/fm_direct: identical at quad2d 7k (0.952 vs 0.955/0.952), −0.006 at 11.5k (0.948 vs 0.954) with the replicate pair straddling it (0.9517/0.9445) — inside the 0.007 replicate noise floor. FNR is the one consistently ordered metric (0.059 baseline vs 0.068-0.076 entropy), suggesting a real but sub-resolvable ~1-point TPR cost. Honest statement: **threshold-free costs at most ~0.01 F1 on quad2d and nothing detectable on the saturated systems.** Given that it removes λ*, δ*, and q_hat from the training loop entirely, this is the primary objective achieved.

## 3. Does criteria-free cost anything on top? **~1-2 F1 points on quad2d, mostly transient.**

mode_sep/gated vs entropy: −0.005 to −0.010 at 11.5k (at/below noise floor), −0.02 to −0.03 at 7k for mode_sep (real — slower early learning, consistent with a label-free score wasting some early budget). Both decisively beat fm_nonadapt (+0.024 to +0.029 at 11.5k, 3-4x noise) — **criteria-free acquisition genuinely works**, unlike its predecessor: dispersion sits *below* fm_nonadapt at matched budget on quad2d (0.846-0.916 clean vs 0.914) with FNR 0.13-0.24, exactly the failure mode diagnosed. gated ≥ mode_sep everywhere measured (+0.005 to +0.02, marginal), so the idempotence gate does not hurt and possibly helps.

## 4. entropy ≡ entropy_tiebreak: **confirmed.**

Cartpole Δ=0.0006-0.0008; quad2d clean-window Δ=0.003-0.007 (late) with one 0.03 mid-training excursion at d2=1.0 — all consistent with pure run-to-run noise, as the tie-inertness predicts. The one place ties exceeded budget (pendulum d2=0.5), tiebreak reads 0.004-0.007 *lower* late (0.9913 vs 0.9985 at tt≥1700) — a single run in the saturated regime; not interpretable. **Nothing supports the tie-breaker; drop it.** Its real value in this campaign was accidental: providing replicates.

## 5. Final-state error: the dissociation survives **only in the tail**, and the mean-based version reverses.

- **Center/mean:** dispersion is now the *worst* regressor where it matters — `full_median_of_medians` 0.95-1.09 vs 0.86-0.88 for entropy/baselines at both quad2d budgets; `full_mean_of_means` worst at d2=1.0 (1.80-1.85 vs 1.73-1.76, ~5x replicate noise of ~0.016). The earlier "best regressor" finding does **not** replicate on mean aggregations.
- **Tail:** `full_p90_of_means` — dispersion d2=1.0 runs uniquely occupy the top-3 at high budget (3.79-3.85 vs 3.89-4.07 for everyone else) and are 2 of the top-3 at 7k. So dispersion's budget-dump into diffuse failure regions did buy ~2-5% lower *worst-region* regression error while costing 4-11 F1 points — a tail-only dissociation, ~1-2x noise, directionally consistent across both budgets and all three selection rules.
- **What the grid reveals that the mean hides:** `certain_failure` error is **5.6x** `certain_success` error (median ratio, quad2d; ~1.6 vs ~0.28 mean-of-means), so `full_mean_of_means` is essentially a failure-class metric — success-class regression is already excellent (0.25-0.39) for every arm and nearly arm-independent. The inner aggregation barely matters (mean_of_means ≈ tracks mean_of_medians); the *outer* choice is what discriminates. Among the new arms, all final-state differences are ≤2x replicate noise — **no dissociation for entropy/mode_sep/gated**: they match baselines on regression and classification simultaneously.

## 6. d2_ratio: the exploration floor is insurance whose value scales inversely with score quality.

Paired 0.5-vs-1.0, late windows: quad2d — dispersion +0.03 to +0.07 (large, real), gated +0.016 (marginal), entropy +0.006/+0.011 (noise-level but same sign in both replicates and in FNR), mode_sep −0.005 (noise). Cartpole/pendulum — d2=1.0 nominally better by 0.001-0.014 (noise-level), **except** cartpole dispersion|greedy where d2=0.5 is the difference between F1 0.98 and total collapse (0.00). Interaction is clear: the worse/more-biased the score, the more the random half rescues it; for a good score on an unsaturated system it costs nothing measurable. Given collapse is catastrophic and the cost is nil, **d2=0.5 is cheap insurance and should be kept.**

## 7. Recommendation

- **Objective 1 (no thresholds in the loop): adopt `entropy`, greedy_diverse, d2=0.5, as the default.** It matches fm_ranked/fm_direct within measurement noise at every matched budget on every system, removes all fitted quantities (λ*, δ*, q_hat) from training, and is the simplest arm. Drop `entropy_tiebreak` — inert by construction and by measurement.
- **Objective 2 (criteria-free): `gated` (mode_sep × idempotence) is the candidate**, at d2=0.5. It clearly beats random (+0.03 at matched budget on quad2d), fixes dispersion's failure mode, and trails entropy by ≤0.01 — but "matches the baseline" is not yet demonstrated beyond noise; "costs ≤1-2 F1 points" is.
- **Fix the evaluation before the next campaign:** the bistable invalid_pct (epoch-to-epoch swings of 0.03↔0.40) both corrupts per-epoch F1 and makes single-checkpoint comparisons of sep_pct meaningless. Either classify committed points by nearest attractor, widen `attractor_radius` at eval, or report metrics averaged over the last k checkpoints.

**What the data does not settle:** (a) anything about high-dimensional behavior — quad3d is absent and is the only pending system with headroom; every conclusion here is provisional for it, and mode_sep's slower early convergence is exactly the kind of effect that could grow with dimension; (b) any ranking among non-dispersion arms on pendulum/cartpole (saturated); (c) differences below ~0.01 F1 on quad2d — one run per cell, and the replicate pair proves single runs cannot resolve finer; (d) whether gated's edge over mode_sep is real (≤2x noise); (e) whether entropy's d2=0.5 gain on quad2d is real (noise-level, though directionally consistent). A 3-seed replication of {entropy, gated, fm_ranked} × d2=0.5 on quad2d/quad3d would settle (c)-(e).