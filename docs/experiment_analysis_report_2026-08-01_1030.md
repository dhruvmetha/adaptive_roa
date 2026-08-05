# Threshold-free scoring of the matched-budget arms

*Generated 2026-08-01 1030 · source `docs/plots/matched_budget_metrics.csv`*

## What this is

The matched-budget table previously scored every arm at a single tuned operating point (λ±δ → F1). That conflates two very different things: how well a model **ranks** states, and where its decision threshold happens to land. This adds four threshold-free scores — **AUC** and **AUPRC** (rank-based, invariant to any monotone recalibration) and **Brier** and **log score** (which additionally charge for miscalibration) — so the two can be told apart.

**Coverage: 54 of 54 arms — complete.** `artifacts_v2.json` stores only aggregates and every run here predates the `threshold_free` block, so the scores were recomputed from the per-point probabilities each eval wrote to disk (`full_roa_per_point.npz`, or the raw MC draws in `mc_cache/` for the older FM baselines). Eight arms had no per-point file on iLab: those runs executed on Amarel and only their JSONs were rsynced back, so the missing 1.2 GB of `.npz` was pulled from `/scratch/st1122/adaptive_roa/experiments/` to finish the table.

**Every recovered row is verified, not assumed.** For each arm the stored λ/δ rule was replayed on the recovered probabilities and the resulting tp/tn/fp/fn compared against the artifact's own confusion matrix; all 54 reproduce it exactly, which proves the probabilities are the same array the recorded metrics came from. The metric implementations themselves agree with scikit-learn to machine precision (≤1.1e-16 on AUC/AUPRC/Brier/log-loss).

## Two things to read the numbers with

**1. Brier and log score are not comparable across arms with different K.** An arm that averages K MC samples can only emit `k/K`, so its confidence — and hence its best possible log score — is capped. Under the Krichevsky–Trofimov smoothing used here the floor is **0.0241 nats at K=20** and **0.0050 at K=100**; the classifier arms (K=1, a continuous score) have no floor at all. The `excess` column in each table below is `log score − floor(K)`: it measures how close an arm is to the best its own MC budget allows. On CartPole `fm_ranked` trails `clf_adaptive` on raw log score (0.0390 vs 0.0358) but sits almost on its own ceiling (excess 0.0149 vs 0.0358) — so the raw gap is mostly quantisation, and `fm_ranked` would likely overtake it at higher K. Read `excess` as "how much of the gap is the model's fault rather than K's", **not** as a corrected ranking: `clf_adaptive` still wins the raw log score outright, and the two split the other calibration metrics (Brier favours `fm_ranked` 0.0048 vs 0.0081, AUC favours `clf_adaptive` 0.9995 vs 0.9989).

**2. AUC flatters the imbalanced systems.** Success base rates are **pendulum 38.6%**, **cartpole 18.0%**, **quad2d 8.0%**. A random ranker scores AUPRC equal to the base rate, so on Quad2D the meaningful range is 0.08→1.0 and an AUPRC of 0.93 is genuinely short of ceiling even though AUC reads 0.99.

## Headline finding: F1 rank ≠ ranking quality, and abstention is why

Within-system rank correlation between F1 and the threshold-free scores:

| system | n | ρ(F1, AUC) | ρ(F1, AUPRC) | ρ(sep%, AUC-rank − F1-rank) | p |
|---|---|---|---|---|---|
| pendulum | 18 | +0.682 | +0.690 | +0.492 | 0.038 |
| cartpole | 18 | +0.734 | +0.746 | +0.296 | 0.233 |
| quad2d | 18 | +0.146 | +0.439 | +0.647 | 0.004 |
| **pooled** | **54** | | | **+0.403** | **0.0025** |

The last column tests a specific mechanism: λ±δ lets an arm **abstain** (the separatrix), and F1 is computed only on the points it commits to. An arm that abstains on the hard cases is scored on an easier subset. The pooled correlation between abstention rate and "how much better an arm looks on F1 than on AUC" is **ρ=+0.403 (p=0.0025, n=54)** — so this is a real effect, not noise. On Quad2D it is strong enough (ρ=+0.601) that the F1 ordering carries almost no information about ranking quality.

Practical consequence: **a low F1 with a high AUC is a threshold problem and is recoverable by retuning λ/δ; a low AUC is a model problem and is not.**

## pendulum

**Target budget:** 500 trajectories · **base rate:** 38.6% success · **scored:** 18/18 arms

### Verdict

Pendulum is saturated. Every arm except the two worst dispersion variants ranks success from failure almost perfectly (AUC 0.96–1.00), and the adaptive classifier is near-perfect (AUC 0.99996, AUPRC 0.9999 — it rounds to 1.0000 in the table but is not exactly 1). At this budget the task no longer discriminates between methods — differences in headline F1 are mostly differences in how much each arm abstains, not in how well it actually scores states. The one genuine failure is `fm_disp_greedy_d10` (AUC 0.7240): its F1 of 0.4290 is not a mis-set threshold, the underlying ranking is broken and no retuning will fix it.

### Technical analysis

Class balance is the most forgiving of the three (38.6% success), so AUC and AUPRC track each other closely. Rank agreement with F1 is the highest here (ρ=+0.682) but still far from 1: `clf_nonadapt`, `fm_direct` and `fm_ranked` each sit 7–9 places lower on F1 than on AUC, i.e. F1 under-sells all three. `clf_nonadapt` is the sharpest illustration — it abstains on just 0.5% of points, so its F1 of 0.9781 is earned on essentially the whole eval set, while two arms ranked above it on F1 (`entropy_tiebreak` d2=1.0 and `entropy` d2=1.0) abstain on 41.9% and 54.3% of points respectively. The d05 dispersion arms hold up (AUC 0.98–0.99) while their d10 counterparts collapse — a consistent d2-ratio effect visible on every campaign-A pair.

Largest F1-vs-AUC rank disagreements:

- `clf_nonadapt` (d2=-): F1 rank **13/18** → AUC rank **4/18** — F1=0.9781, AUC=0.9970, abstains on 0.5% of points
- `fm_direct` (d2=baseline): F1 rank **14/18** → AUC rank **6/18** — F1=0.9741, AUC=0.9957, abstains on 5.9% of points
- `fm_ranked` (d2=baseline): F1 rank **15/18** → AUC rank **8/18** — F1=0.9727, AUC=0.9941, abstains on 11.8% of points

| arm | d2 | K | AUC | AUPRC | Brier | log score | excess | F1 | sep% |
|---|---|---|---|---|---|---|---|---|---|
| clf_adaptive | - | 1 | 1.0000 | 0.9999 | 0.0019 | 0.0067 | 0.0067 | 0.9986 | 0.004 |
| partx | - | 1 | 0.9999 | 0.9998 | 0.0064 | 0.0260 | 0.0260 | 0.9977 | 0.012 |
| entropy_tiebreak | 0.5 | 100 | 0.9980 | 0.9974 | 0.0186 | 0.0748 | 0.0698 | 0.9988 | 0.053 |
| clf_nonadapt | - | 1 | 0.9970 | 0.9924 | 0.0171 | 0.1476 | 0.1476 | 0.9781 | 0.005 |
| fm_nonadapt | baseline | 20 | 0.9968 | 0.9941 | 0.0210 | 0.0847 | 0.0606 | 0.9932 | 0.080 |
| fm_direct | baseline | 20 | 0.9957 | 0.9948 | 0.0106 | 0.0593 | 0.0352 | 0.9741 | 0.059 |
| fm_disp_prop_d05 | campaignA | 100 | 0.9949 | 0.9938 | 0.0343 | 0.1319 | 0.1269 | 0.9919 | 0.060 |
| fm_ranked | baseline | 20 | 0.9941 | 0.9929 | 0.0132 | 0.0715 | 0.0474 | 0.9727 | 0.118 |
| fm_disp_greedy_d05 | campaignA | 100 | 0.9930 | 0.9914 | 0.0389 | 0.1527 | 0.1478 | 0.9917 | 0.107 |
| entropy | 0.5 | 100 | 0.9916 | 0.9884 | 0.0270 | 0.1059 | 0.1009 | 0.9904 | 0.057 |
| mode_sep | 0.5 | 100 | 0.9857 | 0.9827 | 0.0425 | 0.1758 | 0.1708 | 0.9886 | 0.084 |
| fm_disp_gdiv_d05 | campaignA | 100 | 0.9844 | 0.9812 | 0.0564 | 0.2203 | 0.2153 | 0.9844 | 0.116 |
| entropy_tiebreak | 1.0 | 100 | 0.9830 | 0.9795 | 0.0431 | 0.1745 | 0.1695 | 0.9892 | 0.419 |
| fm_disp_prop_d10 | campaignA | 100 | 0.9751 | 0.9705 | 0.0466 | 0.2021 | 0.1971 | 0.9650 | 0.044 |
| entropy | 1.0 | 100 | 0.9707 | 0.9605 | 0.0772 | 0.2762 | 0.2713 | 0.9852 | 0.543 |
| mode_sep | 1.0 | 100 | 0.9589 | 0.9518 | 0.0659 | 0.2791 | 0.2741 | 0.9871 | 0.126 |
| fm_disp_gdiv_d10 | campaignA | 100 | 0.9122 | 0.8902 | 0.2233 | 0.8149 | 0.8100 | 0.9104 | 0.389 |
| fm_disp_greedy_d10 | campaignA | 100 | 0.7240 | 0.5906 | 0.3013 | 1.3184 | 1.3134 | 0.4290 | 0.647 |

*`excess` = log score − the floor imposed by that arm's K; lower is better and is the cross-K-comparable calibration number.*

## cartpole

**Target budget:** 1000 trajectories · **base rate:** 18.0% success · **scored:** 18/18 arms

### Verdict

CartPole is the campaign-B arms' best showing, and the threshold-free view is what proves it. They take five of the top six AUC slots, with `mode_sep` d2=1.0 first outright (AUC 0.9996, AUPRC 0.9985) and `entropy_tiebreak` d2=1.0 second - ahead of the adaptive classifier and both FM baselines. Crucially this is *not* the abstention artefact seen elsewhere: `mode_sep` d2=1.0 abstains on only 6.7% of points, so its ranking advantage is earned rather than bought by declining the hard cases. The campaign-B arms also hold the best calibration once the MC budget is accounted for - `entropy` d2=1.0 has the lowest excess log score on this system (0.0145). The d10 dispersion arms remain the laggards (AUC 0.79-0.92).

### Technical analysis

At 18% success the AUC/AUPRC gap opens up (e.g. `fm_nonadapt` 0.9845 -> 0.9654). With all 18 arms scored, rank agreement with F1 is the highest of the three systems (rho=+0.734) and the abstention effect is weak and not significant (rho=+0.296, p=0.233) - CartPole is the one system where the F1 ordering is broadly trustworthy. The exception is instructive: `fm_disp_gdiv_d10` abstains on 32.1% of points, ranks 6th on F1 and 17th of 18 on AUC. Within campaign B the d2=1.0 variant beats its d2=0.5 counterpart on AUC for all three acquisition scores - the reverse of the pendulum pattern.

Largest F1-vs-AUC rank disagreements:

- `entropy_tiebreak` (d2=1.0): F1 rank **9/18** → AUC rank **2/18** — F1=0.9874, AUC=0.9996, abstains on 8.6% of points
- `fm_disp_gdiv_d10` (d2=campaignA): F1 rank **10/18** → AUC rank **17/18** — F1=0.9873, AUC=0.9030, abstains on 32.1% of points
- `partx` (d2=-): F1 rank **17/18** → AUC rank **10/18** — F1=0.9598, AUC=0.9972, abstains on 1.6% of points

| arm | d2 | K | AUC | AUPRC | Brier | log score | excess | F1 | sep% |
|---|---|---|---|---|---|---|---|---|---|
| mode_sep | 1.0 | 100 | 0.9996 | 0.9985 | 0.0044 | 0.0197 | 0.0148 | 0.9937 | 0.067 |
| entropy_tiebreak | 1.0 | 100 | 0.9996 | 0.9985 | 0.0053 | 0.0233 | 0.0184 | 0.9874 | 0.086 |
| clf_adaptive | - | 1 | 0.9995 | 0.9980 | 0.0081 | 0.0358 | 0.0358 | 0.9919 | 0.017 |
| entropy | 1.0 | 100 | 0.9993 | 0.9984 | 0.0042 | 0.0195 | 0.0145 | 0.9967 | 0.179 |
| entropy_tiebreak | 0.5 | 100 | 0.9991 | 0.9978 | 0.0054 | 0.0240 | 0.0190 | 0.9922 | 0.026 |
| entropy | 0.5 | 100 | 0.9989 | 0.9968 | 0.0063 | 0.0265 | 0.0215 | 0.9948 | 0.067 |
| fm_ranked | baseline | 20 | 0.9989 | 0.9967 | 0.0048 | 0.0390 | 0.0149 | 0.9924 | 0.016 |
| fm_direct | baseline | 20 | 0.9984 | 0.9957 | 0.0047 | 0.0387 | 0.0146 | 0.9896 | 0.010 |
| mode_sep | 0.5 | 100 | 0.9974 | 0.9944 | 0.0068 | 0.0290 | 0.0240 | 0.9943 | 0.063 |
| partx | - | 1 | 0.9972 | 0.9865 | 0.0155 | 0.0564 | 0.0564 | 0.9598 | 0.016 |
| clf_nonadapt | - | 1 | 0.9955 | 0.9824 | 0.0402 | 0.1636 | 0.1636 | 0.9837 | 0.082 |
| fm_nonadapt | baseline | 20 | 0.9845 | 0.9654 | 0.0195 | 0.0855 | 0.0614 | 0.9839 | 0.076 |
| fm_disp_prop_d05 | campaignA | 100 | 0.9769 | 0.9506 | 0.0424 | 0.1510 | 0.1460 | 0.9849 | 0.122 |
| fm_disp_gdiv_d05 | campaignA | 100 | 0.9736 | 0.9415 | 0.0466 | 0.1657 | 0.1607 | 0.9844 | 0.132 |
| fm_disp_greedy_d05 | campaignA | 100 | 0.9587 | 0.9151 | 0.0560 | 0.2088 | 0.2038 | 0.9680 | 0.155 |
| fm_disp_prop_d10 | campaignA | 100 | 0.9236 | 0.8373 | 0.0912 | 0.3384 | 0.3334 | 0.9786 | 0.427 |
| fm_disp_gdiv_d10 | campaignA | 100 | 0.9030 | 0.8277 | 0.1077 | 0.4163 | 0.4113 | 0.9873 | 0.321 |
| fm_disp_greedy_d10 | campaignA | 100 | 0.7898 | 0.6103 | 0.1278 | 0.5582 | 0.5533 | 0.9094 | 0.471 |

*`excess` = log score − the floor imposed by that arm's K; lower is better and is the cross-K-comparable calibration number.*

## quad2d

**Target budget:** 12000 trajectories · **base rate:** 8.0% success · **scored:** 18/18 arms

### Verdict

Quad2D is where the headline metric and the underlying model quality come apart most violently: across 18 arms the correlation between F1 rank and AUC rank is only +0.15, i.e. effectively unrelated. The adaptive classifier ranks best of everything (AUC 0.9916) while sitting 16th of 18 on F1 (0.8653) — that gap is a mis-placed threshold, not a weak model. Part-X is the opposite case and the one arm the threshold-free view does *not* rescue: it is last on F1 (0.6359), last on AUPRC (0.7979) and 17th of 18 on AUC, so its problem is the ranking itself. This system is also the most imbalanced (8% success), so AUPRC is the metric to trust — and it shows real headroom (best 0.9334) that AUC's 0.99 conceals.

### Technical analysis

With only 8% positives, AUPRC is the binding constraint: AUC compresses everything into 0.969-0.992 while AUPRC spreads the same arms over 0.798-0.933. The abstention effect is strongest and clearest here (rho=+0.647, p=0.0037) and it lands squarely on the campaign-B arms: `entropy` d2=1.0 and `mode_sep` d2=0.5 take the top two F1 slots while abstaining on 35% of points, yet sit 9th and 11th of 18 on AUC. On calibration the FM baselines still lead (`fm_direct` excess 0.0534) with the campaign-B arms close behind (0.0663-0.0741).

Largest F1-vs-AUC rank disagreements:

- `clf_adaptive` (d2=-): F1 rank **16/18** → AUC rank **1/18** — F1=0.8653, AUC=0.9916, abstains on 10.7% of points
- `fm_disp_prop_d05` (d2=campaignA): F1 rank **4/18** → AUC rank **16/18** — F1=0.9643, AUC=0.9755, abstains on 36.3% of points
- `clf_nonadapt` (d2=-): F1 rank **14/18** → AUC rank **2/18** — F1=0.9095, AUC=0.9880, abstains on 11.5% of points

| arm | d2 | K | AUC | AUPRC | Brier | log score | excess | F1 | sep% |
|---|---|---|---|---|---|---|---|---|---|
| clf_adaptive | - | 1 | 0.9916 | 0.9138 | 0.0424 | 0.1401 | 0.1401 | 0.8653 | 0.107 |
| clf_nonadapt | - | 1 | 0.9880 | 0.8717 | 0.0372 | 0.1211 | 0.1211 | 0.9095 | 0.115 |
| fm_direct | baseline | 100 | 0.9872 | 0.9334 | 0.0157 | 0.0584 | 0.0534 | 0.9634 | 0.117 |
| fm_ranked | baseline | 100 | 0.9863 | 0.9263 | 0.0168 | 0.0619 | 0.0570 | 0.9528 | 0.092 |
| entropy_tiebreak | 1.0 | 100 | 0.9837 | 0.9146 | 0.0199 | 0.0720 | 0.0671 | 0.9417 | 0.156 |
| fm_disp_gdiv_d10 | campaignA | 100 | 0.9834 | 0.9010 | 0.0224 | 0.0779 | 0.0730 | 0.9316 | 0.171 |
| entropy | 0.5 | 100 | 0.9819 | 0.9143 | 0.0195 | 0.0717 | 0.0667 | 0.9397 | 0.114 |
| fm_disp_gdiv_d05 | campaignA | 100 | 0.9815 | 0.8944 | 0.0233 | 0.0817 | 0.0768 | 0.9759 | 0.361 |
| entropy | 1.0 | 100 | 0.9809 | 0.9130 | 0.0193 | 0.0713 | 0.0663 | 0.9838 | 0.351 |
| entropy_tiebreak | 0.5 | 100 | 0.9808 | 0.9092 | 0.0203 | 0.0743 | 0.0693 | 0.9384 | 0.121 |
| mode_sep | 0.5 | 100 | 0.9797 | 0.9074 | 0.0214 | 0.0779 | 0.0729 | 0.9833 | 0.353 |
| fm_disp_prop_d10 | campaignA | 100 | 0.9793 | 0.8903 | 0.0235 | 0.0829 | 0.0779 | 0.8969 | 0.147 |
| mode_sep | 1.0 | 100 | 0.9788 | 0.9036 | 0.0217 | 0.0791 | 0.0741 | 0.9362 | 0.132 |
| fm_disp_greedy_d05 | campaignA | 100 | 0.9782 | 0.8833 | 0.0251 | 0.0879 | 0.0829 | 0.9551 | 0.285 |
| fm_nonadapt | baseline | 100 | 0.9774 | 0.8909 | 0.0235 | 0.0846 | 0.0796 | 0.9095 | 0.125 |
| fm_disp_prop_d05 | campaignA | 100 | 0.9755 | 0.8872 | 0.0228 | 0.0838 | 0.0788 | 0.9643 | 0.363 |
| partx | - | 1 | 0.9730 | 0.7979 | 0.0336 | 0.1108 | 0.1108 | 0.6359 | 0.021 |
| fm_disp_greedy_d10 | campaignA | 100 | 0.9693 | 0.8600 | 0.0275 | 0.0988 | 0.0938 | 0.8633 | 0.141 |

*`excess` = log score − the floor imposed by that arm's K; lower is better and is the cross-K-comparable calibration number.*

## Provenance note

Eight arms — all six CartPole campaign-B arms and both Quad2D `mode_sep` arms — initially had no per-point probabilities on iLab. They were not lost: those runs executed on Amarel and only their JSON artifacts were rsynced back. The tell is in the directory name, `exp_<SLURM_JOB_ID>`: 6-digit IDs are iLab runs (full artifacts), 8-digit are Amarel runs (JSONs only), and this correlated 18/18 across campaign B. A partial transfer also leaves every file with an identical mtime and drops `datasets/`, `run_adaptive.log` and `.hydra/overrides.yaml`. The 140 missing `.npz` files (1.2 GB) were pulled from `/scratch/st1122/adaptive_roa/experiments/`, and all eight reproduce their artifacts' confusion matrices exactly — which independently confirms the recovered files belong to the same runs whose JSONs were already on iLab.

## Recommendations

1. **Retune λ/δ for the high-AUC / low-F1 arms rather than discarding them.** The clearest case is Quad2D `clf_adaptive`: 1st of 18 on AUC (0.9916) but 16th on F1 (0.8653) while abstaining on only 10.7% of points — the ranking is already the best available and the loss is entirely in where the threshold sits, which λ/δ search controls directly. Quad2D `clf_nonadapt` (2nd on AUC, 14th on F1) is the same story. Note this does **not** apply to Quad2D `partx`, which is last on both F1 and AUPRC — retuning will not save it.
2. **Report AUPRC, not AUC, as the primary threshold-free number for CartPole and Quad2D.** At 18% and 8% base rates AUC compresses the arms into a narrow band (Quad2D: 0.969–0.992) while AUPRC spreads the same arms over 0.798–0.933.
3. **Pair every F1 with its `sep_pct`.** The pooled ρ=+0.403 (p=0.0025) between abstention and F1-vs-AUC rank gain means an F1 quoted without its abstention rate is not interpretable; on Quad2D the top two arms by F1 (`entropy` d2=1.0, `mode_sep` d2=0.5) both abstain on ~35% of points and sit 9th and 11th of 18 on AUC.
4. **Rsync `full_roa_per_point.npz` whenever pulling Amarel results back.** The JSON-only transfer that produced this gap is easy to repeat and silently costs every threshold-free metric; the checkpoints for those eight runs are also still only on Amarel. Going forward `full_roa.py` emits the `threshold_free` block inline, so the numbers ride along in `artifacts_v2.json` and no reconstruction is needed.
5. **Retire `fm_disp_greedy_d10` on pendulum.** AUC 0.7240 / AUPRC 0.5906 against a 0.386 base rate is barely above chance on precision-recall, and unlike the other weak arms this is a ranking failure that no threshold change can recover.

