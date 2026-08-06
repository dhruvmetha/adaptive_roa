# Stochastic pendulum: adaptive vs non-adaptive, FM vs classifier

Ground truth is the 158x315 eval grid, M=90 rollouts per cell. Model probabilities are the raw p(success) (never renormalised against p_invalid, which would destroy calibration). Every squared-error quantity is debiased on both sides: the model's K-sample noise and the grid's M-sample noise are subtracted, so a perfect model scores 0.

`Brier_deb` lower is better; `SS` (skill vs the climatology of the true field) and `sAUROC` higher is better. `REL_deb` is miscalibration, `RES` resolution, `UNC_deb` the irreducible spread of the true field. `AURC` is the area under the debiased risk-coverage curve (selective risk where the model is most confident).

## Summary

Each line asks one question: does the best adaptive arm beat the non-adaptive arm by more than this system's own run-to-run noise?

- **Flow matching**

    - low ep9: best adaptive = entropy d2=1.0, gap vs non-adaptive -0.00094 → within the noise floor (2×SD=0.00164, n=5) — not distinguishable
    - med ep10: best adaptive = entropy d2=1.0, gap vs non-adaptive -0.00153 → within the noise floor (2×SD=0.00305, n=5) — not distinguishable
    - high ep9: best adaptive = entropy d2=1.0, gap vs non-adaptive -0.00174 → within the noise floor (2×SD=0.02043, n=5) — not distinguishable
    - xhigh ep9: best adaptive = entropy+modesep d2=0.5, gap vs non-adaptive -0.00082 → within the noise floor (2×SD=0.03155, n=5) — not distinguishable

- **Classifier**

    - low ep18: best adaptive = entropy d2=0.5, gap vs non-adaptive -0.00362 → **exceeds** the noise floor (2×SD=0.00314, n=3) — adaptive better
    - med ep18: best adaptive = entropy d2=1.0, gap vs non-adaptive -0.00286 → floor from seed replicates (see seed_variance.md)
    - high ep18: best adaptive = entropy d2=0.5, gap vs non-adaptive +0.02500 → floor from seed replicates (see seed_variance.md)
    - xhigh ep18: best adaptive = entropy d2=0.5, gap vs non-adaptive +0.04248 → floor from seed replicates (see seed_variance.md)

Rankings alone are not evidence here: the arms sit close together, so a gap must clear both the paired test (eval-grid noise) and the run-to-run floor above / `seed_variance.md` before it means anything.

## Depth reached (epoch index = acquisition budget)

| predictor | level | non-adaptive (d2=0) | entropy d2=0.5 | entropy d2=1.0 | entropy+modesep d2=0.5 | entropy+modesep d2=1.0 |
|---|---|---|---|---|---|---|
| fm | low | 9 | 16 | 13 | 10 | 13 |
| fm | med | 13 | 15 | 13 | 10 | 13 |
| fm | high | 13 | 9 | 9 | 10 | 13 |
| fm | xhigh | 14 | 9 | 10 | 10 | 16 |
| clf | low | 18 | 18 | 18 | - | - |
| clf | med | 18 | 18 | 18 | - | - |
| clf | high | 18 | 18 | 18 | - | - |
| clf | xhigh | 18 | 18 | 18 | - | - |

## Run-to-run noise floor (measured at epoch 0)

Epoch 0 is pre-acquisition, so every arm of a row below is the *same* configuration on the *same* data. Any spread between them is pure run-to-run variability. Flow matching is not bit-reproducible (training kernels and the K-sample MC evaluation both vary), so a bit-identical control cannot be built — instead, an arm gap later on is only meaningful if it exceeds this floor. The classifier is deterministic here, so its floor comes from the seed replicates instead.

Treat these SDs as rough: they come from only 4-5 runs, and the flow-matching estimate varies by an order of magnitude across levels, which is itself a sign the sample is small. The seed replicates are the better-powered floor.

| predictor | level | runs | SD at epoch 0 | max spread | 2×SD threshold |
|---|---|---|---|---|---|
| fm | low | 5 | 0.00082 | 0.00175 | 0.00164 |
| fm | med | 5 | 0.00152 | 0.00282 | 0.00305 |
| fm | high | 5 | 0.01022 | 0.01888 | 0.02043 |
| fm | xhigh | 5 | 0.01578 | 0.02921 | 0.03155 |
| clf | low | 3 | 0.00157 | 0.00272 | 0.00314 |
| clf | med | 3 | 0.00000 | 0.00000 | deterministic — use seed replicates |
| clf | high | 3 | 0.00000 | 0.00000 | deterministic — use seed replicates |
| clf | xhigh | 3 | 0.00000 | 0.00000 | deterministic — use seed replicates |

## Flow matching vs classifier (non-adaptive arm, equal budget)

| level | epoch | FM Brier_deb | CLF Brier_deb | FM SS | CLF SS | FM sAUROC | CLF sAUROC | winner |
|---|---|---|---|---|---|---|---|---|
| low | 9 | +0.00124 | +0.01249 | 0.9946 | 0.9453 | 0.9991 | 0.9992 | **FM** (10.1× lower) |
| med | 13 | +0.00173 | +0.01309 | 0.9921 | 0.9405 | 0.9970 | 0.9972 | **FM** (7.6× lower) |
| high | 13 | +0.00093 | +0.05746 | 0.9952 | 0.7004 | 0.9808 | 0.9810 | **FM** (61.9× lower) |
| xhigh | 14 | +0.00193 | +0.06948 | 0.9856 | 0.4814 | 0.8964 | 0.8981 | **FM** (36.1× lower) |

## Dose-response: does more acquisition make it worse (or better)?

Grouped by the fraction of each epoch's budget chosen by the acquisition rule. A monotone ordering in d2 that persists over many epochs is much stronger evidence than any single-epoch gap, because seed noise has no reason to sort itself by d2. An ordering is only asserted when every step carries real weight (each ≥15% of the span) and the whole span clears this level's run-to-run noise floor.

| predictor | level | d2 | arms | mean Brier (last 3 common epochs) | slope/epoch |
|---|---|---|---|---|---|
| fm | low | 0.0 | dir00 | +0.00095 | -0.00037 |
| fm | low | 0.5 | ent05,tb05 | +0.00036 | -0.00023 |
| fm | low | 1.0 | ent10,tb10 | +0.00036 | -0.00007 |
| | | | _ordering not asserted_ | smallest step is 1% of the span; span 0.00059 < noise floor 0.00164 | epochs 7–9 |
| fm | med | 0.0 | dir00 | +0.00173 | -0.00021 |
| fm | med | 0.5 | ent05,tb05 | +0.00027 | -0.00026 |
| fm | med | 1.0 | ent10,tb10 | +0.00026 | -0.00016 |
| | | | _ordering not asserted_ | smallest step is 0% of the span; span 0.00147 < noise floor 0.00305 | epochs 8–10 |
| fm | high | 0.0 | dir00 | +0.00158 | -0.00005 |
| fm | high | 0.5 | ent05,tb05 | +0.00049 | -0.00007 |
| fm | high | 1.0 | ent10,tb10 | +0.00047 | -0.00036 |
| | | | _ordering not asserted_ | smallest step is 3% of the span; span 0.00111 < noise floor 0.02043 | epochs 7–9 |
| fm | xhigh | 0.0 | dir00 | +0.00241 | -0.00193 |
| fm | xhigh | 0.5 | ent05,tb05 | +0.00201 | -0.00023 |
| fm | xhigh | 1.0 | ent10,tb10 | +0.00257 | -0.00011 |
| | | | _ordering not asserted_ | span 0.00016 < noise floor 0.03155 | epochs 7–9 |
| clf | low | 0.0 | dir00 | +0.01006 | -0.00028 |
| clf | low | 0.5 | ent05 | +0.00559 | -0.00021 |
| clf | low | 1.0 | ent10 | +0.00702 | -0.00055 |
| | | | _ordering not asserted_ | span 0.00304 < noise floor 0.00314 | epochs 16–18 |
| clf | med | 0.0 | dir00 | +0.01532 | -0.00013 |
| clf | med | 0.5 | ent05 | +0.01633 | -0.00009 |
| clf | med | 1.0 | ent10 | +0.01027 | -0.00026 |
| clf | high | 0.0 | dir00 | +0.05335 | -0.00038 |
| clf | high | 0.5 | ent05 | +0.08266 | -0.00021 |
| clf | high | 1.0 | ent10 | +0.19934 | +0.00709 |
| | | | **monotone ↑ in d2** | more acquisition = worse | epochs 16–18 |
| clf | xhigh | 0.0 | dir00 | +0.06844 | -0.00037 |
| clf | xhigh | 0.5 | ent05 | +0.10442 | +0.00068 |
| clf | xhigh | 1.0 | ent10 | +0.14879 | +0.00092 |
| | | | **monotone ↑ in d2** | more acquisition = worse | epochs 16–18 |

## Flow matching

### low — matched epoch 9 (n=39770)

| arm | Brier_deb | SS | REL_deb | RES | UNC_deb | sAUROC | SHARP | SHARP* | AURC | KL | F1@.5 | p_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | +0.00124 | 0.9946 | 0.00018 | 0.22736 | 0.22842 | 0.9991 | 0.0140 | 0.0094 | +0.00004 | 0.0084 | 0.9929 | 0.165 |
| entropy d2=0.5 | +0.00031 | 0.9986 | -0.00001 | 0.22809 | 0.22842 | 0.9993 | 0.0123 | 0.0094 | +0.00001 | 0.0047 | 0.9957 | 0.169 |
| entropy d2=1.0 | +0.00030 | 0.9987 | -0.00006 | 0.22806 | 0.22842 | 0.9993 | 0.0110 | 0.0094 | +0.00001 | 0.0039 | 0.9953 | 0.167 |
| entropy+modesep d2=0.5 | +0.00041 | 0.9982 | 0.00004 | 0.22805 | 0.22842 | 0.9992 | 0.0126 | 0.0094 | +0.00001 | 0.0053 | 0.9955 | 0.171 |
| entropy+modesep d2=1.0 | +0.00045 | 0.9980 | -0.00004 | 0.22792 | 0.22842 | 0.9992 | 0.0118 | 0.0094 | +0.00001 | 0.0048 | 0.9951 | 0.183 |

| arm | F1@.25 | F1@.5 | F1@.75 | acc@.5 | RoA_true | RoA_pred | risk@20% | risk@50% | risk@100% | logS | logS_orac | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | 0.9935 | 0.9929 | 0.9888 | 0.9945 | 0.3897 | 0.3892 | +0.00000 | +0.00000 | +0.00124 | 0.0379 | 0.0304 | 0.0099 |
| entropy d2=0.5 | 0.9960 | 0.9957 | 0.9945 | 0.9967 | 0.3897 | 0.3890 | +0.00000 | +0.00000 | +0.00031 | 0.0342 | 0.0304 | 0.0063 |
| entropy d2=1.0 | 0.9948 | 0.9953 | 0.9956 | 0.9964 | 0.3897 | 0.3887 | +0.00000 | +0.00000 | +0.00030 | 0.0334 | 0.0304 | 0.0055 |
| entropy+modesep d2=0.5 | 0.9955 | 0.9955 | 0.9938 | 0.9965 | 0.3897 | 0.3882 | +0.00000 | +0.00000 | +0.00041 | 0.0348 | 0.0304 | 0.0069 |
| entropy+modesep d2=1.0 | 0.9935 | 0.9951 | 0.9947 | 0.9962 | 0.3897 | 0.3906 | +0.00000 | +0.00000 | +0.00045 | 0.0343 | 0.0304 | 0.0066 |

Paired vs non-adaptive at epoch 9 (negative = adaptive better; |z| > 2 is significant):

| arm | Δ debiased Brier | SE | z | Δ vs own epoch 0 |
|---|---|---|---|---|
| _non-adaptive (reference)_ | – | – | – | -0.00864 |
| entropy d2=0.5 | -0.000923 ** | 0.000044 | -21.03 | -0.01087 |
| entropy d2=1.0 | -0.000939 ** | 0.000044 | -21.37 | -0.01089 |
| entropy+modesep d2=0.5 | -0.000833 ** | 0.000046 | -17.92 | -0.00903 |
| entropy+modesep d2=1.0 | -0.000784 ** | 0.000042 | -18.58 | -0.00939 |

Pairwise deepest comparison (each row at its own deepest shared epoch):

> The `z` column counts **only** eval-grid noise, so it is badly overconfident: it treats one training run as the whole story. This level's run-to-run floor is 2×SD = 0.00164, and a Δ smaller than that is indistinguishable from rerunning the same arm with a different seed, no matter how large |z| looks. The last column applies that test.

| arm | epoch | Δ debiased Brier | SE | z | beats run-to-run floor? |
|---|---|---|---|---|---|
| entropy d2=0.5 | 9 | -0.000923 ** | 0.000044 | -21.03 | no (|Δ| < 0.00164) |
| entropy d2=1.0 | 9 | -0.000939 ** | 0.000044 | -21.37 | no (|Δ| < 0.00164) |
| entropy+modesep d2=0.5 | 9 | -0.000833 ** | 0.000046 | -17.92 | no (|Δ| < 0.00164) |
| entropy+modesep d2=1.0 | 9 | -0.000784 ** | 0.000042 | -18.58 | no (|Δ| < 0.00164) |

### med — matched epoch 10 (n=39770)

| arm | Brier_deb | SS | REL_deb | RES | UNC_deb | sAUROC | SHARP | SHARP* | AURC | KL | F1@.5 | p_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | +0.00176 | 0.9920 | 0.00007 | 0.21826 | 0.21995 | 0.9968 | 0.0226 | 0.0181 | +0.00012 | 0.0122 | 0.9888 | 0.393 |
| entropy d2=0.5 | +0.00029 | 0.9987 | -0.00014 | 0.21951 | 0.21995 | 0.9975 | 0.0207 | 0.0181 | +0.00002 | 0.0054 | 0.9932 | 0.410 |
| entropy d2=1.0 | +0.00023 | 0.9990 | -0.00015 | 0.21956 | 0.21995 | 0.9975 | 0.0197 | 0.0181 | +0.00002 | 0.0049 | 0.9932 | 0.411 |
| entropy+modesep d2=0.5 | +0.00028 | 0.9987 | -0.00015 | 0.21952 | 0.21995 | 0.9975 | 0.0211 | 0.0181 | +0.00002 | 0.0052 | 0.9930 | 0.406 |
| entropy+modesep d2=1.0 | +0.00031 | 0.9986 | -0.00012 | 0.21952 | 0.21995 | 0.9975 | 0.0211 | 0.0181 | +0.00002 | 0.0056 | 0.9933 | 0.417 |

| arm | F1@.25 | F1@.5 | F1@.75 | acc@.5 | RoA_true | RoA_pred | risk@20% | risk@50% | risk@100% | logS | logS_orac | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | 0.9898 | 0.9888 | 0.9855 | 0.9912 | 0.3905 | 0.3877 | +0.00002 | +0.00002 | +0.00176 | 0.0692 | 0.0579 | 0.0146 |
| entropy d2=0.5 | 0.9927 | 0.9932 | 0.9920 | 0.9947 | 0.3905 | 0.3905 | +0.00000 | +0.00000 | +0.00029 | 0.0624 | 0.0579 | 0.0089 |
| entropy d2=1.0 | 0.9922 | 0.9932 | 0.9928 | 0.9947 | 0.3905 | 0.3900 | +0.00000 | +0.00000 | +0.00023 | 0.0619 | 0.0579 | 0.0084 |
| entropy+modesep d2=0.5 | 0.9926 | 0.9930 | 0.9915 | 0.9945 | 0.3905 | 0.3911 | +0.00000 | +0.00000 | +0.00028 | 0.0622 | 0.0579 | 0.0090 |
| entropy+modesep d2=1.0 | 0.9926 | 0.9933 | 0.9919 | 0.9948 | 0.3905 | 0.3895 | +0.00000 | +0.00000 | +0.00031 | 0.0626 | 0.0579 | 0.0090 |

Paired vs non-adaptive at epoch 10 (negative = adaptive better; |z| > 2 is significant):

| arm | Δ debiased Brier | SE | z | Δ vs own epoch 0 |
|---|---|---|---|---|
| _non-adaptive (reference)_ | – | – | – | -0.01247 |
| entropy d2=0.5 | -0.001467 ** | 0.000072 | -20.29 | -0.01114 |
| entropy d2=1.0 | -0.001529 ** | 0.000072 | -21.23 | -0.01120 |
| entropy+modesep d2=0.5 | -0.001476 ** | 0.000072 | -20.38 | -0.01123 |
| entropy+modesep d2=1.0 | -0.001453 ** | 0.000073 | -19.85 | -0.01395 |

Pairwise deepest comparison (each row at its own deepest shared epoch):

> The `z` column counts **only** eval-grid noise, so it is badly overconfident: it treats one training run as the whole story. This level's run-to-run floor is 2×SD = 0.00305, and a Δ smaller than that is indistinguishable from rerunning the same arm with a different seed, no matter how large |z| looks. The last column applies that test.

| arm | epoch | Δ debiased Brier | SE | z | beats run-to-run floor? |
|---|---|---|---|---|---|
| entropy d2=0.5 | 13 | -0.001392 ** | 0.000070 | -19.91 | no (|Δ| < 0.00305) |
| entropy d2=1.0 | 13 | -0.001371 ** | 0.000071 | -19.36 | no (|Δ| < 0.00305) |
| entropy+modesep d2=0.5 | 10 | -0.001476 ** | 0.000072 | -20.38 | no (|Δ| < 0.00305) |
| entropy+modesep d2=1.0 | 13 | -0.001356 ** | 0.000072 | -18.75 | no (|Δ| < 0.00305) |

### high — matched epoch 9 (n=39770)

| arm | Brier_deb | SS | REL_deb | RES | UNC_deb | sAUROC | SHARP | SHARP* | AURC | KL | F1@.5 | p_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | +0.00197 | 0.9897 | 0.00003 | 0.18984 | 0.19179 | 0.9804 | 0.0521 | 0.0490 | +0.00020 | 0.0144 | 0.9699 | 0.564 |
| entropy d2=0.5 | +0.00046 | 0.9976 | -0.00038 | 0.19095 | 0.19179 | 0.9809 | 0.0511 | 0.0490 | +0.00009 | 0.0107 | 0.9839 | 0.561 |
| entropy d2=1.0 | +0.00023 | 0.9988 | -0.00045 | 0.19111 | 0.19179 | 0.9810 | 0.0492 | 0.0490 | +0.00006 | 0.0100 | 0.9852 | 0.563 |
| entropy+modesep d2=0.5 | +0.00046 | 0.9976 | -0.00041 | 0.19092 | 0.19179 | 0.9809 | 0.0518 | 0.0490 | +0.00009 | 0.0111 | 0.9839 | 0.560 |
| entropy+modesep d2=1.0 | +0.00029 | 0.9985 | -0.00039 | 0.19111 | 0.19179 | 0.9809 | 0.0491 | 0.0490 | +0.00008 | 0.0107 | 0.9847 | 0.565 |

| arm | F1@.25 | F1@.5 | F1@.75 | acc@.5 | RoA_true | RoA_pred | risk@20% | risk@50% | risk@100% | logS | logS_orac | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | 0.9745 | 0.9699 | 0.9718 | 0.9766 | 0.3973 | 0.3818 | +0.00003 | +0.00003 | +0.00197 | 0.1732 | 0.1593 | 0.0268 |
| entropy d2=0.5 | 0.9808 | 0.9839 | 0.9818 | 0.9873 | 0.3973 | 0.3912 | +0.00003 | +0.00003 | +0.00046 | 0.1695 | 0.1593 | 0.0210 |
| entropy d2=1.0 | 0.9845 | 0.9852 | 0.9818 | 0.9883 | 0.3973 | 0.3944 | +0.00004 | +0.00004 | +0.00023 | 0.1688 | 0.1593 | 0.0190 |
| entropy+modesep d2=0.5 | 0.9819 | 0.9839 | 0.9812 | 0.9873 | 0.3973 | 0.3921 | +0.00003 | +0.00002 | +0.00046 | 0.1699 | 0.1593 | 0.0214 |
| entropy+modesep d2=1.0 | 0.9836 | 0.9847 | 0.9813 | 0.9879 | 0.3973 | 0.3928 | +0.00004 | +0.00005 | +0.00029 | 0.1695 | 0.1593 | 0.0197 |

Paired vs non-adaptive at epoch 9 (negative = adaptive better; |z| > 2 is significant):

| arm | Δ debiased Brier | SE | z | Δ vs own epoch 0 |
|---|---|---|---|---|
| _non-adaptive (reference)_ | – | – | – | -0.00956 |
| entropy d2=0.5 | -0.001515 ** | 0.000053 | -28.82 | -0.01112 |
| entropy d2=1.0 | -0.001743 ** | 0.000057 | -30.60 | -0.03018 |
| entropy+modesep d2=0.5 | -0.001513 ** | 0.000053 | -28.28 | -0.02995 |
| entropy+modesep d2=1.0 | -0.001679 ** | 0.000055 | -30.76 | -0.01189 |

Pairwise deepest comparison (each row at its own deepest shared epoch):

> The `z` column counts **only** eval-grid noise, so it is badly overconfident: it treats one training run as the whole story. This level's run-to-run floor is 2×SD = 0.02043, and a Δ smaller than that is indistinguishable from rerunning the same arm with a different seed, no matter how large |z| looks. The last column applies that test.

| arm | epoch | Δ debiased Brier | SE | z | beats run-to-run floor? |
|---|---|---|---|---|---|
| entropy d2=0.5 | 9 | -0.001515 ** | 0.000053 | -28.82 | no (|Δ| < 0.02043) |
| entropy d2=1.0 | 9 | -0.001743 ** | 0.000057 | -30.60 | no (|Δ| < 0.02043) |
| entropy+modesep d2=0.5 | 10 | -0.000999 ** | 0.000040 | -24.74 | no (|Δ| < 0.02043) |
| entropy+modesep d2=1.0 | 13 | -0.000579 ** | 0.000029 | -19.93 | no (|Δ| < 0.02043) |

### xhigh — matched epoch 9 (n=39770)

| arm | Brier_deb | SS | REL_deb | RES | UNC_deb | sAUROC | SHARP | SHARP* | AURC | KL | F1@.5 | p_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | +0.00240 | 0.9821 | 0.00028 | 0.13186 | 0.13398 | 0.8959 | 0.1136 | 0.1156 | +0.00151 | 0.0223 | 0.9608 | 0.530 |
| entropy d2=0.5 | +0.00217 | 0.9838 | 0.00047 | 0.13228 | 0.13398 | 0.8959 | 0.1136 | 0.1156 | +0.00173 | 0.0226 | 0.9674 | 0.535 |
| entropy d2=1.0 | +0.00388 | 0.9711 | 0.00200 | 0.13211 | 0.13398 | 0.8953 | 0.1009 | 0.1156 | +0.00484 | 0.0466 | 0.9706 | 0.549 |
| entropy+modesep d2=0.5 | +0.00158 | 0.9882 | -0.00008 | 0.13232 | 0.13398 | 0.8963 | 0.1122 | 0.1156 | +0.00133 | 0.0195 | 0.9716 | 0.524 |
| entropy+modesep d2=1.0 | +0.00173 | 0.9871 | 0.00005 | 0.13229 | 0.13398 | 0.8959 | 0.1124 | 0.1156 | +0.00146 | 0.0212 | 0.9731 | 0.529 |

| arm | F1@.25 | F1@.5 | F1@.75 | acc@.5 | RoA_true | RoA_pred | risk@20% | risk@50% | risk@100% | logS | logS_orac | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | 0.9554 | 0.9608 | 0.9587 | 0.9677 | 0.4217 | 0.4011 | +0.00143 | +0.00173 | +0.00240 | 0.3850 | 0.3629 | 0.0506 |
| entropy d2=0.5 | 0.9564 | 0.9674 | 0.9596 | 0.9731 | 0.4217 | 0.4051 | +0.00200 | +0.00200 | +0.00217 | 0.3853 | 0.3629 | 0.0512 |
| entropy d2=1.0 | 0.9546 | 0.9706 | 0.9606 | 0.9755 | 0.4217 | 0.4122 | +0.00564 | +0.00500 | +0.00388 | 0.4093 | 0.3629 | 0.0600 |
| entropy+modesep d2=0.5 | 0.9511 | 0.9716 | 0.9659 | 0.9763 | 0.4217 | 0.4123 | +0.00122 | +0.00164 | +0.00158 | 0.3822 | 0.3629 | 0.0465 |
| entropy+modesep d2=1.0 | 0.9610 | 0.9731 | 0.9628 | 0.9775 | 0.4217 | 0.4135 | +0.00138 | +0.00173 | +0.00173 | 0.3839 | 0.3629 | 0.0472 |

Paired vs non-adaptive at epoch 9 (negative = adaptive better; |z| > 2 is significant):

| arm | Δ debiased Brier | SE | z | Δ vs own epoch 0 |
|---|---|---|---|---|
| _non-adaptive (reference)_ | – | – | – | -0.01779 |
| entropy d2=0.5 | -0.000236 ** | 0.000044 | -5.31 | -0.01740 |
| entropy d2=1.0 | +0.001474 ** | 0.000056 | +26.46 | -0.04490 |
| entropy+modesep d2=0.5 | -0.000820 ** | 0.000044 | -18.73 | -0.04719 |
| entropy+modesep d2=1.0 | -0.000667 ** | 0.000051 | -13.12 | -0.01843 |

Pairwise deepest comparison (each row at its own deepest shared epoch):

> The `z` column counts **only** eval-grid noise, so it is badly overconfident: it treats one training run as the whole story. This level's run-to-run floor is 2×SD = 0.03155, and a Δ smaller than that is indistinguishable from rerunning the same arm with a different seed, no matter how large |z| looks. The last column applies that test.

| arm | epoch | Δ debiased Brier | SE | z | beats run-to-run floor? |
|---|---|---|---|---|---|
| entropy d2=0.5 | 9 | -0.000236 ** | 0.000044 | -5.31 | no (|Δ| < 0.03155) |
| entropy d2=1.0 | 10 | +0.005310 ** | 0.000093 | +57.34 | no (|Δ| < 0.03155) |
| entropy+modesep d2=0.5 | 10 | -0.000886 ** | 0.000051 | -17.24 | no (|Δ| < 0.03155) |
| entropy+modesep d2=1.0 | 14 | -0.000587 ** | 0.000040 | -14.81 | no (|Δ| < 0.03155) |

## Classifier

### low — matched epoch 18 (n=39770)

| arm | Brier_deb | SS | REL_deb | RES | UNC_deb | sAUROC | SHARP | SHARP* | AURC | KL | F1@.5 | p_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | +0.00868 | 0.9620 | 0.00480 | 0.22508 | 0.22842 | 0.9992 | 0.0091 | 0.0094 | +0.00025 | 0.0301 | 0.9710 | 0.000 |
| entropy d2=0.5 | +0.00506 | 0.9779 | 0.00274 | 0.22523 | 0.22842 | 0.9993 | 0.0096 | 0.0094 | +0.00013 | 0.0173 | 0.9783 | 0.000 |
| entropy d2=1.0 | +0.00726 | 0.9682 | 0.00432 | 0.22505 | 0.22842 | 0.9993 | 0.0105 | 0.0094 | +0.00021 | 0.0248 | 0.9738 | 0.000 |

| arm | F1@.25 | F1@.5 | F1@.75 | acc@.5 | RoA_true | RoA_pred | risk@20% | risk@50% | risk@100% | logS | logS_orac | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | 0.9707 | 0.9710 | 0.9710 | 0.9767 | 0.3897 | 0.4128 | +0.00001 | +0.00001 | +0.00868 | 0.0596 | 0.0304 | 0.0224 |
| entropy d2=0.5 | 0.9775 | 0.9783 | 0.9785 | 0.9828 | 0.3897 | 0.4068 | +0.00000 | +0.00000 | +0.00506 | 0.0468 | 0.0304 | 0.0169 |
| entropy d2=1.0 | 0.9726 | 0.9738 | 0.9746 | 0.9790 | 0.3897 | 0.4106 | +0.00000 | +0.00000 | +0.00726 | 0.0543 | 0.0304 | 0.0212 |

Paired vs non-adaptive at epoch 18 (negative = adaptive better; |z| > 2 is significant):

| arm | Δ debiased Brier | SE | z | Δ vs own epoch 0 |
|---|---|---|---|---|
| _non-adaptive (reference)_ | – | – | – | -0.00376 |
| entropy d2=0.5 | -0.003619 ** | 0.000132 | -27.42 | -0.01011 |
| entropy d2=1.0 | -0.001419 ** | 0.000101 | -14.07 | -0.00791 |

Pairwise deepest comparison (each row at its own deepest shared epoch):

> The `z` column counts **only** eval-grid noise, so it is badly overconfident: it treats one training run as the whole story. This level's run-to-run floor is 2×SD = 0.00314, and a Δ smaller than that is indistinguishable from rerunning the same arm with a different seed, no matter how large |z| looks. The last column applies that test.

| arm | epoch | Δ debiased Brier | SE | z | beats run-to-run floor? |
|---|---|---|---|---|---|
| entropy d2=0.5 | 18 | -0.003619 ** | 0.000132 | -27.42 | **yes** |
| entropy d2=1.0 | 18 | -0.001419 ** | 0.000101 | -14.07 | no (|Δ| < 0.00314) |

### med — matched epoch 18 (n=39770)

| arm | Brier_deb | SS | REL_deb | RES | UNC_deb | sAUROC | SHARP | SHARP* | AURC | KL | F1@.5 | p_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | +0.01394 | 0.9366 | 0.01182 | 0.21727 | 0.21995 | 0.9975 | 0.0171 | 0.0181 | +0.00069 | 0.0451 | 0.9487 | 0.000 |
| entropy d2=0.5 | +0.01558 | 0.9292 | 0.01378 | 0.21777 | 0.21995 | 0.9977 | 0.0172 | 0.0181 | +0.00079 | 0.0501 | 0.9463 | 0.000 |
| entropy d2=1.0 | +0.01108 | 0.9496 | 0.00914 | 0.21724 | 0.21995 | 0.9976 | 0.0174 | 0.0181 | +0.00052 | 0.0352 | 0.9555 | 0.000 |

| arm | F1@.25 | F1@.5 | F1@.75 | acc@.5 | RoA_true | RoA_pred | risk@20% | risk@50% | risk@100% | logS | logS_orac | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | 0.9569 | 0.9487 | 0.9483 | 0.9578 | 0.3905 | 0.4327 | +0.00000 | +0.00000 | +0.01394 | 0.1021 | 0.0579 | 0.0381 |
| entropy d2=0.5 | 0.9486 | 0.9463 | 0.9449 | 0.9557 | 0.3905 | 0.4348 | +0.00000 | +0.00000 | +0.01558 | 0.1071 | 0.0579 | 0.0413 |
| entropy d2=1.0 | 0.9600 | 0.9555 | 0.9539 | 0.9636 | 0.3905 | 0.4268 | -0.00000 | +0.00000 | +0.01108 | 0.0922 | 0.0579 | 0.0336 |

Paired vs non-adaptive at epoch 18 (negative = adaptive better; |z| > 2 is significant):

| arm | Δ debiased Brier | SE | z | Δ vs own epoch 0 |
|---|---|---|---|---|
| _non-adaptive (reference)_ | – | – | – | -0.00302 |
| entropy d2=0.5 | +0.001642 ** | 0.000137 | +11.98 | -0.00138 |
| entropy d2=1.0 | -0.002859 ** | 0.000112 | -25.55 | -0.00588 |

Pairwise deepest comparison (each row at its own deepest shared epoch):

| arm | epoch | Δ debiased Brier | SE | z | beats run-to-run floor? |
|---|---|---|---|---|---|
| entropy d2=0.5 | 18 | +0.001642 ** | 0.000137 | +11.98 | no floor yet |
| entropy d2=1.0 | 18 | -0.002859 ** | 0.000112 | -25.55 | no floor yet |

### high — matched epoch 18 (n=39770)

| arm | Brier_deb | SS | REL_deb | RES | UNC_deb | sAUROC | SHARP | SHARP* | AURC | KL | F1@.5 | p_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | +0.05387 | 0.7191 | 0.05146 | 0.18932 | 0.19179 | 0.9811 | 0.0611 | 0.0490 | +0.01138 | 0.1767 | 0.8555 | 0.000 |
| entropy d2=0.5 | +0.07887 | 0.5887 | 0.07702 | 0.18991 | 0.19179 | 0.9816 | 0.0691 | 0.0490 | +0.02183 | 0.2710 | 0.8270 | 0.000 |
| entropy d2=1.0 | +0.20101 | -0.0481 | 0.19215 | 0.18356 | 0.19179 | 0.9760 | 0.0493 | 0.0490 | +0.06686 | 0.8122 | 0.7070 | 0.000 |

| arm | F1@.25 | F1@.5 | F1@.75 | acc@.5 | RoA_true | RoA_pred | risk@20% | risk@50% | risk@100% | logS | logS_orac | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | 0.8706 | 0.8555 | 0.8401 | 0.8658 | 0.3973 | 0.5315 | +0.00000 | +0.00210 | +0.05387 | 0.3356 | 0.1593 | 0.1381 |
| entropy d2=0.5 | 0.8076 | 0.8270 | 0.8172 | 0.8338 | 0.3973 | 0.5635 | +0.00000 | +0.00697 | +0.07887 | 0.4298 | 0.1593 | 0.1792 |
| entropy d2=1.0 | 0.7490 | 0.7070 | 0.6973 | 0.6707 | 0.3973 | 0.7266 | +0.00056 | +0.03535 | +0.20101 | 0.9710 | 0.1593 | 0.2951 |

Paired vs non-adaptive at epoch 18 (negative = adaptive better; |z| > 2 is significant):

| arm | Δ debiased Brier | SE | z | Δ vs own epoch 0 |
|---|---|---|---|---|
| _non-adaptive (reference)_ | – | – | – | +0.01395 |
| entropy d2=0.5 | +0.025001 ** | 0.000308 | +81.24 | +0.03896 |
| entropy d2=1.0 | +0.147139 ** | 0.001152 | +127.70 | +0.16109 |

Pairwise deepest comparison (each row at its own deepest shared epoch):

| arm | epoch | Δ debiased Brier | SE | z | beats run-to-run floor? |
|---|---|---|---|---|---|
| entropy d2=0.5 | 18 | +0.025001 ** | 0.000308 | +81.24 | no floor yet |
| entropy d2=1.0 | 18 | +0.147139 ** | 0.001152 | +127.70 | no floor yet |

### xhigh — matched epoch 18 (n=39770)

| arm | Brier_deb | SS | REL_deb | RES | UNC_deb | sAUROC | SHARP | SHARP* | AURC | KL | F1@.5 | p_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | +0.06570 | 0.5097 | 0.06440 | 0.13249 | 0.13398 | 0.8980 | 0.1426 | 0.1156 | +0.02759 | 0.1678 | 0.7877 | 0.000 |
| entropy d2=0.5 | +0.10818 | 0.1926 | 0.10636 | 0.13209 | 0.13398 | 0.8976 | 0.1351 | 0.1156 | +0.04822 | 0.2759 | 0.6609 | 0.000 |
| entropy d2=1.0 | +0.14917 | -0.1133 | 0.13945 | 0.12422 | 0.13398 | 0.8867 | 0.1244 | 0.1156 | +0.07341 | 0.4130 | 0.6239 | 0.000 |

| arm | F1@.25 | F1@.5 | F1@.75 | acc@.5 | RoA_true | RoA_pred | risk@20% | risk@50% | risk@100% | logS | logS_orac | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-adaptive (d2=0) | 0.7253 | 0.7877 | 0.8333 | 0.7727 | 0.4217 | 0.6490 | +0.00014 | +0.02604 | +0.06570 | 0.5305 | 0.3629 | 0.2095 |
| entropy d2=0.5 | 0.7087 | 0.6609 | 0.7823 | 0.5673 | 0.4217 | 0.8544 | +0.00030 | +0.05368 | +0.10818 | 0.6386 | 0.3629 | 0.2710 |
| entropy d2=1.0 | 0.7028 | 0.6239 | 0.7547 | 0.4916 | 0.4217 | 0.9301 | +0.01262 | +0.07104 | +0.14917 | 0.7757 | 0.3629 | 0.3187 |

Paired vs non-adaptive at epoch 18 (negative = adaptive better; |z| > 2 is significant):

| arm | Δ debiased Brier | SE | z | Δ vs own epoch 0 |
|---|---|---|---|---|
| _non-adaptive (reference)_ | – | – | – | +0.00337 |
| entropy d2=0.5 | +0.042481 ** | 0.000217 | +195.64 | +0.04585 |
| entropy d2=1.0 | +0.083469 ** | 0.000474 | +176.00 | +0.08684 |

Pairwise deepest comparison (each row at its own deepest shared epoch):

| arm | epoch | Δ debiased Brier | SE | z | beats run-to-run floor? |
|---|---|---|---|---|---|
| entropy d2=0.5 | 18 | +0.042481 ** | 0.000217 | +195.64 | no floor yet |
| entropy d2=1.0 | 18 | +0.083469 ** | 0.000474 | +176.00 | no floor yet |
