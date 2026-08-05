# Seed replicates

A difference between arms only means something if it is larger than the spread you get by rerunning the same arm with a different seed. `dir00` replicates change training stochasticity only (its data order is fixed), so their spread is the training-noise floor.

## med — Flow matching — epoch 13

| arm | seeds | debiased Brier per seed | mean | SD |
|---|---|---|---|---|
| dir00 | 3 | 0.00173, 0.00079, 0.00094 | 0.00115 | 0.00051 |
| ent10 | 3 | 0.00036, 0.00029, 0.00027 | 0.00031 | 0.00004 |

Gap (ent10 − dir00) = **-0.00084**; pooled seed SD = 0.00036 (2 SD = 0.00072). The gap is larger than 2 seed SDs, so it is distinguishable from run-to-run noise.

Stability across the last usable epochs:

The 2xSD test uses a pooled standard deviation, which one outlier seed can inflate enough to hide a real separation. `ranges disjoint?` is the non-parametric companion: it asks whether every seed of one arm beats every seed of the other, which no single outlier can fake in the favourable direction. Trust a verdict that both columns agree on.

| epoch | gap | 2×seed SD | significant? | ranges disjoint? |
|---|---|---|---|---|
| 10 | -0.00099 | 0.00064 | **yes** | **yes** |
| 11 | -0.00072 | 0.00007 | **yes** | **yes** |
| 12 | -0.00073 | 0.00024 | **yes** | **yes** |
| 13 | -0.00084 | 0.00072 | **yes** | **yes** |

## med — Classifier — epoch 18

| arm | seeds | debiased Brier per seed | mean | SD |
|---|---|---|---|---|
| dir00 | 3 | 0.01394, 0.00945, 0.01757 | 0.01365 | 0.00407 |
| ent10 | 3 | 0.01108, 0.01046, 0.01228 | 0.01127 | 0.00093 |

Gap (ent10 − dir00) = **-0.00238**; pooled seed SD = 0.00295 (2 SD = 0.00590). The gap is smaller than 2 seed SDs, so it is **not** distinguishable from run-to-run noise.

Stability across the last usable epochs:

The 2xSD test uses a pooled standard deviation, which one outlier seed can inflate enough to hide a real separation. `ranges disjoint?` is the non-parametric companion: it asks whether every seed of one arm beats every seed of the other, which no single outlier can fake in the favourable direction. Trust a verdict that both columns agree on.

| epoch | gap | 2×seed SD | significant? | ranges disjoint? |
|---|---|---|---|---|
| 15 | -0.00614 | 0.00362 | **yes** | **yes** |
| 16 | -0.00792 | 0.00163 | **yes** | **yes** |
| 17 | -0.00454 | 0.00087 | **yes** | **yes** |
| 18 | -0.00238 | 0.00590 | no | no |

## high — Flow matching — epoch 8

| arm | seeds | debiased Brier per seed | mean | SD |
|---|---|---|---|---|
| dir00 | 3 | 0.00136, 0.00132, 0.00162 | 0.00143 | 0.00017 |
| ent10 | 3 | 0.00056, 0.00036, 0.00061 | 0.00051 | 0.00013 |

Gap (ent10 − dir00) = **-0.00092**; pooled seed SD = 0.00015 (2 SD = 0.00030). The gap is larger than 2 seed SDs, so it is distinguishable from run-to-run noise.

Stability across the last usable epochs:

The 2xSD test uses a pooled standard deviation, which one outlier seed can inflate enough to hide a real separation. `ranges disjoint?` is the non-parametric companion: it asks whether every seed of one arm beats every seed of the other, which no single outlier can fake in the favourable direction. Trust a verdict that both columns agree on.

| epoch | gap | 2×seed SD | significant? | ranges disjoint? |
|---|---|---|---|---|
| 5 | -0.00069 | 0.00015 | **yes** | **yes** |
| 6 | -0.00106 | 0.00034 | **yes** | **yes** |
| 7 | -0.00288 | 0.00457 | no | **yes** |
| 8 | -0.00092 | 0.00030 | **yes** | **yes** |

## high — Classifier — epoch 18

| arm | seeds | debiased Brier per seed | mean | SD |
|---|---|---|---|---|
| dir00 | 3 | 0.05387, 0.05395, 0.03926 | 0.04903 | 0.00846 |
| ent10 | 3 | 0.20101, 0.22861, 0.18978 | 0.20647 | 0.01998 |

Gap (ent10 − dir00) = **+0.15744**; pooled seed SD = 0.01534 (2 SD = 0.03069). The gap is larger than 2 seed SDs, so it is distinguishable from run-to-run noise.

Stability across the last usable epochs:

The 2xSD test uses a pooled standard deviation, which one outlier seed can inflate enough to hide a real separation. `ranges disjoint?` is the non-parametric companion: it asks whether every seed of one arm beats every seed of the other, which no single outlier can fake in the favourable direction. Trust a verdict that both columns agree on.

| epoch | gap | 2×seed SD | significant? | ranges disjoint? |
|---|---|---|---|---|
| 15 | +0.14909 | 0.02354 | **yes** | **yes** |
| 16 | +0.15044 | 0.01963 | **yes** | **yes** |
| 17 | +0.15318 | 0.02415 | **yes** | **yes** |
| 18 | +0.15744 | 0.03069 | **yes** | **yes** |

## xhigh — Classifier — epoch 18

| arm | seeds | debiased Brier per seed | mean | SD |
|---|---|---|---|---|
| dir00 | 4 | 0.06570, 0.07023, 0.07093, 0.07444 | 0.07033 | 0.00359 |
| ent10 | 5 | 0.14917, 0.14588, 0.15170, 0.15215, 0.17590 | 0.15496 | 0.01197 |

Gap (ent10 − dir00) = **+0.08463**; pooled seed SD = 0.00884 (2 SD = 0.01767). The gap is larger than 2 seed SDs, so it is distinguishable from run-to-run noise.

Stability across the last usable epochs:

The 2xSD test uses a pooled standard deviation, which one outlier seed can inflate enough to hide a real separation. `ranges disjoint?` is the non-parametric companion: it asks whether every seed of one arm beats every seed of the other, which no single outlier can fake in the favourable direction. Trust a verdict that both columns agree on.

| epoch | gap | 2×seed SD | significant? | ranges disjoint? |
|---|---|---|---|---|
| 15 | +0.08551 | 0.01467 | **yes** | **yes** |
| 16 | +0.08950 | 0.01720 | **yes** | **yes** |
| 17 | +0.09124 | 0.01767 | **yes** | **yes** |
| 18 | +0.08463 | 0.01767 | **yes** | **yes** |
