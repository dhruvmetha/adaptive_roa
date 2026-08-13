# What each acquisition strategy actually selected

`mean |p-0.5|` near 0 means the acquired states are genuinely ambiguous (the model cannot know the outcome); near 0.5 means their outcome is essentially determined. `frac_ambiguous` is the share with 0.2 < p < 0.8, `frac_decided` the share with p < 0.05 or p > 0.95.

D1 is the random half of the budget, D2 the entropy-scored half. The non-adaptive arm draws its whole budget from the uniform pool, so its D1 row is the reference for 'no selection at all'.

## Flow matching

`mean p` is the average TRUE success probability of the acquired states. The random split shows what the pool actually looks like, so a scored split that drifts away from it is training on a sample that no longer represents the space the model is scored on.

| level | arm | split | n | mean \|p-0.5\| | frac ambiguous | frac decided | mean p |
|---|---|---|---|---|---|---|---|
| low | non-adaptive (d2=0) | D1(random) | 10000 | 0.4868 | 0.0280 | 0.9467 | 0.3911 |
| low | entropy d2=0.5 | D1(random) | 8500 | 0.4936 | 0.0138 | 0.9741 | 0.3919 |
| low | entropy d2=0.5 | D2(scored) | 8500 | 0.4265 | 0.1539 | 0.6962 | 0.6157 |
| low | entropy d2=1.0 | D2(scored) | 14000 | 0.4674 | 0.0686 | 0.8641 | 0.3791 |
| low | entropy+modesep d2=0.5 | D1(random) | 5500 | 0.4920 | 0.0165 | 0.9653 | 0.3738 |
| low | entropy+modesep d2=0.5 | D2(scored) | 5500 | 0.4084 | 0.1925 | 0.6227 | 0.6695 |
| low | entropy+modesep d2=1.0 | D2(scored) | 14000 | 0.4651 | 0.0728 | 0.8545 | 0.3805 |
| med | non-adaptive (d2=0) | D1(random) | 19000 | 0.4748 | 0.0522 | 0.8925 | 0.3918 |
| med | entropy d2=0.5 | D1(random) | 8000 | 0.4876 | 0.0227 | 0.9421 | 0.3626 |
| med | entropy d2=0.5 | D2(scored) | 8000 | 0.3474 | 0.3382 | 0.3771 | 0.5932 |
| med | entropy d2=1.0 | D2(scored) | 19000 | 0.4314 | 0.1455 | 0.7124 | 0.4310 |
| med | entropy+modesep d2=0.5 | D1(random) | 5500 | 0.4840 | 0.0311 | 0.9264 | 0.3821 |
| med | entropy+modesep d2=0.5 | D2(scored) | 5500 | 0.3213 | 0.3945 | 0.2764 | 0.5400 |
| med | entropy+modesep d2=1.0 | D2(scored) | 14000 | 0.4128 | 0.1859 | 0.6352 | 0.5211 |
| high | non-adaptive (d2=0) | D1(random) | 19000 | 0.4333 | 0.1318 | 0.7168 | 0.4047 |
| high | entropy d2=0.5 | D1(random) | 5000 | 0.4463 | 0.0946 | 0.7570 | 0.3982 |
| high | entropy d2=0.5 | D2(scored) | 5000 | 0.1959 | 0.7714 | 0.0286 | 0.5144 |
| high | entropy d2=1.0 | D2(scored) | 19000 | 0.3125 | 0.4129 | 0.2444 | 0.4971 |
| high | entropy+modesep d2=0.5 | D1(random) | 5500 | 0.4463 | 0.0964 | 0.7560 | 0.3961 |
| high | entropy+modesep d2=0.5 | D2(scored) | 5500 | 0.2025 | 0.7487 | 0.0389 | 0.5130 |
| high | entropy+modesep d2=1.0 | D2(scored) | 14000 | 0.2890 | 0.4741 | 0.1678 | 0.4945 |
| xhigh | non-adaptive (d2=0) | D1(random) | 15000 | 0.3457 | 0.2682 | 0.2315 | 0.4827 |
| xhigh | entropy d2=0.5 | D1(random) | 5000 | 0.3564 | 0.2346 | 0.2456 | 0.4796 |
| xhigh | entropy d2=0.5 | D2(scored) | 5000 | 0.1823 | 0.8074 | 0.0164 | 0.4831 |
| xhigh | entropy d2=1.0 | D2(scored) | 11000 | 0.2216 | 0.6858 | 0.0588 | 0.5116 |
| xhigh | entropy+modesep d2=0.5 | D1(random) | 5500 | 0.3538 | 0.2447 | 0.2405 | 0.4776 |
| xhigh | entropy+modesep d2=0.5 | D2(scored) | 5500 | 0.1880 | 0.7916 | 0.0329 | 0.4922 |
| xhigh | entropy+modesep d2=1.0 | D2(scored) | 17000 | 0.2269 | 0.6534 | 0.0174 | 0.4560 |

## Classifier

`mean p` is the average TRUE success probability of the acquired states. The random split shows what the pool actually looks like, so a scored split that drifts away from it is training on a sample that no longer represents the space the model is scored on.

| level | arm | split | n | mean \|p-0.5\| | frac ambiguous | frac decided | mean p |
|---|---|---|---|---|---|---|---|
| low | non-adaptive (d2=0) | D1(random) | 19000 | 0.4868 | 0.0275 | 0.9461 | 0.3906 |
| low | entropy d2=0.5 | D1(random) | 9500 | 0.4962 | 0.0074 | 0.9832 | 0.3906 |
| low | entropy d2=0.5 | D2(scored) | 9500 | 0.4096 | 0.1889 | 0.6277 | 0.4369 |
| low | entropy d2=1.0 | D2(scored) | 19000 | 0.4534 | 0.0973 | 0.8076 | 0.3701 |
| med | non-adaptive (d2=0) | D1(random) | 19000 | 0.4748 | 0.0522 | 0.8925 | 0.3918 |
| med | entropy d2=0.5 | D1(random) | 9500 | 0.4864 | 0.0263 | 0.9391 | 0.4067 |
| med | entropy d2=0.5 | D2(scored) | 9500 | 0.3513 | 0.3206 | 0.3884 | 0.3080 |
| med | entropy d2=1.0 | D2(scored) | 19000 | 0.4105 | 0.1873 | 0.6195 | 0.4263 |
| high | non-adaptive (d2=0) | D1(random) | 19000 | 0.4333 | 0.1318 | 0.7168 | 0.4047 |
| high | entropy d2=0.5 | D1(random) | 9500 | 0.4312 | 0.1446 | 0.7263 | 0.4374 |
| high | entropy d2=0.5 | D2(scored) | 9500 | 0.4564 | 0.0065 | 0.6387 | 0.0436 |
| high | entropy d2=1.0 | D2(scored) | 19000 | 0.4722 | 0.0130 | 0.8322 | 0.0279 |
| xhigh | non-adaptive (d2=0) | D1(random) | 19000 | 0.3458 | 0.2677 | 0.2299 | 0.4806 |
| xhigh | entropy d2=0.5 | D1(random) | 9500 | 0.3439 | 0.2878 | 0.2514 | 0.5132 |
| xhigh | entropy d2=0.5 | D2(scored) | 9500 | 0.3684 | 0.0551 | 0.0043 | 0.1316 |
| xhigh | entropy d2=1.0 | D2(scored) | 19000 | 0.3655 | 0.0634 | 0.0048 | 0.1348 |
