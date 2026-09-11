# partx_faithful pendulum gate

Criteria from spec §8.3, fixed before any run.

| name | n0 | c1 | coverage | c2 | wrong | n_classified | c3 | amb_init | amb_acq | c4 | unspent | KL | sAUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low | 10 | pass | 0.5933 | pass | 0.0000 | 28936 | pass | 0.0200 | 0.0295 | pass | 0 | 0.0770 | 0.9960 |
| med | 10 | pass | 0.5018 | n/a | 0.0137 | 24472 | pass | 0.0800 | 0.1195 | pass | 0 | 0.0695 | 0.9634 |
| high | 10 | pass | 0.3590 | n/a | 0.0179 | 17509 | pass | 0.5700 | 0.7363 | pass | 0 | 0.0527 | 0.8138 |
| med_n030 | 30 | pass | 0.3456 | n/a | 0.0046 | 16856 | pass | 0.0800 | 0.1053 | pass | 0 | 0.0582 | 0.9661 |

**Gate (n0 = 10 runs): PASS**

C5 is the n0 = 10 vs n0 = 30 comparison in the table above; it has no pass/fail.
