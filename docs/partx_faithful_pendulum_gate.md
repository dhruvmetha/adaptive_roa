# partx_faithful pendulum gate

Criteria from spec §8.3, fixed before any run.

| name | n0 | c1 | coverage | c2 | wrong | n_classified | c3 | amb_init | amb_acq | c4 | unspent | KL | sAUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low | 10 | pass | 0.5933 | pass | 0.0000 | 28936 | pass | 0.0200 | 0.0289 | pass | 0 | 0.0801 | 0.9957 |
| med | 10 | pass | 0.5785 | n/a | 0.0082 | 28212 | pass | 0.0800 | 0.1168 | pass | 0 | 0.0801 | 0.9606 |
| high | 10 | pass | 0.2974 | n/a | 0.0339 | 14505 | pass | 0.5700 | 0.7584 | pass | 0 | 0.0535 | 0.8233 |
| med_n030 | 30 | pass | 0.3463 | n/a | 0.0200 | 16891 | pass | 0.0800 | 0.1153 | pass | 0 | 0.0587 | 0.9636 |

**Gate (n0 = 10 runs): PASS**

C5 is the n0 = 10 vs n0 = 30 comparison in the table above; it has no pass/fail.
