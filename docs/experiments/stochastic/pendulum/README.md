# Stochastic pendulum — gaussian_signal adaptive acquisition campaign

Written 2026-08-19. **Scope: experiments run on the `stochastic/` dataset tree only.**

- **Data root** — `${data_dir}/stochastic/pendulum/gaussian_signal/lqr/{low,med,high}`
- **Experiment dir** — `/common/users/shared/pracsys/adaptive_roa_experiments/gaussian_torque/pen_{low_,,high_}<arm>`
  (note: med has **no** level prefix — `pen_dir00_s42`, not `pen_med_dir00_s42`)
- **Status** — COMPLETE. 10 arms × 20 epochs × 3 levels, all scored to full depth.

**Not in scope:** the `fm_high_i100_*` campaign reads `noisy/pendulum/lqr/high` (`noise_regime:
noisy`), a different dataset tree. It lives in
`docs/experiments/ensemble_epistemic/` with its figures under `i100_high/`. The two families have
different noise processes, pools, and ground-truth grids, and **must never be pooled or compared**.

**Figure in this directory:** `gaussian_all_levels.png` — rows = low/med/high, columns = KL /
debiased Brier / sAUROC. Solid = FM, dotted = classifier, dash-dot = Part-X GP. Shaded band is the
3-seed non-adaptive range; the 2·SD floor is printed per panel.


**Methods:** see [`../METHODS.md`](../METHODS.md) for what each arm, score and predictor actually does, the d1/d2 split, the metric definitions, and the standard of evidence. This file reports results and assumes those definitions.

---

## 1. Directory naming traps

`gaussian_torque/` holds **three different dataset families**:

| prefix | dataset |
|---|---|
| `pen_low_*`, `pen_*`, `pen_high_*` | `stochastic/pendulum/gaussian_signal/lqr/*` ← this doc |
| `cp_{low,med,high}_v2_*` | `stochastic/cartpole/gaussian_signal/lqr/*` |
| `cp_med_*` (10 runs, no `_v2`) | same, but **INVALID** — all ran `d2_ratio=0` |
| `cp_*` (15 older runs) | `stochastic/cartpole/noisy_torque/lqr/med` |

The directory name matches only the last. A glob like `cp_*` silently mixes three cartpole
families, and `cp_med_*` matches both the invalid v1 runs and the valid v2 ones. See
`../cartpole/`.

**Config repoint.** `configs/adaptive_v2/system/pendulum_stoch.yaml` changed from
`pendulum/lqr/${noise_level}` to `pendulum/gaussian_signal/lqr/${noise_level}` in commits
**9183d16 (2026-08-16 23:45)** and **615ec1c (2026-08-16 23:50)**. Launch times: med epoch_000 at
**23:58 the same night — only 8 minutes after the repoint**; high 2026-08-17 22:25; low 22:30. All
three are post-repoint and read gaussian_signal, verified from the runs' own logs, which print the
resolved path. Any earlier med-noise result is on the old dataset and is not comparable.

---

## 2. Arms

10 per level, seed 42 unless noted:

| arm | acquisition | predictor |
|---|---|---|
| `dir00_s42/s43/s44` | direct (uniform), d2_ratio=0 | fm_ensemble |
| `epi_var` | decomp_epi_var | fm_ensemble |
| `epi_bald` | decomp_epi_bald | fm_ensemble |
| `yield_a1` | decomp_epi_var_yield (alpha=1) | fm_ensemble |
| `yield_mlp` | decomp_yield_mlp | fm_ensemble |
| `partx` | partx | gp_reg |
| `clf_dir00` | direct | clf_ensemble |
| `clf_yield` | decomp_yield_mlp | clf_ensemble |

No anchored (d2_ratio=0.5) arm exists in this campaign — that variant was only run on the noisy
i100 pendulum campaign and, from 2026-08-19, on cartpole v2.

---

## 3. Results

All three levels scored to full depth: **200 rows each = 10 arms × 20 epochs, verified.**
Scoring output: `GAUSS_low/`, `GAUSS_med/`, `GAUSS_high/`.

Floors — 2·SD pooled over shared epochs, **epoch 0 excluded**:

| level | KL | debiased Brier | sAUROC |
|---|---:|---:|---:|
| low | 0.0081 | 0.0021 | 0.0004 |
| med | 0.0382 | 0.0145 | 0.0013 |
| high | 0.0394 | 0.0154 | 0.0018 |

**Epoch-19 verdicts vs the 3-seed control mean, judged against those floors:**

- **low** — KL winners: `yield_mlp` (.0057), `epi_var` (.0060), `partx` (.0066), `epi_bald` (.0079)
  vs control .0211. `clf_yield` **loses** (.0494).
- **med** — KL winners: `partx` (.0155), `yield_a1` (.0225), `yield_mlp` (.0269) vs control .0668.
  `epi_var` **loses** (.1132). `clf_yield` loses badly (.5574).
- **high** — **no winners.** Every KL and Brier gap sits inside the floor. sAUROC shows nominal
  gains of +.0026 to +.0054 against a floor of .0018 — small enough to treat as marginal.
  **High noise is a null and is reported as one.**

**Other observations**

- **Part-X (GP) is the strongest arm overall**, below the FM band for all of med and high and
  converged by ~epoch 5 while the FM arms are still descending. It is a **different model class**
  on the same axes — this is GP-vs-FM, not an acquisition comparison.
- **`epi_var` degrades sharply at med, epoch 14** — KL 2.19, sAUROC 0.6198, RES 0.0177 — then
  recovers completely at ep15 (KL .2126, sAUROC .9379) and keeps improving to ep19. **Screened
  2026-08-19: NOT a collapse.** `scripts/screen_collapsed_epochs.py` flags chance-level sAUROC
  (≤0.60) *together with* RES ≤0.01; ep14 misses both thresholds, so it is a severe single-epoch
  degradation rather than a training collapse, and it is not excluded. It also does not contaminate
  the ep19 verdict below, which is measured five epochs later — `epi_var` genuinely loses on KL at
  med (.1132 vs control .0668).
- **Classifier arms diverge upward at low** — `clf_yield` KL rises from ~ep10 to .0494 at ep19,
  worse than its own starting point, while every FM arm keeps improving.
- Arms converge at different rates, so a vertical slice mid-run flatters early converging arms
  (Part-X saturates by ~ep6) and penalises late ones. **Read the right-hand end of the figure.**

---

## 4. Methodology and traps

**Scoring.** `scripts/stoch_prob_metrics.py --spec <json> --data-root
${data_dir}/stochastic/pendulum/gaussian_signal/lqr --epochs all --out <dir>`.
Use **sAUROC, KL, and recal = UNC_debiased − RES**. Never the artifact AUC/Brier — it scores
against a 0.5-dichotomised truth and 28.6% of the grid lies in [0.05, 0.95].

**Verify row count == arms × epochs.** A silent grid rejection once produced 18 rows instead of 75.
Cached metrics also go stale: this figure was nearly built from arms scored at 14/18/19 epochs
while disk held 20 for all of them. **Re-score against disk before plotting.**

**Two predictor families share each panel and are not interchangeable.** FM arms are judged against
the 3-seed FM band; classifier arms against their own single uniform run; Part-X is a GP. Line
style encodes the family so an arm is never read against the wrong baseline. There is **no
classifier floor** — those arms have one seed each.

**Magnitude screen, not non-finite screen.** Flag any epoch whose max of
`success_mae`/`failure_mae`/`overall_mae` exceeds 1.0 **or** is non-finite. A non-finite-only screen
once cleared a cartpole arm carrying 9.4e+09.

**A stale artifact mtime is not a hang.** Arms routinely go 10–30 min between artifacts inside MC
estimation or a 5,200-batch ROA eval. Confirm from the log tail
(`tail -c 1500 | tr '\r' '\n'`) before diagnosing.

**A timed-out `find` returns 0**, which reads as catastrophic data loss. Gate on the exit code and
report "count unreliable" rather than the number.

**Job-level state hides task failure.** `sacct -X` can report COMPLETED / ExitCode 0:0 while the
python step exited 1. Check the `.0` step's exit code.

**Infrastructure.** `/common/users` is krb5-authenticated. When the ticket expires, access flips to
`EACCES` while `/common/home` keeps working, and every tool touching the scratchpad dies —
including Bash, which cannot create its own task dir. Fix: `kinit st1122@CS.RUTGERS.EDU`.
Tickets last ~1 day.

---

## 5. Provenance

- All 600 epochs (3 levels × 10 arms × 20) have `filter_diagnostics` null.
- **Collapsed-epoch screen: 0 hits across all 600 rows** (`scripts/screen_collapsed_epochs.py`,
  run 2026-08-19 over `GAUSS_low/med/high`). No epoch in this campaign sits at chance, so no floor
  is inflated and no arm carries a manufactured catastrophe.
- Datasets verified present and complete for all three levels: `train.npz`,
  `eval_success_prob.npz`, `eval_states.txt`, `cal_set.txt`, `test_set.txt`, `train_test_splits`.
- Companion: `../cartpole/` for the cartpole gaussian_signal campaign (**complete** 2026-08-19,
  14 arms × 12 epochs × 3 levels). Its result is the opposite of this one: every FM epistemic arm
  wins at every noise level there, while high noise is a null here.
