# State-space slice figures (Fig 1 candidates)

Requested 2026-11-12 by the paper_writer session for Dhruv. Every 2D slice of the five non-pendulum
settings, as a pair: deterministic ground truth on the left, TERA-FM's certified decisions on the right.
Previews only. Nothing here writes to the paper clones, and nothing here edits the rest of dv_l4dc_plot.

## Layout

| Path | Owner | Contents |
|---|---|---|
| `lib/` | infra agent | `style.py` (colors, `render_gt`, `render_pair`), `infer.py` (TERA-FM counts + certified decisions), `metrics.py` (GT clarity, TERA match, rank score), `sheet.py` (contact sheets, index rows) |
| `data/<setting>/<plane>.npz` | per-system agent | standardized ground-truth slice (schema below) |
| `checks/<setting>/` | per-system agent | label spot checks (re-simulation) and their logs |
| `pred/<setting>/<plane>.npz` | per-system agent, stage 2 | TERA-FM counts and decisions on the slice states |
| `figs_gt/<setting>/<plane>.{png,pdf}` | per-system agent, stage 1 | ground truth only |
| `figs/<setting>/<plane>.{png,pdf}` | per-system agent, stage 2 | paired GT / TERA-FM figure |
| `sheets/<setting>_pairs.png` | per-system agent | contact sheet of the pairs |
| `index_<setting>.csv`, `index.csv` | per-system agent, then merged | one row per slice |
| `shortlist.md` | coordinator | ranked shortlist per system |

Settings: `cp-lqr`, `cp-ppo`, `q2d-ppo`, `q3d-lqr`, `q3d-ppo`.

## Standardized slice npz (`data/<setting>/<plane>.npz`)

| Key | Type | Meaning |
|---|---|---|
| `setting` | str | one of the five settings |
| `system_label`, `controller` | str | for titles, e.g. "Cartpole", "LQR" |
| `plane` | str | unique id within the setting, e.g. `viz_theta_vs_thetadot`, `k100_theta_theta_dot` |
| `dim_names` | str[2] | dataset state names of the horizontal and vertical axes |
| `dim_tex` | str[2] | matplotlib labels without units, e.g. `$\theta$` |
| `dim_units` | str[2] | e.g. `rad`, `rad/s`, `m`, `m/s` |
| `xs`, `ys` | float[nx], float[ny] | grid coordinates of dim 1 (horizontal) and dim 2 (vertical), ascending |
| `labels` | int8[ny, nx] | deterministic ground truth, 1 = success, 0 = fail; `labels[j, i]` is at (`xs[i]`, `ys[j]`) |
| `model_states` | float32[ny*nx, D] | full start states in the method run's dataset convention and column order, row r = `labels.ravel()[r]` (C order) |
| `fixed_json` | str | JSON dict of the held dims and their values, in the units the slice was specified in |
| `source` | str | the file(s) the labels came from |
| `source_note` | str | anything a reader needs (plant variant, re-fly agreement, all-one-class flag) |
| `gt_check_json` | str | JSON summary of the label spot check, or `{}` if none was run |

## Method runs (TERA-FM = FM, U_H, d2 0.75, final round epoch_010, K = 100 endpoints)

Seed rule, fixed before any slice was looked at: the seed whose D_test certified success recall
(`c6c_rec` in `_index/scored/matrix_rows.jsonl`) is the median of its three seeds, so each picture shows a
typical run rather than the best one.

| Setting | Run | Seed | c6c_rec of the three seeds (r1, r2, r3) | t_s | t_f | u_s | u_f |
|---|---|---|---|---|---|---|---|
| cp-lqr | x29_cplqr_fm_uh_a6d075_r1 | 99512 | .985, .991, .971 | 0.45 | 0.32 | 0.31 | 0.19 |
| cp-ppo | x161_cpppo_fm_uh_a6d075_r2 | 33407 | .741, .721, .717 | 0.62 | 0.17 | 0.15 | 0.58 |
| q2d-ppo | x31_q2dppo_fm_uh_a6d075_r2 | 33407 | .81705, .81690, .763 | 0.53 | 0.27 | 0.52 | 0.36 |
| q3d-lqr | x33_q3dlqr_fm_uh_a6d075_r2 | 33407 | .742, .716, .708 | 0.28 | 0.32 | 0.20 | 0.22 |
| q3d-ppo | x70_q3dppo_fm_uh_a6d075_r1 | 99512 | .600, .481, .647 | 0.78 | 0.03 | 0.07 | 0.96 |

Run directories: `real_dir` column of `/common/users/shared/pracsys/adaptive_roa_experiments/det_v3_seeds/_index/runs_index.tsv`.
Thresholds: `c6c_t_s`, `c6c_t_f`, `c6c_u_s`, `c6c_u_f` of the matching rows (A-1d at alpha = beta = delta = .05).
Decision: Success = in_S and not in_F, Fail = in_F and not in_S, Abstain otherwise, with `in_S`/`in_F` from
`olympics-classifier-detv3/adaptive_roa/adaptive_v2/eval/certificates.py` (tie rounding included).
