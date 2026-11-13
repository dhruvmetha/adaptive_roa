# slices/lib

Shared code for the state-space slice figures (see `../README.md` for the slice schema, the five
runs and the decision rule). Four modules and two scripts:

| File | What it does |
|---|---|
| `infer.py` | TERA-FM endpoint counts on any start states (the run's own evaluator), thresholds, certified decisions |
| `metrics.py` | GT clarity, TERA match, rank score; in-box and full-slice versions |
| `style.py` | `load_slice`, `render_gt`, `render_pair` |
| `sheet.py` | `contact_sheet`, `INDEX_COLUMNS`, `index_row`, `write_index`, caption helpers |
| `render_gt_all.py` | GT figures for every `data/<setting>/*.npz` plus a GT contact sheet per setting |
| `validate_infer.py` | reruns 5,000 D_test states per run and compares with the stored epoch_010 predictions |
| `composite.py` | paper Figure 1: `render_triplet` (three systems, 6.0 in wide) |
| `build_manifest.py` | writes `slices/manifest.json` for the picker page |
| `rerender_pairs.py` | re-renders every `figs/<setting>/<plane>__rK.{png,pdf}` from its pred npz in the current style |

`_scratch/` holds throwaway files (temporary eval files, the example slice, test renders).
`_example/` holds the worked example figure.

## Environment

```bash
PY=/common/users/dm1487/envs/arcmg/bin/python
DETV3=$(git rev-parse --show-toplevel)     # the adaptive_roa checkout
SLICES=$DETV3/dv_l4dc_plot/slices
PYTHONPATH=$DETV3:$SLICES $PY your_script.py      # then: from lib import infer, style, metrics, sheet
```

The arcmg env's editable install points at `olympics-classifier`, not detv3. `infer.py` puts
detv3 first on `sys.path` and raises if `adaptive_roa` was already imported from anywhere else;
`infer.adaptive_roa_file()` returns the path that ran. Only `infer.py` needs torch and detv3; the
other modules need numpy, scipy, matplotlib and PIL.

## API

### infer.py

- `RUNS`: setting -> run name (the README table). `run_dir(run_name)`: `real_dir` from
  `_index/runs_index.tsv`.
- `thresholds(run_name) -> dict(t_s, t_f, u_s, u_f)` from the `c6c_*` columns of the one row of
  `_index/scored/matrix_rows.jsonl` whose `name` matches. Raises when the row is missing or
  duplicated, `c6c_pass` is not true, or a threshold is null.
- `run_counts(run_dir, model_states, device="cuda:1", seed=0, out_dir=None, batch_size=None,
  labels01=None, keep_files=False, verbose=False) -> dict(n_success, n_failure, n_invalid, K,
  seconds, ckpt)`. `model_states` is float32 (N, D) in the run's dataset state order (the
  `test_set.txt` columns). Counts are int32 arrays of length N, K is 100.
- `decide(counts, thr) -> int8`: 1 Success, 0 Fail, -1 Abstain. ps = tie(n_success/K),
  pf = tie(n_failure/K); `in_S`, `in_F` and `tie` are imported from detv3's
  `adaptive_v2/eval/certificates.py` and `det_v2_rule.py`.
- `counts_from_per_point(path)`: counts from a stored `full_roa_per_point.npz`.
- `load_run(run_dir, device)`: (cfg, system, flow_matcher, evaluator, ckpt), cached.

How `run_counts` gets its numbers, so nothing is reimplemented:

1. `system = hydra.utils.instantiate(cfg.system)` from the run's `resolved_config.yaml` (engine.py ~147).
2. `flow_matcher = <cfg.flow_matcher class>.load_from_checkpoint(<run>/epoch_010/checkpoints/best-*.ckpt)`,
   then `.eval().to(device)` (flow_matching_trainer.py ~219, engine.py ~433).
3. `FullROAEvaluator(cfg.eval, system, device).evaluate_epoch(...)` on a temporary eval file in
   the probabilistic `test_set.txt` layout (D state columns + p_success), batch size
   `cfg.get("val_batch_size", 2048)` as the engine passes it. K = `eval.num_mc_samples_eval`
   (100), `refine_invalids` false, attractor radius and `binary_outcomes` from `cfg.eval`.
4. Counts = round(p_* x K) from the evaluator's `full_roa_per_point.npz`; the function checks that
   the evaluator's `start_states` equal the input exactly and that the counts sum to K.

The temporary directory goes under `lib/_scratch/` (or `out_dir`, which must be under `slices/`;
paths under genMoPlan and the experiment roots are refused) and is deleted afterwards.

Seeding: `random`, `numpy` and `torch` are seeded with `seed` before the call and cuDNN is set
deterministic, as the engine does. The same seed, batch size and row order give identical counts
(checked). The run's own RNG state at its epoch_010 evaluation is not recoverable, so a rerun
matches the stored predictions in distribution, not bit for bit. A different `batch_size`
reassigns noise draws to rows (same distribution).

### metrics.py

- `gt_metrics(labels)`, `tera_metrics(labels, decision)`, `rank_score(gt_m, tera_m)`: the
  definitions in the task (`GT_KEYS`, `TERA_KEYS`). NaN where a denominator is 0.
  `tera_match` is NaN when both R_s and R_f are NaN.
- `slice_metrics(sl, decision=None) -> (gt_m, tera_m)`: the numbers to use. Primary keys are on
  the in-box cells; `<key>_full` are the whole slice. `gt_m` adds `in_box_frac`; `tera_m` adds
  `rank_score` (in-box) and `rank_score_full`.
- In-box rule: a cell is in when its centre lies inside `box_x` x `box_y` (inclusive, NaN side =
  unbounded). Without box keys the whole slice is in the box. The in-box cells form one
  sub-rectangle (`box_crop(sl)`), so the image metrics (boundary, components) run on that crop.

### style.py

- `load_slice(path) -> dict`: the schema keys as python strings / lists / arrays, plus `fixed`
  and `gt_check` (parsed JSON), `nx`, `ny`, `path`, and `box_x` / `box_y` (float[2] or None).
  Raises on a missing key, a shape mismatch, labels outside {0, 1}, or non-ascending axes.
- `render_gt(sl, out_stem)` and `render_pair(sl, decision, out_stem, metrics=None)` write
  `out_stem.png` (200 dpi) and `out_stem.pdf` (images rasterized, text and lines vector) and return
  both paths. `metrics` is the `tera_m` from `slice_metrics`; it adds one line under the legend
  (R_s, R_f, abstain, FP with FDR, FN with FOR, rank).
- Fixed layout in inches, no tight bbox: every pair figure of a setting has the same pixel size.
  Panel width 2.75 in, panel height = width x clip(ny/nx, 0.6, 1.0); a pair is 6.7 in wide.
- Colors: Success `#0072B2`, Fail `#D55E00`, Abstain `#A8A8A8`. Wrong calls are filled cells,
  Success the positive class (FDR = FP/(TP+FP), FOR = FN/(FN+TN)): FP "Success declared, was Fail"
  dark crimson `#A3001E`, FN "Fail declared, was Success" Okabe-Ito yellow `#F0E442`. FP sits in
  GT-fail territory beside orange and FN beside blue, and each differs most from its neighbour;
  grey levels (Rec. 601 luma) run FN .84 > abstain .66 > fail .47 > success .34 > FP .20, so all
  five stay apart in grayscale. (Reddish purple `#CC79A7` for FP would sit at .59, between fail
  and abstain.) `style.decision_rgb(decision, labels)` builds the image. GT boundary black contour of `labels` at 0.5 on both panels, start box dashed `#3C3C3C`
  (only the finite sides that fall inside the plot; legend "dashed: D_test start box").
- The fixed-values line wraps to the figure width and gets "(single class)" when the slice has
  one label.

### sheet.py

- `INDEX_COLUMNS`, `index_row(sl, gt_m, tera_m=None, run=None, extra=None)`, `write_index(rows, path)`.
  `run` is a run name (seed, K = 100 and thresholds looked up) or a dict of those columns;
  `extra` sets other columns such as `png` and `pdf` and raises on an unknown key.
- `contact_sheet(png_paths, captions, out_path, ncols=2, tile_width=1300)`: PIL tiling, one
  caption line over each tile. `pair_caption(row)` gives plane, rank_score, gt_clarity, R_s, R_f,
  abstain, n_wrong; `gt_caption(row)` the GT-only version.
- Metric columns without a suffix (including `succ_frac` and `rank_score`) are the in-box numbers;
  `_full` columns are the whole slice.

### render_gt_all.py

`$PY slices/lib/render_gt_all.py [--setting S]` renders every `slices/data/<setting>/*.npz` to
`slices/figs_gt/<setting>/<plane>.{png,pdf}` and writes `slices/sheets/<setting>_gt.png` (3 columns,
sorted by in-box gt_clarity, best first). No GPU and no detv3 import. `--data-root`, `--figs-root`
and `--sheets-root` redirect it (used for the tests under `_scratch/`).

### composite.py (paper Figure 1)

`render_triplet(entries, out_stem, layout="2x3", aspect=None, fixed_text=False, row_labels=True)
-> (png, pdf, txt)`.

- `entries`: three dicts `{"slice": data/<setting>/<plane>.npz, "pred": <pred npz with decision and run>,
  "label": optional column title}`. The default title is `"<system_label> (<controller>)"`. A decision
  equal to `labels[::s, ::s]` for some s (2 in practice) cuts the slice to those cells; both rows then
  show the same cells. The pred's stored thresholds must equal `infer.thresholds(run)` (raises otherwise).
- `layout="2x3"`: columns are systems, top row ground truth, bottom row TERA-FM; `"1x3"`: TERA-FM only.
  GT boundary (black) on every panel, dashed start box where the slice has one, one legend.
- Width is 6.0 in. A first pass measures each column's tick and axis labels and the panels take the
  rest, about 1.45 in wide; height = width x aspect (default clip(mean ny/nx, 0.6, 1.0)). Ticks 7 pt,
  axis labels 8 pt, titles and row labels 9 pt. PNG 300 dpi; the PDF has rasterized images, vector text and lines.
- Wrong calls: FP and FN filled cells in the TERA image, colours as in style.py, never crosses.
  Nothing is dilated, so each coloured area is the true FP or FN area; a lone cell on a 400-wide grid
  is about one 300-dpi pixel, which is why the sidecar carries the counts for the caption. The legend
  has two rows: classes and lines, then the FP and FN labels.
- Fixed values go in the sidecar (`fixed_text=True` also prints them under each column, but a spatial
  quadrotor slice holds ten or more dims fixed). `<out_stem>.txt` lists per panel: slice and pred paths,
  grid and stride, cell size, fixed values, box, run, training and inference seed, K, the c6c thresholds,
  and in-box and full-slice R_s, R_f, abstain, FP, FN, FDR, FOR, gt_clarity and rank_score.

### build_manifest.py (picker page)

`$PY slices/lib/build_manifest.py [--out slices/manifest.json]`. One `"done"` entry per
`pred/<setting>/<plane>__r{1,2,3}.npz` (full resolution only; `_s2` and `.tmp.npz` files are ignored) and
one `"pending"` GT-only entry (figs_gt PNG, TERA fields null) per slice with fewer than 3 done seeds.
Keys: setting, system, controller, plane, dims, fixed, seed (training seed from matrix_rows), rk, run,
png, pdf (absolute, null if the file is missing), gt_clarity, R_s, R_f, abstain, FDR, FOR, wrong_count,
wrong_S, wrong_F, n_FP (= wrong_S), n_FN (= wrong_F), rank_score, n_in_box, status, source_note
(+ seeds_done on pending entries). Count fields are ints on done entries and null on pending ones. Metrics
are in-box. A pred's cache key includes its data npz, so a rewritten data file refreshes its entries. source_note is the slice's note (which carries the plant description) plus bracketed caveats: the
share outside the start box, "single class", and any non-numeric fixed value (e.g. q3d's attitude note).
No NaN in the JSON. Both the manifest and the cache (`lib/_scratch/manifest_cache.json`, keyed by file
mtime and size) are written atomically. A cold run takes about 1 s, a cached one 0.1 s.

### rerender_pairs.py

`$PY slices/lib/rerender_pairs.py [--setting S]` re-reads each data npz and each
`pred/<setting>/<plane>__r{1,2,3}.npz` (symlinks followed, `_s2` and `.tmp` skipped) and rewrites
`figs/<setting>/<plane>__rK.{png,pdf}` through a temporary name and `os.replace`. It never writes a pred
file. 73 figures took 42 s on 2026-11-12.

## Typical per-slice code

```python
from lib import infer, metrics, sheet, style
sl = style.load_slice("slices/data/cp-ppo/<plane>.npz")
name = infer.RUNS[sl["setting"]]
counts = infer.run_counts(infer.run_dir(name), sl["model_states"].astype("float32"), device="cuda:1", seed=0)
dec = infer.decide(counts, infer.thresholds(name)).reshape(sl["ny"], sl["nx"])
gt_m, tera_m = metrics.slice_metrics(sl, dec)
png, pdf = style.render_pair(sl, dec, f"slices/figs/{sl['setting']}/{sl['plane']}", metrics=tera_m)
row = sheet.index_row(sl, gt_m, tera_m, run=name, extra={"png": png, "pdf": pdf})
```

Save `counts` (and `dec`) to `pred/<setting>/<plane>.npz` before rendering; inference is the
expensive step.

## Validation

`validate_infer.py`, run 2026-11-12 on arrakis cuda:1 (shared, 100% busy with other users' jobs),
batch 2048 as in the runs, seed 0. Raw results: `_scratch/validation.json`, log `_scratch/validation.log`.

Before any rerun, `decide()` on the stored epoch_010 counts reproduces each matrix row's `c6c_rec`,
`c6c_fail_rec`, `c6c_abstain` and the S/F set sizes exactly on the full D_test (all five runs).

Row alignment: for every run, `load_eval_states(cfg.data_source.test_set_file, state_dim=D)`
(eval_states_file is the same file) returns start states and labels exactly equal to
`start_states` and `true_labels` in `epoch_010/full_roa_per_point.npz`. The 5,000 rows are
`np.sort(default_rng(0).choice(n_test, 5000, replace=False))`.

"noise" is the expectation from K = 100 resampling alone: two independent multinomial(100, stored p)
draws per state, decided and compared (50 replicates). "GT agree" is the share of states whose
decision equals the GT label (an abstention never agrees).

| setting | run | mean abs dps (noise) | decision agree (noise) | GT agree stored / rerun | abstain stored / rerun | states/s, batch 2048 |
|---|---|---|---|---|---|---|
| cp-lqr | x29_cplqr_fm_uh_a6d075_r1 | 0.0014 (0.0014) | 0.9990 (0.9983) | 0.9958 / 0.9962 | 0.0026 / 0.0028 | 122 |
| cp-ppo | x161_cpppo_fm_uh_a6d075_r2 | 0.0114 (0.0109) | 0.9576 (0.9639) | 0.8608 / 0.8618 | 0.1308 / 0.1300 | 131 |
| q2d-ppo | x31_q2dppo_fm_uh_a6d075_r2 | 0.0036 (0.0033) | 0.9948 (0.9960) | 0.9830 / 0.9824 | 0.0042 / 0.0058 | 122 |
| q3d-lqr | x33_q3dlqr_fm_uh_a6d075_r2 | 0.0099 (0.0099) | 0.9742 (0.9732) | 0.9090 / 0.9070 | 0.0470 / 0.0496 | 113 |
| q3d-ppo | x70_q3dppo_fm_uh_a6d075_r1 | 0.0030 (0.0029) | 0.9456 (0.9433) | 0.6898 / 0.6952 | 0.3096 / 0.3036 | 92 |
| cp-lqr | x29_cplqr_fm_uh_a6d075_r3 (Amarel, remapped) | 0.0019 (0.0017) | 0.9976 (0.9978) | 0.9940 / 0.9944 | 0.0018 / 0.0022 | 68 |
| cp-ppo | x161_cpppo_fm_uh_a6d075_r3 (Amarel, remapped) | 0.0091 (0.0089) | 0.9602 (0.9612) | 0.7982 / 0.8002 | 0.1964 / 0.1942 | 70 |

The two r3 rows (run later, card busier, hence the lower rate; `_scratch/validation_r3.json`) are the
runs trained on Amarel. Their config and checkpoint name data under
`/scratch/dm1487/genMoPlan-exp/data_trajectories`, which this host cannot read. `infer.PATH_REMAP` rewrites
that prefix to `/common/users/shared/pracsys/genMoPlan/data_trajectories` in memory: in every string of
the loaded config (system.dataset_dir, the split and test-set files, ...) and in the checkpoint
`hyper_parameters` (system_dataset_dir), which `load_from_checkpoint` then reads from the patched
in-memory dict. Each rewrite is printed as `[infer] path remap ...`. Nothing in the run directory or
the checkpoint file changes. Their D_test rows still match the stored per-point npz exactly, and the
agreement sits at the noise level, so the shared copy of the data matches what they trained on.

cp-ppo and q2d-ppo sit slightly under the simulated noise. The simulation understates noise a little
(a state stored at p = 0 or 1 never moves in it), so `_scratch/seed_check.py` compared two reruns
(seed 0 vs 1) on the same rows: cp-ppo rerun-vs-rerun agree 0.9614, mean abs dps 0.0111, seed 1 vs
stored 0.9618 / 0.0114; q2d-ppo rerun-vs-rerun 0.9942 / 0.0035, seed 1 vs stored 0.9960 / 0.0036.
The stored predictions are as close to a rerun as two reruns are to each other.

Same seed, batch size and rows give identical counts (checked on cp-ppo).

Throughput at K = 100 (100 Euler steps per draw), cuda:1 shared and fully busy, so treat these as
noisy: 92-131 states/s at batch 2048 (table); at batch 16384 on 16,384 states, cp-ppo 177/s
(vs 114/s at 2048 in the same session) and q3d-ppo 154/s (vs 104/s); q3d-ppo at batch 32768
measured 110/s later, when the card was busier. A 400 x 400 slice (160,000 states) costs about
15-30 min per process with the card to ourselves among our jobs; five agents sharing cuda:1 split
that. GPU memory is small: peak 0.05 GB at batch 2048, 0.17 GB at 16384, 0.32 GB at 32768.
Passing `batch_size=16384` is the cheap speedup; it changes which noise draw each row gets, not
the distribution.

## Provenance

Worktree HEAD `0cdbcfa`. Launch commits from each run's `provenance.json` (all `dirty: null`):
`acae65bf33` for x29 (cp-lqr) and x70 (q3d-ppo), `a840c33993` for x161 (cp-ppo), x31 (q2d-ppo)
and x33 (q3d-lqr). `git diff <launch> 0cdbcfa -- adaptive_roa/` is empty for both, so the model,
sampling, endpoint-classification, evaluator and certificate code used here is byte-identical to
what the runs launched with. Files changed since then are scripts, manifests and tests only.

## Example

`_scratch/make_example_slice.py` builds `_scratch/example_cp-ppo_k100_theta_theta_dot.npz` from
`/common/users/dm1487/slices_k100_stage/cartpole/root/low/slices/theta_theta_dot/eval_success_prob.npz`
(`det_labels` reshaped to (na, nx) = (81, 121), theta varying fastest, checked against `starts`;
`state_order` x, theta, x_dot, theta_dot already matches the cp-ppo dataset; box = D_test min/max
along theta and theta_dot, which sits just inside the slice). `_scratch/make_example.py` runs it
through `run_counts` / `decide` with the cp-ppo run (9,801 states, 81 s) and writes
`_example/cp-ppo_k100_theta_theta_dot.{png,pdf}`, its counts `_example/cp-ppo_k100_theta_theta_dot_pred.npz`,
a one-row `_example/index_example.csv` and `_example/sheet_example.png`. In-box: R_s 0.797,
R_f 0.778, abstain 19.3%, 168 wrong calls (161 Success on GT fail), rank_score 0.225.

## Notes for the per-system agents

- `model_states` columns follow each dataset's `dataset_description.json`, which is also the
  `test_set.txt` order: cartpole x, theta, x_dot, theta_dot (cp-ppo's env order is x, x_dot,
  theta, theta_dot, permutation [0, 2, 1, 3]; the stage npz files are already in dataset order);
  q2d x, z, theta, x_dot, z_dot, theta_dot (theta wrapped to [-pi, pi]); q3d x, y, z, qw, qx, qy,
  qz, x_dot, y_dot, z_dot, p, q, r (scalar-first unit quaternion, canonicalised qw >= 0; body rates).
- `run_counts` wants float32 and checks the column count against the system's `state_dim`.
- cp-lqr's config has `attractor_radius: null` (box success test); the evaluator handles it.
- Load once per process: `load_run` caches the model per (run_dir, device). Run one process per
  setting and loop over planes; save counts to `pred/` before rendering.
- `run_counts` caps torch CPU threads at 4 (this host's CPUs are shared).
