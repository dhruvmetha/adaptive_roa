# dv_l4dc_plot

Every figure and table body in the L4DC paper (TERA), the script that makes it, and the numbers it shows.
This folder is the only place paper figures and tables are made.

## Start here (in git since 2026-11-13)

- **Where it lives.** A top-level folder of `dhruvmetha/adaptive_roa`, branch `feat/det-v3`. Pull the branch
  to get it. dm1487's sessions work in the worktree `tripods/olympics-classifier-detv3`. The older copies at
  `/common/users/shared/pracsys/genMoPlan/dv_l4dc_plot` (and the `tripods/dv_l4dc_plot` symlink to it) and
  `tripods/dv_l4dc_plot.pre-genmo` are frozen. Never write there.
- **What git holds.** Code, docs, the small inputs and the numbers under `values/`, `tables/`, `data/`,
  `audit/`, plus the eight slice `.npz` files that Figure 1 draws from. The repo's `.gitignore` drops other
  `*.png`, `*.pdf`, `*.npz`, `*.txt` and `*.log` files, so figures, previews and slice caches are rebuilt
  by running the scripts. `slices/checks/` (one-off checks with vendored safe-control-gym copies) and
  `plots/archive/` stayed out and remain only in the frozen copies.
- **Imports.** `predictor_comparison_data.py`, `uncertainty_signals_data.py` and `slices/lib/infer.py`
  import `adaptive_roa` from this same checkout (`common.REPO`, the folder above this one). No separate
  checkout or `PYTHONPATH` entry is needed.
- **Python.** The commands below use dm1487's env `/common/users/dm1487/envs/arcmg`, which others can't
  read. Any env with numpy, scipy, matplotlib, Pillow and PyYAML runs the figure scripts. torch is
  needed only for `slices/lib/infer.py`. Figures render text with matplotlib's usetex, so `latex` with
  the `times`, `amsmath` and `bm` packages must be on PATH.
- **The paper clone (`DV_PAPER`).** Scripts read `method-names.tex` and the committed Table 2 from a local
  clone of the Overleaf repo, and `--paper` copies figures into it. `common.PAPER` comes from the
  `DV_PAPER` environment variable and defaults to dm1487's clone,
  `/common/home/dm1487/robotics_research/tripods/deterministic-verification-pw`. Anyone else sets
  `DV_PAPER` to their own clone.
- **Other inputs.** `inputs/results-discussion-assets/` holds copies, taken 2026-11-13, of the four files
  the scripts read from the gitignored `results-discussion-assets/` in the paper's original checkout
  (Codex works there; never write or run git there). Refresh them by copying again, or point
  `DV_ASSETS` at another folder. Run data in `/common/users/shared/pracsys/adaptive_roa_experiments/` and
  `/common/users/shared/pracsys/genMoPlan/data_trajectories/` is shared and readable.
- **Style.** All paper figures share one style block in `scripts/common.py`: Times bold text, each figure
  drawn at its printed width, one color scheme. `plots/style/style_card.pdf` shows the whole scheme
  (rebuilt locally; PDFs are not in git).

## Run

```bash
cd scripts
/common/users/dm1487/envs/arcmg/bin/python plot_learning_curves.py            # values/, old-style draws, paper figure
/common/users/dm1487/envs/arcmg/bin/python plot_learning_curves.py --restyle  # the paper figure only, fast
/common/users/dm1487/envs/arcmg/bin/python plot_learning_curves.py --paper    # also copies the paper figure
```

The six paper figures (1, 5, 6, S1, S2, S3) are the paper-style draws of `scripts/common.py`, and
`plots/<stem>.pdf` is byte for byte the paper's `figures/results/<stem>.pdf` (checked 2026-11-13 against paper
commit `34eb36f`). A plain run of a figure script rebuilds its `values/` and old-style draws first (those land in
`plots/legacy/`, never in the paper), then draws the paper figure, which checks its numbers against the `values/`
just written. `--restyle` skips straight to the paper figure. Figure 5 is the exception: its default is the paper
figure only, and the old 2026-11-12 design sits behind `--baseline`, because that one rewrites
`values/pendulum_uncertainty_signals_*`, which today hold the work-in-progress script's numbers.

Without `--paper` a script never touches the paper repo. With it, `common.install_paper` copies `plots/<stem>.pdf`
into `../deterministic-verification-pw/figures/results/` when the bytes differ, so `md5sum` shows whether the paper
has the current version. PDFs carry no creation date, so a
rerun on unchanged data gives the same bytes. After `--paper`, the paper still needs the
build, commit and Overleaf push described in its `AGENTS.md`.

## Layout

- `scripts/common.py` holds the paths, setting order, colors, rc settings, the `save` step, and
  `load_final`/`cell`, which read the w = 0.2 FM fits, and `spread`, the one place a spread is computed.
- `scripts/split.py` draws the recall-over-error split layout of the adaptive-fraction figures, with the
  marker and band checks.
- `scripts/banded.py` holds the separate-bands layout of the rejected adaptive_fraction v2 preview.
- `plots/<stem>.{pdf,png}` is what each script last drew; for the six paper figures it is the paper's file.
  `plots/legacy/` holds the old-style draws of those six (and `pendulum_uncertainty_signals_wip`, the parked WIP
  design). `plots/variants/` holds non-paper versions of paper figures (Figure 5 at true cell size).
  `plots/style/` holds the style card, CVD grids and print-size contact sheet. `plots/systems/` has copies of the
  system renders.
- `tables/<name>_body.tex` is what `make_tables.py` last wrote, the generated part of each paper table.
- `values/<stem>_values.json` has every plotted number (mean, min, max, seed count) and the sha256 of each input.
  Tables have `values/table_<name>_values.json`: one entry per cell with its mean, seed count, LaTeX and bold flag.
- `archive/` under `scripts/`, `plots/` and `values/` holds retired versions. Never cite them.

## Figures in the paper

**Style and labels (Sumanth, 2026-11-13, paper commit `ffbadef`).** These override the older notes below.
- Lines, bars and markers: success side soft blue `#5497d9`, failure side soft orange `#ca7d44` (`common.LINE`), in
  Figures 6, S1, S2 and S3. The map fills (`common.FILL`, Figures 1 and 5) are unchanged.
- All figure text is regular weight: `common.TEX_PREAMBLE` no longer sets `\bfseries\boldmath`.
- Labels: "5% limit" (Figure 6) and "5% error limit" (Figure S3), not "target"; the acquisition score is `U`, not
  `U_H`, on the Figure S1 axis (the data key stays `U_H`); trajectory counts print `K` for thousands (`count_label`).
- Figure 5 prints each run's thresholds as tau^s_s, tau^s_f (stored `t_s`, `t_f`) and tau^f_s, tau^f_f (stored `u_s`,
  `u_f`), with a line skip sized to the symbols, and names its rows TERA and Unif-TERA.
- The paper writes the number of samples per query as M; K means one thousand only.
- Figure 5's legend is two rows, each its own centred legend (paper commit `1daddc1`): TP, TN, FP, FN and the two
  contours, then the three abstention kinds. It was four rows in three columns; the figure is 0.30 in shorter.

**Labels (Sumanth, 2026-11-13, paper commit `2125ad9`).** The three learned-policy settings are named RL, not PPO, and
the spatial quadrotor is 12D, not 13D (the state space is R^3 x SO(3) x R^6, stored as 13 coordinates with a unit
quaternion). This holds in every figure title, table header and panel label: `common.TITLES`, `make_tables.TITLES`,
`HEADS` and `PANELS`, `plot_learning_curves.TITLE`, `make_figure1.META`. Setting ids (`cp-ppo`, `q2d-ppo`, `q3d-ppo`)
and file names keep `ppo`. Figures 1, 6, S1, S2 and S3 and all table bodies were redrawn with only these labels
changed. Older notes below that quote PPO or 13D titles describe the figures before this change.

All data figures use FM and fit the soft four-threshold rule on validation at w = 0.2, then
report test rates, except the three certified figures (learning curves, adaptive fraction, acquisition
score), which use A-1d. All paper figures are certified since 2026-11-12.
FDR/FOR average over seeds with a nonempty predicted class. Inputs live in
`../deterministic-verification/results-discussion-assets/` (read-only) unless a full path is given.

Since paper commit `082bf21` the appendices build as a separate PDF from `supplementary.tex`, which inputs
the same `sections/appendix_*.tex` files. As of paper commit `c697005` the main paper has Figures 1-3 (1 systems in `06_experiments.tex`,
2 abstention triangle in `results_uncertainty.tex`, 3 certified learning curves in `results_ablations.tex`)
and Tables 1-2; the supplement has Figure S1 (acquisition score, `appendix_acquisition_strategy.tex`),
Figure S2 (certified adaptive fraction, Appendix C.2, `appendix_experiment_details.tex`) and Table S1. File paths and `--paper` outputs did not change.

| Paper file (`figures/...`) | Label | Script | Plot here | Input |
|---|---|---|---|---|
| `results/adaptive_learning_a1d.pdf` (Figure 6, certified learning curves, `sections/results_ablations.tex`; reworked 2026-11-12) | `fig:adaptive_learning_a1d` | `plot_learning_curves.py` | `adaptive_learning_a1d` (paper style; old style in `legacy/`) | `/common/users/shared/pracsys/adaptive_roa_experiments/det_v3_seeds/_index/scored/epochs.jsonl` (`c6c_*`) |
| `results/adaptive_fraction_certified.pdf` (Figure S2, supplement Appendix C.2, since `c697005`) | `fig:adaptive_fraction` | `plot_adaptive_fraction_certified.py` | `adaptive_fraction_certified` (paper style; old style in `legacy/`) | `c6c_*` of `/common/users/shared/pracsys/adaptive_roa_experiments/det_v3_seeds/_index/scored/matrix_rows.jsonl` |
| `results/acquisition_strategy_certified.pdf` (Figure S1, supplementary PDF since paper commit `082bf21`) | `fig:acquisition_strategy` | `plot_acquisition_strategy_bars.py` | `acquisition_strategy_certified` (paper style; old style in `legacy/`) | `c6c_*` of `matrix_rows.jsonl` via `certified.py` |

Dhruv chose the certified versions of the adaptive-fraction and acquisition-score figures on 2026-11-12, and paper_writer removed
`results/adaptive_fraction.pdf` and `results/acquisition_strategy.pdf` from the paper. Their empirical scripts
remain as previews with no `--paper` output, so a rerun can't put them back (checked: `--paper` leaves -pw with
exactly the four files above, byte-identical).
| `results/predictor_comparison_uniform.pdf` (Figure S3, supplement Appendix E, `sections/appendix_predictor_details.tex`; Unif- names from `method-names.tex`, certified wording, notes wrap between items, 2026-11-12) | `fig:predictor_comparison_uniform_full` | `plot_predictor_comparison.py` + `predictor_comparison_data.py` | `predictor_comparison_uniform` (paper style; old style in `legacy/`) | 90 uniform runs: `soft_four_wsweep.jsonl` (w = 0.2), `c6c_*` of `matrix_rows.jsonl`, each run's `epoch_010/full_roa_per_point.npz` |
| `results/pendulum_uncertainty_signals.pdf` (Figure 5, `sections/results_uncertainty.tex`; paper-style full-grid version with 1 pt abstention / FP / FN cells since paper commit `34eb36f`) | `fig:pendulum_uncertainty_signals` | `plot_uncertainty_signals_paper.py` + `uncertainty_signals_data.py` (the WIP `plot_uncertainty_signals.py` now draws `legacy/pendulum_uncertainty_signals_wip`) | `pendulum_uncertainty_signals` (+ `.txt` counts); true-size cells in `variants/` | `data/pendulum_full_grid/` (TERA-FM, Pendulum LQR, seed 99512, round 10, adaptive and uniform), each run's `epochs.jsonl` thresholds and training states from the genMoPlan pendulum `train.npz` |
| `results/flagship_slices.pdf` (Figure 1, introduction, 3.30 in; the paper-style `figure1_H_2x3_noticks` layout since paper commit `34eb36f`) | `fig:intro` | `slices/composites/make_figure1.py` (`restyle()` -> `slices/lib/composite.render_flagship_tight_h`); a plain run also redraws the `composites/figure1_*` previews | `flagship_slices.{pdf,png,txt}` | cp-ppo k100_x_x_dot r1, q2d-ppo viz_z_vs_zdot r2 cropped to its D_test box (`composites/crop_to_box.py`), q3d-ppo k100_x_x_dot r3; two-sided c6c thresholds. Reference-figure look, paper naming (TP/TN/FP/FN/Abstention, systems as in the tables, state symbols from appendix_experiment_details.tex), \name from method-names.tex |
| `systems/{pendulum,cartpole,quad2d_v2,quad3d_v2}.png` (Fig 1) | `fig:systems` | none, uploaded through Overleaf 2026-10-27 | `systems/` (copies) | none |
| `abstention_triangle/abstention_triangle.pdf` (Fig 2, `results_uncertainty.tex`) | `fig:abstention_triangle` | none here: Sumanth's pgfplots source `abstention_triangle.tex` in the same paper folder, from data files `g1.dat`-`g4.dat` (paper commits `74f22e3`, `fe43d56`); no script in the paper repo makes the `.dat` files | `abstention_triangle/` (reference copy of PDF, `.tex` and `.dat`, 2026-11-12) | `.dat` files |

The temporary comparison figure `results/adaptive_fraction_split.pdf` (paper commits `73ae670`, `54d8571`)
was retired on 2026-11-12, when the split layout became the adaptive-fraction figure. paper_writer removes the file and its figure
block from the paper.

## Spread: figures show min-max, tables show means only, nothing shows SD

Dhruv's rules (2026-11-12). Figures: every figure draws the three-seed mean as the line, marker or bar,
with a band or whisker from the min to the max over seeds. Tables: every table prints the mean only, no
range and no SD. `make_tables.assert_means_only` fails on any table body (paper or preview) that prints a
range, a +/- or a stacked or scripted spread. One helper computes means and min-max, `common.spread`, and
every figure and table script calls it; no script in `scripts/` computes a standard deviation. Errors (FDR,
FOR) use only seeds with a nonempty predicted class, and bands appear only where at least two seeds
contribute. Scripts in `archive/` still draw SD; they are retired and never write to the paper.

## Metric labels

**Since 2026-11-13 (Sumanth, paper commit `84c58cb`)** the method section's notation is recall `Rec_l` and prediction
error `Err_l` for outcome l, and results report their empirical test rates with hats. `common.REC_S`, `REC_F`,
`ERR_S` and `ERR_F` hold the four symbols; the table headers (`make_tables.METRICS`), `split.LEGEND_FOUR`, the
Figure S1 legend, the Figure 6 legend and the Figure S3 axis labels all take them from there. They replace R_s, R_f,
FDR and FOR. Data keys (`rec`, `fail_rec`, `fdr`, `for`) are unchanged. The preview-only scripts (`banded.py`,
`plot_adaptive_fraction_lines.py`, `plot_uncertified_options_*.py`) still print the old symbols. The paragraph below
is the rule as first written.

Dhruv's standing rule (2026-11-12): every figure and table label uses the metric notation of the paper's
`sections/04_proposed_method.tex`, nothing invented. Today that is R_s, R_f, FDR and FOR
(`$R_{\mathrm{s}}$`, `$R_{\mathrm{f}}$`, `FDR`, `FOR`, lines 145-146 of that file), which all scripts here use.
If the method section's notation changes, every figure follows. paper_writer sends that request.

On 2026-11-12 the learning curves were checked against the paper PDFs (0 differing pixels at 150 dpi),
and every figure since has been installed from here with `--paper`.

## Notes per figure

- `adaptive_learning_a1d` (Figure 6, reworked 2026-11-12 to Dhruv's spec, 5.98 x 4.10 in, drawn at the printed 6 in width).
  Later the same day: x ticks became trajectory counts (100 ... 400; 500, 1k, 1.5k, 2k; 2k ... 12k; 10k ... 40k; the
  paper-wide rule is integers under 1000, k from 1000, `count_label()`), the x label became "Training trajectories",
  and the arm entries became TERA / Unif-TERA from `method_names()` (`\name`, `\nameUnif`). The text below gives the
  rest of the layout; read "Adaptive / Uniform" there as TERA / Unif-TERA.
  Six systems in 2 x 3, each a recall panel (0.88 in) over an error panel (0.56 in, 64% of it), 0.04 in apart.
  Titles "Pendulum LQR", "Cartpole LQR", "Spatial quadrotor LQR", "Cartpole PPO", "Planar quadrotor PPO",
  "Spatial quadrotor PPO" (local to the script; `common.TITLES` is unchanged), 8.5 pt regular. Axis labels and
  legend 8 pt, ticks 7.5 pt, nothing bold. Every x axis is in thousands of trajectories, the pendulum included, with
  tick labels only under the error panels and one shared label "Training trajectories (x10^3)". "Recall (%)" and
  "Error (%)" only on the left column; no interior y tick labels. One legend row: Success recall, Failure recall,
  FDR, FOR, Adaptive, Uniform. Blue Success pair, orange Failure pair, solid recall and dash-dot (`split.DASHDOT`)
  error lines, 1.15 pt. Adaptive: full color, 3 pt circles. Uniform: 30% white tint (45% before, too faint over the
  bands), 2.7 pt squares. Edge 0.6 pt, so hollow markers keep a visible hole. The 5% line (0.7 pt) is labelled once,
  "5% target" in Pendulum LQR. Row axes: recall -4..103 (0/50/100) and error -0.3..6 (0/5) in both rows. The floors
  moved from -3 and -0.18 because the marker check now counts the edge stroke. Bands only where 2+ seeds contribute,
  hollow markers below 3 seeds, gaps where none (38 and 46 such cells). The script fails if a marker (edge included)
  or band leaves its panel, the 5% line leaves an error axis, the x label box overlaps the tick labels, or the PDF
  falls outside 5.95-6.05 x 3.95-4.2 in. An independent validator recomputed all 528 cells from `epochs.jsonl` and
  matched them to the drawn artists (means, band ends, hollow markers, gaps); the values JSON's points and
  source hashes are unchanged, so `check_fig3` and Table 4 still pass.
  Shared first point (Sumanth, 2026-11-13, paper commit `9db7c7b`). Within a setting and seed, TERA and Unif-TERA start from
  the same initial batch, seed and pool order (checked for all 18 pairs: both train on pool-order indices 0..N-1 of the
  same `pool_order_<seed>.txt`), so their epoch-0 checkpoints are two trainings of one configuration. Both arms now show
  the epoch-0 certified rates of the better of the two, the run with the higher certified recall
  (`share_first_point`; it fails unless that run is at least as high on both recalls, and unless equal recalls come with
  equal FDR and FOR). 15 of the 18 pairs were already equal (both certify nothing). The three Spatial quadrotor LQR
  pairs take the TERA run, so four plotted cells change, all Unif-TERA at 10k: Success recall 16.69 -> 21.44, Failure
  recall 64.72 -> 88.38, FDR 0.72 -> 0.88, FOR 2.96 -> 4.65. `values/adaptive_learning_a1d_values.json` lists the 18
  picks under `shared_first_point`; its `source_hashes` keep the committed path key (same sha256). The empirical preview
  is not shared. The paper's file was drawn with matplotlib 3.10.8, so its bytes differ from a 3.10.0 draw (0 differing
  pixels at 400 dpi for the unchanged script).
- `adaptive_learning` (empirical learning curves): preview only since 2026-11-12, same style and script, with its
  own row axes (recall -4..103; error -0.5..10 LQR, -0.9..18 PPO). paper_writer removed it from the paper.
- `adaptive_fraction_lines_split_rowaxes_dotted` (preview; the empirical adaptive-fraction figure until the certified one replaced it,
  6.44 x 4.81 in). The certified adaptive-fraction figure uses the same layout. Per setting
  a recall panel over an error panel of equal height. y axes are zoomed and shared within each row (`ROW_AXES`):
  LQR recall 48-102 (ticks 50/75/100), error -0.18-6 (0/5); PPO recall 38-102 (40/70/100), error -0.36-12 (0/5/10).
  R_s/R_f solid, FDR/FOR dotted, same colors, circle markers and min-max bands. The 5% line is solid
  gray (`REF_SOLID`) so it can't pass for a dotted error line, and it is not in the legend. The 2x2 legend
  (`LEGEND_FOUR`) reads "R_s (Success recall)", "R_f (Fail recall)", "FDR (error of Success declarations)",
  "FOR (error of Fail declarations)". The script fails if a marker or band would be cut, or if the 5% line
  would fall off an error axis.
- `acquisition_strategy_certified` (Figure S1, supplementary) and the empirical `acquisition_strategy_bars` (preview, no paper output),
  both drawn by `bar_figure` in
  `plot_acquisition_strategy_bars.py` (6.44 x 4.36 in). Six settings, U_H / q_A / U_tau at rho = 0.75. Recall bars
  (R_s, R_f, 0-100) over FDR/FOR bars with their own axis, min-max whiskers, no abstention. Thin solid gray 5% line,
  not in the legend. The legend pairs each color with its two metrics in the method's terms: "R_s (Success recall) /
  FDR (error of Success declarations)" and "R_f (Fail recall) / FOR (error of Fail declarations)". A bar from fewer
  than three seeds gets an "n=k" label (none today). Empirical: soft rule at w = 0.2, error axis 0-8%. Certified:
  `c6c_*` via `certified.py`, error axis 0-6%, and the script fails unless U_H equals Table 2's FM adaptive row.
- `adaptive_fraction_certified` (Figure S2, Dhruv's pick 2026-11-12, 6.44 x 4.81 in). Its x label reads "Adaptive fraction rho (rho = 0 is Unif-TERA)" since 2026-11-12 (Dhruv): the name comes from
  `common.method_names()`, which reads the paper's `method-names.tex`; a rerun after the switch gave the same bytes. The installed PDF differs from the previous one only in that label (300 dpi pixel diff). FDR/FOR are dash-dot (`split.DASHDOT`, dash 3.2, gap 1.3,
  dot 0.9, gap 1.3 line widths, tighter than matplotlib's '-.' so the learning curves' short segments still show it) since
  2026-11-12, in both the learning-curve and adaptive-fraction figures and their legends; the empirical previews keep dotted lines.
  Markers in both are 1.8pt with a 0.5pt edge (`split.PAPER_MS`, `split.PAPER_MEW`), so hollow markers keep a
  visible hole at 300 dpi; at the old 0.75pt edge the hole nearly closed. The learning curves' two arm legend samples are
  markers without a line, because at 1.8pt a line hides the circle/square that names the arm. The empirical split layout on the same 72
  FM runs, scored with two-sided A-1d at alpha = delta = 0.05 (`c6c_*` of `matrix_rows.jsonl`, sha256-pinned), as
  `make_tables.py` scores Table 2. The script fails unless rho = 0.75 equals the paper's Table 2 "FM adaptive (ours)"
  row (24 cells). Own row axes: LQR recall 48-102 (50/75/100), error -0.18-6 (0/5); PPO recall -3-103 (0/50/100,
  Planar Quad PPO uniform certifies nothing on two seeds), error -0.18-6 (0/5). Bands only where 2+ seeds
  contribute; Planar Quad PPO FDR/FOR are hollow at rho = 0 (1 seed) and 0.5 (2 seeds).

## Previews, not in the paper

| Plot here | Script | What it is |
|---|---|---|
| `adaptive_fraction_lines_split_rowaxes_dotted` | `plot_adaptive_fraction_lines.py` | Empirical adaptive-fraction figure, in the paper until the certified version replaced it (2026-11-12). |
| `acquisition_strategy_bars` | `plot_acquisition_strategy_bars.py` | Empirical acquisition-score figure, in the paper until the certified version replaced it (2026-11-12). |
| `adaptive_fraction_lines_combined` | `plot_adaptive_fraction_lines.py` | Rejected adaptive-fraction figure (v1): all four lines overlaid on twin axes spanning the whole panel, legend with (left)/(right). It was in the paper from the v2 swap until split_rowaxes_dotted replaced it. |
| `adaptive_fraction_lines_combined_v2` | `plot_adaptive_fraction_lines.py` + `banded.py` | Rejected adaptive-fraction figure: recall and errors in separate vertical bands. Dhruv said it reads as a split plot. It was in the paper from `a52afc4` until v1 replaced it. |
| `adaptive_fraction_lines_split` | `plot_adaptive_fraction_lines.py` | Adaptive-fraction alternative: recall panel over an error panel. |
| `adaptive_fraction_lines_split_rowaxes` | `plot_adaptive_fraction_lines.py` (`ROW_AXES`) | The paper figure before its last edits: same axes and checks, but solid FDR/FOR lines, a dashed 5% line in the legend, and the two-entry "R_s / FDR" legend (6.44 x 4.71 in). |
| `archive/adaptive_fraction_bars` | `archive/plot_adaptive_fraction_bars.py` | Retired adaptive-fraction bar version (four settings). |
| `archive/acquisition_strategy_abstain` | `archive/plot_acquisition_strategy_abstain.py` | Retired acquisition-score figure: half width, four settings, abstention bars. Ported from the strategy branch of `plot_adaptive_comparisons.py`. |
| `archive/predictor_comparison_uniform_certified` (+ `_4x1`) | `archive/plot_predictor_comparison_certified.py` | Certified-only identical-data dot plot. In the paper on 2026-11-12 until Table 4 (`identical_data_certified_body.tex`) replaced it in paper commit `7a21aa8`. No `--paper` option; it writes only to `plots/archive` and `values/archive` and reproduces the archived PDF byte for byte (md5 `1ef6b5f1`). |

## Tables in the paper

**Certified tables (Sumanth, 2026-11-13, paper commit `793bde7`).** The main paper's two-sided table is now
`matrix_certified_main_body.tex`: `\nameBNNLabel`, `\nameBNN`, `\name`, bold among these three rows, wrapped in
`sections/06_results.tex`. The five-row `matrix_certified_twosided_body.tex` moved to the supplement
(`sections/appendix_predictor_ablation.tex`, `tab:certified_all`) with TERA-MLP and MLP-Label. Both are written by
`scripts/split_certified_tables.py`, which fails unless the three-row cells equal the five-row ones. The rows of the
table list below predate this.

`scripts/make_tables.py` writes three bodies; `--paper` copies the ones that differ into the paper's `tables/`.
The former Table 3 (`verification_application_body.tex`) was folded into Table 2 on 2026-11-12; the generator
no longer writes it, and its last version is in `tables/archive/`. A body is
only the tabular part: column headers, numbers, bold marks, the `\resizebox` wrappers and the `\par\medskip`
between the LQR and PPO panels. The table environment, `\centering`/`\small`/`\tabcolsep`/`\arraystretch`,
caption and label live in the paper's sections (paper commit `b2401ba`) and the generator never writes them.

| Paper file (`tables/...`) | Wrapped in | Rows | Input |
|---|---|---|---|
| `matrix_uncertified_body.tex` (Table 1) | `sections/06_experiments.tex` | 4 prior-method adaptations, 2 BNN, FM uniform, FM adaptive; no group-label rows, `\midrule` between groups | soft rule at w = 0.2: `soft_four_wsweep.jsonl` + `partx_soft_wsweep.jsonl` |
| `matrix_certified_body.tex` (Table 2) | `sections/06_experiments.tex` | `MLP-Label`, `BNN-Label`, `\name-MLP`, `\name-BNN`, `\name-FM (adaptive)` as one group, then a `\midrule`, then `\name-FM, Success only`. The Success-only row is the one-sided 6a certificate (`c6a_rec`/`c6a_fp`) of the FM method runs, n/a for R_f, FOR, Abst., never bold and outside the bold comparison (the former Table 3) | A-1d at delta = 0.05: `c6c_*` of `matrix_rows.jsonl`, pinned by sha256 in `generalized_6c_values.json` |
| `identical_data_certified_body.tex` (Table 4, `tab:identical_data`, since paper commit `7a21aa8`; it replaced the dot plot `predictor_comparison_uniform_certified`) | `sections/06_results.tex` | `BNN-Label`, `\name-BNN`, `\midrule`, `\name-FM`. The FM row is absolute means. In the two BNN rows R_s, R_f and Abst. are signed differences from FM in percentage points (unrounded means subtracted, then 2 dp; all 3 seeds, all-Abstain included). FDR and FOR stay absolute, with (n) below 3 seeds and -- for none. No bold. Written by `scripts/make_identical_data_table.py`, which fails unless all 90 cells match a recomputation from Figure S3's per-seed table (`values/predictor_comparison_uniform_per_seed.csv`) | the `cal` records of `predictor_comparison_data.build()` (`c6c_*` of `matrix_rows.jsonl`, 90 uniform runs) |
| `predictor_ablation_body.tex` (Table S1, supplementary) | `sections/appendix_predictor_ablation.tex` | 5 adaptive predictor variants | soft rule at w = 0.2, as matrix_uncertified |

`assert_means_only()` lets a single signed value such as `$-27.24$` through (a difference, not a spread) and still rejects `\pm`, `[min, max]`, `min--max` and stacked or scripted ranges; the rule checks itself against both kinds of cell on import.

Each body starts with two comment lines saying it is generated and which section wraps it. They don't name
this folder on purpose: co-authors read them on Overleaf and can't see `dv_l4dc_plot` (paper_writer, 2026-11-12).

Row order is one rule for every table, `make_tables.ordered_groups()` (Dhruv, 2026-11-12): TERA rows always
below the baselines. Groups, each separated by a `\midrule`: prior methods; the -Label rows then the other TERA
variants in family order (MLP, BNN), so each TERA variant sits below its matching -Label row; then the
`\name-FM` rows, uniform before adaptive; `\name-FM, Success only` always last. Table 2 passes
`fm_own_group=False`, so its FM row joins the variants' group and its only rule sits directly above the
Success-only row. Tables 1 and 2 use it (Table 1
came out byte-identical when it switched). Table S1 still has its old hand order (FM first) and adopts
`ordered_groups()` at Dhruv's final supplement sync.

Row labels follow the paper's `method-names.tex` (paper commit `811e036`, \input by `main.tex` and
`supplementary.tex`) since 2026-11-12. `make_tables.method_label(role, model)` derives each learned-predictor label
from the row's data, never from old label text: roles `method` and `model comparison` are the adaptive arm (U_H,
rho = 0.75) and give `\name{}`, `\nameMLP{}`, `\nameBNN{}`, `\nameMLPLabel{}`, `\nameBNNLabel{}`; role `uniform data`
gives the `\nameUnif...{}` forms. `check_roles()` re-reads `runs_index.tsv` on every run and fails if a role's
acquisition changes or a macro is missing from `method-names.tex`. Prior-method rows are plain text: DeepReach,
Lyapunov NN, MORALS, Part-X GP (no "score" / "levels", Dhruv). Figures read the same file through
`common.method_names()` (`PAPER / 'method-names.tex'`). `certified.FM_ROW` (`\name{}`) and
`predictor_comparison_data.check_table1` (`\nameUnif{}`) hold the labels the cross-checks parse in the paper's tables.
Wording rule from Sumanth's `WRITING_GUIDE.md` (paper commit `811e036`): certified / certification for the
procedure and its outputs, calibration only for the independent data.

Column headers use the method section's notation: R_s, FDR, R_f, FOR, Abst. Bold marks the best displayed
recall and error per setting and metric, ties included, never Abst. On 2026-11-12 a fresh run reproduced all
four committed bodies byte for byte, header comments included.

## Table 1 layout previews (not in the paper)

`scripts/table_previews.py` (run by `make_tables.py`) writes two layouts of Table 1, the live design options
(2026-11-12). `--paper` never copies them; paper_writer installs a chosen one by hand. Both print means only,
one decimal, booktabs rules only (no `\tablepairdivider`/`\tablesystemdivider`), text-mode digits (Times,
like the body text, not the math digits of today's tables), best value judged at the displayed 1 dp (same
24 cells as today's 2-dp table).

| File (`tables/`) | Layout | Measured at `\small`, `\textwidth` 433.6pt |
|---|---|---|
| `matrix_uncertified_body_clean.tex` | B: today's wide layout, best mean bold, 12pt gaps instead of dividers | scale 0.836 / 0.828 (today 0.771 / 0.758) |
| `matrix_uncertified_body_shaded.tex` | D: B without bold, best cell `\cellcolor{black!10}`, FM rows `\rowcolor{blue!4}`; needs `\usepackage{colortbl}` | scale 0.836 / 0.828 |

Retired to `tables/archive/` because they print ranges: A (`long_lqr`, `long_ppo`), C (`clean_ranges`), the
earlier stacked and sub/superscript previews, and `table1_previews_rendered_ABCD.pdf`, which renders A-D.

## Predictor comparison and pendulum uncertainty signals (2026-11-12)

Neither figure establishes correctness at any individual state. They describe aggregate test-set rates and regions.

**`predictor_comparison_uniform`** (6.45 x 6.21 in). All five predictors on the same uniform trajectories per setting
and seed, final round (budgets 400 / 2,000 / 2,000 / 12,000 / 30,000 / 40,000; K = 100 endpoints; BNN-Label
`n_marginal_samples` = 64). Two evaluations of each final checkpoint: thresholds fitted on D_val at w = 0.2
(`soft_four_wsweep.jsonl`) and two-sided A-1d at alpha = delta = 0.05 (`c6c_*`). Every per-seed point is recomputed
from per-state test predictions with `certificates.in_S/in_F` and must reproduce the stored counts.
Checks that stop the build: identical training trajectories across the five models of every (setting, seed) at
trajectory-ID level (18/18 groups); one D_val, one D_cal and one test set per setting across models and seeds; budgets;
TERA-FM validation-fitted points = Table 1's `\name-FM (uniform)` row (30/30 cells); TERA-FM calibrated points =
the uniform round-10 end points of `adaptive_learning_a1d` (24/24). Per-seed table with counts, abstention,
thresholds, certification status and the calibration upper bounds (separate from test rates):
`values/predictor_comparison_uniform_per_seed.csv`. 23 calibrated seed-runs are all-Abstain; they are listed under
their system, never placed on the scatter.

**Retired: `predictor_comparison_uniform_certified`** (6.0 x 3.81 in, 2x2). The certified half of the figure above:
R_s, R_f over FDR, FOR, five dots per setting, mean with min-max. Dhruv compared it with the compact table on Overleaf
and kept the table (Table 4, paper commit `7a21aa8`), which also deleted this PDF from the paper. Script, PDFs (2x2 and
the `_4x1` preview) and values are in the `archive/` folders. The table's numbers come from the same `cal` records, and
its generator checks all 90 cells against Figure S3's per-seed table instead of this figure's values.

**Figure 5 since paper commit `34eb36f` (2026-11-13)** is the paper-style, full-grid version from
`plot_uncertainty_signals_paper.py` (see "Paper-wide restyle"). The history below is kept for reference. Until then
the figure was parked (Dhruv, 2026-11-12): the paper held `f9c695a4`, which Sumanth's commit `811e036` made from the
6.52 x 4.86 in version described below by editing its text. The work in progress, `scripts/plot_uncertainty_signals.py`
(a 3-column A | B | C figure, nearest-state grid fill of test + calibration states, C = abstentions only), now draws
into `plots/legacy/pendulum_uncertainty_signals_wip.*`; it and `plot_uncertainty_signals_paper.py --baseline` both
write `values/pendulum_uncertainty_signals_*`, which today hold the WIP's numbers. The earlier 3-column decisions
version is `scripts/archive/plot_uncertainty_signals_with_decisions.py`.

**`pendulum_uncertainty_signals`**, as installed before 2026-11-12 noon (6.52 x 4.86 in). TERA-FM, Pendulum LQR, seed 99512, round 10, adaptive and uniform
rows, each with its own saved thresholds (adaptive 0.22 / 0.33 / 0.46 / 0.15, as in the abstention triangle;
uniform 0.05 / 0.13 / 0.11 / 0.01). Panels I(x), D(x) on one [0, 1] scale, and the decisions with the three
abstention kinds. Each test state is drawn where it was evaluated; white specks are lattice cells with no test state
(the test set fills 20,702 of 158 x 315 cells), never interpolated. Statistics for all three seeds at rounds 2 and
10: `values/pendulum_uncertainty_signals_stats.csv`; the manuscript comparison is in
`values/pendulum_uncertainty_signals_values.json`.

Missing or incompatible artifacts, and how each was handled:
- Pool-order files configured at `/scratch/dm1487/...` and `/home/st1122/...` are unreadable here; the genMoPlan copies
  under `data_trajectories/.../splits_det_rerun/` were used and accepted only because they reproduce the terminal-state
  runs' own `train_trajectories.txt` in every group.
- MLP-Label and BNN-Label runs save no trajectory list; their IDs come from `dataset_builder_state.json` indices through
  the verified pool order.
- Three pendulum runs point at a `/scratch` `train.npz`; genMoPlan's copy reproduces each run's own
  `train_endpoint_dataset.txt` exactly, so it was used.
- Uniform seed 24475 fails two-sided certification at round 2 and has no thresholds; that run-round is excluded.

Threshold ordering: the strict invalid / weak / outcome-uncertain kinds partition the abstentions only when
t_s < u_s and u_f < t_f, which holds in 5 of the 11 certified run-rounds at rounds 2 and 10 (and in both runs of the
figure). Elsewhere: weak is empty when t_s >= u_s; invalid and outcome-uncertain overlap at seed 33407 round 2 (216
states adaptive, 217 uniform); and some abstentions fit no kind (adaptive 24475: 5 at round 2, 14 at round 10;
uniform 24475 round 10: 10; uniform 99512 round 2: 141). No priority rule is applied.

Comparison with `sections/results_uncertainty.tex` (paper commit 5daf370), means of per-seed values over certified
seeds: 12 of 19 numbers reproduce at their stated precision, and t_s + u_f <= 0.61 holds (max 0.61, adaptive 33407
round 9). The manuscript's weak numbers (47% / 48% near the boundary, mean p_inv 0.57 / 0.82) match only when weak is
taken as every abstention that is neither invalid nor outcome-uncertain (47.2% / 47.5%, 0.565 / 0.821), not its
stated definition (49.3% / 46.3%, 0.616 / 0.868, from 2 seeds each). Its round-2 invalid shares (5.0% / 9.0%) match
only when the 216 / 217 overlap states are left out (4.95% / 8.99%; with them 5.30% / 9.52%). The 0.3% of
near-boundary dense states that are invalid abstentions under uniform sampling does not reproduce under any reading
tried: it is 2.3% (10/585, 10/550, 18/541 by seed).

## Old scripts in the paper repo

`results-discussion-assets/` in the original checkout (gitignored, so Overleaf never sees it) still holds the scripts these were
ported from: `render_learning_metrics.py`, `plot_adaptive_comparisons.py`, `plot_adaptive_a1d.py`,
`plot_a1d_learning.py`. On 2026-11-12 each one's paper write was redirected to its own local folder,
because a rerun would overwrite the paper figures. `plot_adaptive_a1d.py` was the worst case: it writes A-1d
versions under the names `adaptive_learning`, `adaptive_fraction` and `acquisition_strategy`.
No script outside this folder writes `figures/results/` now. Keep it that way.

`write_paper_tables.py` (same folder) wrote the old full tables, captions included, straight into the paper's
`tables/`. Since 2026-11-12 it writes them to `results-discussion-assets/tables_local/` instead. Its captions are
stale ("soft objective", "A-1d"), so never copy its output into the paper. No script outside this folder writes
`tables/` now.

## Pendulum full-grid predictions (2026-11-12, the data of Figure 5)

`data/pendulum_full_grid/<run>_epoch010_grid.npz` holds round-10 TERA-FM predictions for all 49,770 pendulum start
states (158 x 315 grid, no holes) for seed 99512, adaptive (x231) and uniform (m65). The values are the stored ones
where the run saved them (test 20,702, cal 1,000, val 500, `source` field) and fresh inference with the run's own
checkpoint and evaluation path elsewhere (27,568, incl. 68 timeouts labelled -1). Fresh draws match the stored test
predictions to K=100 Monte Carlo noise (provenance JSON next to each npz). Not yet validated independently and not
yet wired into `plot_uncertainty_signals.py`; when used, decisions must follow the A-1d scorer (Success on S\F,
Fail on F\S, abstain otherwise).

## Paper-wide restyle (2026-11-13, in the paper since commit 34eb36f)

One style for every figure, defined once in `scripts/common.py` (block "Paper style"): usetex Times bold with bold
Computer Modern math; ticks and legend 7 pt, labels and titles 8 pt; `FILL` for regions, `LINE` / `LS` / `LW` for
lines (colour = side, line style = metric kind), `METHOD_MARKER` for methods (shape and fill, never colour),
`ABSTAIN_KIND`, `INK`, `PARTIAL_TINT`, `SIGNAL_CMAP`, `PRINT_WIDTH`. No restyle path defines a colour of its own.

Every paper figure is drawn at its print width (scale 1.000 in the paper, measured by compiling a scratch copy).
paper_writer installed them in paper commit `34eb36f`; since then the paper-style draw is each script's default
output, `plots/<stem>.pdf` (see Run). The previews from before the install are in
`plots/archive/restyle_previews_2026-11-13/`.

| Figure | Command (from `scripts/` unless noted) |
|---|---|
| 1 `flagship_slices` | `python composites/make_figure1.py` (from `slices/`; `--restyle` skips the composites) |
| 5 `pendulum_uncertainty_signals` | `python plot_uncertainty_signals_paper.py` (full 158 x 315 grid, abstention / FP / FN cells as 1 pt squares); `--rare-pt 0` writes the true-size version to `plots/variants/` |
| 6 `adaptive_learning_a1d` | `python plot_learning_curves.py` |
| S1 `acquisition_strategy_certified` | `python plot_acquisition_strategy_bars.py` |
| S2 `adaptive_fraction_certified` | `python plot_adaptive_fraction_certified.py` |
| S3 `predictor_comparison_uniform` | `python plot_predictor_comparison.py` |
| 2, 3, S4 (TikZ) | edited and recompiled in place in the paper clone; sources live there, not here |

Deliverables, rebuilt from the files above:
- `python style_card.py`: `plots/style/style_card.{pdf,png}`, and `style_card.txt` with CIELAB dE76 for every
  colour pair that meets in a figure, under normal vision, grayscale and Machado (2009) protan / deutan / tritan.
- `python cvd_grid.py`: `plots/style/cvd/<stem>_cvd.png` for all nine figures, one method.
- `python contact_sheet.py`: `plots/style/contact_sheet.pdf` (+ page PNGs), every figure at print size with its
  paper linewidth, scale and md5.

Figure 5 cells are 0.51 x 0.21 pt at print. Anything drawn at true cell size (abstentions, FP, FN) does not show at
print, which is why the paper version draws those cells as 1 pt squares.

## Table 2 as a figure: options (2026-11-13, for choosing, nothing installed)

dm1487 via paper_writer: Table 2 (tab:uncertified, validation-fitted w = 0.2) drawn as a figure, BNN-Label and
TERA-BNN left out. `scripts/uncertified_options_data.py` loads the 180 cells from
`values/table_matrix_uncertified_values.json` and asserts each equals the paper clone's
`tables/matrix_uncertified_body.tex`. Both scripts read their drawn artists back and assert every mean and whisker
against it.

| Option | Script (from `scripts/`) | Output in `plots/options_uncertified/` |
|---|---|---|
| A scatter 2 x 3, B S3-style rows 4 x 3 | `python plot_uncertified_options_ab.py` | `optA_scatter`, `optB_rows` |
| C1 bars + printed errors, C2 bars + log error axis, D heatmap table | `python plot_uncertified_options_cd.py` | `optC1_bars_numbers`, `optC2_bars_erroraxis`, `optD_heatmap` |
| contact sheet with a note per option | `python contact_sheet_options.py` | `contact_sheet_options.pdf` |

CVD grids in `plots/options_uncertified/cvd/`. `check_drawn` on recall-vs-error views must drop MORALS R_s in both
Spatial Quad settings, whose FDR is undefined (n = 0); the A/B script asserts the dropped cells are exactly those.
