# Stage 2 instructions (all settings)

TERA-FM predictions and paired figures for every slice in `data/<setting>/`. Read `lib/README.md` first
(API, the typical per-slice code, validation). Thresholds: `infer.thresholds(infer.RUNS[setting])` only,
the certified c6c ones. Never substitute other thresholds.

    PY=/common/users/dm1487/envs/arcmg/bin/python
    PYTHONPATH=$(git rev-parse --show-toplevel):$(git rev-parse --show-toplevel)/dv_l4dc_plot/slices

1. One background process per setting (Bash run_in_background, `nice -n 5`), on your assigned card,
   `batch_size=16384`, `seed=0`, model loaded once. It loops over the slices, skips any slice whose pred
   file already exists (resumable), and logs to `pred/<setting>/run.log`.
2. After each slice, save `pred/<setting>/<id>.npz` with n_success, n_failure, n_invalid, K, decision
   (ny, nx), stride, seed, run, t_s, t_f, u_s, u_f and seconds. Then render
   `figs/<setting>/<id>.{png,pdf}` with `style.render_pair(sl, dec, stem, metrics=tera_m)`, where
   `gt_m, tera_m = metrics.slice_metrics(sl, dec)`.
3. 121x81 slices run at full resolution, `<id>` = plane.
4. 400x400 slices take two passes.
   - Pass 1 is stride 2. Build a sub-slice dict from the loaded slice (xs[::2], ys[::2],
     labels[::2, ::2], and the model_states rows of exactly those cells, with nx/ny updated) and give
     it `<id>` = `<plane>_s2`.
   - Once pass 1 has covered every slice of the setting, pass 2 runs at full resolution
     (`<id>` = plane), in descending pass-1 rank_score order.
   - Pass 2 belongs in the same background process, after pass 1, so it continues without you.
5. After pass 1, and again at the very end, the process rewrites:
   - `index_<setting>.csv` (`sheet.index_row` / `write_index`), one row per slice. Each row uses the
     full-resolution result if one exists, else stride 2; the `resolution` field must say which, e.g.
     "400x400 GT; TERA 200x200 stride 2". Include `png` and `pdf`.
   - `sheets/<setting>_pairs.png` (`sheet.contact_sheet` + `sheet.pair_caption`), sorted by
     rank_score, best first.
6. Sanity check after the first slices. Read 2-3 pair PNGs yourself. The TERA panel must line up with
   the GT boundary (no transpose, no shift). If in-box R_s or R_f falls far below the run's D_test
   c6c_rec / c6c_fail_rec (`_index/scored/matrix_rows.jsonl`), suspect the state convention: stop
   and report.
7. Write only to `pred/<setting>/`, `figs/<setting>/`, `sheets/<setting>_pairs.png` and
   `index_<setting>.csv`. Don't edit `lib/`: report a lib bug instead of patching it. Don't touch
   other settings' files.

Report (under 300 words) when pass 1 is done (or all slices for 121x81-only settings). Give per slice:
rank_score, gt_clarity, R_s, R_f, abstain, n_wrong (in-box), resolution. Then your top 3 with one
line each on why, the wall time, anything odd, and whether pass 2 is still running. When pass 2
finishes, check its outputs and report again briefly.
