# All slices x all three seeds, full resolution (2026-11-12, supersedes the ordering in FLAGSHIP.md)

Dhruv picks the final 3 himself from a picker page. TERA-FM (certified c6c thresholds only, from the
same run whose checkpoint loads) runs on EVERY slice in `data/<setting>/`, on all three seeds, at
full resolution.

Order:
1. Your FLAGSHIP.md finalists, all three seeds, full resolution.
2. Every other slice, all three seeds, full resolution, in descending gt_clarity. Single-class slices
   (q2d k100_x_z, q3d-lqr viz_psi_vs_r) go last.

Stride 2 is dropped for the work that remains. Don't screen at stride 2 any more: go straight to full
resolution. Stride-2 files already written stay as they are, but the picker ignores them.

Naming, which the manifest builder relies on: full-resolution outputs are
`pred/<setting>/<plane>__<rK>.npz` and `figs/<setting>/<plane>__<rK>.{png,pdf}`, with rK in r1, r2, r3.
The pred npz must contain n_success, n_failure, n_invalid, K, decision (ny, nx), stride (= 1), seed,
run, t_s, t_f, u_s, u_f, seconds. Write each npz atomically (np.savez to `<name>.tmp.npz`, then
os.replace), and write it LAST, after its png and pdf, so a reader never sees an npz without a figure.
If a full-resolution median-seed file exists under the old name (no `__rK`), symlink it to the new
name rather than recomputing.

Cards (at most 2 processes per card, `nice -n 5`):
- cp-lqr: cuda:0, plus cuda:2 once the cp-ppo agent's work there is done (check nvidia-smi for its
  processes).
- q3d-lqr: cuda:1, plus cuda:4 once the q3d-ppo agent's work there is done.
- q2d-ppo: cuda:3.
- cp-ppo: cuda:2. q3d-ppo: cuda:4. Both are cheap (121x81), so finish and free the card.

Keep `index_<setting>_finalists.csv` current, and add an `index_<setting>_all.csv` with one row per
(plane, seed). Report when your finalists are done (if you haven't already) and again when
everything is done, with total wall time.
