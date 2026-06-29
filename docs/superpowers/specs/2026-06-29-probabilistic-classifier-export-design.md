# General Probabilistic Classifier + Generalized Per-Point Export

**Date:** 2026-06-29
**Status:** Approved (design)
**Branch:** outcome-plane-fm

## Goal

Provide a general probabilistic-classifier abstraction so one method can be swapped for
another (MLP classification, FM final-state prediction, or future methods) behind a uniform
interface, and produce, for every method at each evaluation step, four per-point record files
(train, val, calibration, test) holding the query state, the method's native output
probabilities, and the ground-truth label.

Each method outputs a probability of success (or a scalar score reflecting it). Methods that
natively support it also output an uncertainty signal. The classifier produces `p_success`
only. FM produces `p_success`, `p_failure`, and `p_invalid` (its native uncertainty signal is
the Monte-Carlo invalid fraction).

## Current State (what already exists)

The v2 pipeline is already interface-based, not branch-scattered:

- Common output type `OutcomeProbabilities(p_success, p_failure, p_invalid)`
  (`adaptive_roa/adaptive_v2/types.py:11-17`).
- Common backend contract `ProbabilityBackend.estimate(states) -> OutcomeProbabilities`
  (`adaptive_roa/adaptive_v2/interfaces.py`).
- Two implementations: `ClassifierProbabilityBackend` (`sigmoid(logit)`, `p_invalid=0`) and
  `EndpointMCProbabilityBackend` (MC counting; `p_invalid` = fraction of rollouts in no
  attractor).
- One config switch `predictor: classifier | generative` (`adaptive_v2/engine.py:92`). The
  adaptive loop, threshold optimization, calibration, and acquisition are model-agnostic.

The per-point export the user wants partially exists, but only as a standalone offline script
(`scripts/export_probabilities.py`) driven by hard-coded run maps (`CLF_DIR`, `FM_SPEC`) and
external MC-cache files produced by `scripts/reevaluate.py`. It writes 3 splits (test/cal/val,
no train) as `.npz`, with method-specific branching in the writer. The training engine itself
only writes `full_roa_per_point.npz` for the test split per epoch.

## Decisions

- **Abstraction:** new thin wrapper module on top of the existing backends;
  `OutcomeProbabilities` and the `ProbabilityBackend` implementations are left untouched.
- **Export site:** generalize the standalone offline script (not integrated into the engine's
  eval step). "Each evaluation step" = each `epoch_NNN`; the script iterates epochs.
- **Format:** `.npz`, matching existing conventions.
- **FM second output:** export all three (`p_success`, `p_failure`, `p_invalid`). Classifier
  exports `p_success` only. The set of arrays written is whatever the method declares native.
- **Discovery:** point the script at a training run output directory; auto-discover epochs,
  split files, predictor type, and checkpoints. No hard-coded maps.
- **Train cost:** export the full split for every method, no cap (FM train/val use live MC
  inference from each epoch's checkpoint).

## Architecture

### 1. Wrapper module: `adaptive_roa/probabilistic_classifier/`

- **`base.py` — `ProbabilisticClassifier` (ABC)**
  - `native_probs: tuple[str, ...]` — the capability declaration that drives the export schema.
    Classifier `("p_success",)`; FM `("p_success", "p_failure", "p_invalid")`.
  - `predict(states: np.ndarray) -> OutcomeProbabilities` — delegates to the wrapped backend's
    `estimate`.
  - `classmethod load_from_run(run_dir, epoch, device) -> ProbabilisticClassifier` — load the
    correct model/checkpoint for offline export.
- **`classifier.py` — `ClassifierProbabilisticClassifier`** — wraps
  `ClassifierProbabilityBackend` / `ClassifierProbabilityEstimator`.
- **`flow_matching.py` — `FMProbabilisticClassifier`** — wraps
  `EndpointMCProbabilityBackend` / `MCCache` compute; reuse `mc_cache` files when present.
- **`registry.py`** — maps the `predictor` string to the wrapper class. Adding a new method =
  write one wrapper class and register it. This is the entire extension path for "something
  else."

### 2. Generalized export script: `scripts/export_probabilities.py` (rewrite)

- **Inputs:** `--run-dir <training output dir>` (replaces `CLF_DIR`/`FM_SPEC`), plus
  `--out-root`, `--epochs` (optional subset), `--device`.
- **Auto-discovery:** read `.hydra/config.yaml` -> `predictor` -> select wrapper via registry;
  glob `epoch_NNN/` directories; resolve the four split files (train/val from `datasets/`,
  cal/test from `cfg.data_source.cal_set_file` / `test_set_file`).
- **Per epoch x per split:** load the states + GT labels with the appropriate existing loader,
  call `predict`, write `{split}.npz`. Reuse `mc_cache/epoch_NNN_{test,cal}.npz` when present
  (fast path); compute live otherwise (train/val for FM).

### 3. Output files

Per `epoch_NNN/` directory under the out-root: `train.npz`, `val.npz`, `cal.npz`, `test.npz`,
each with arrays:

- `query_state [N, state_dim]` (float32)
- `gt_label [N]` (int64; 1=success, -1=failure, 0=invalid)
- the **native** prob arrays only: `p_success` (classifier) or `p_success`, `p_failure`,
  `p_invalid` (FM)

Plus one `metadata.json` per run: predictor type, probability definitions, GT-label
convention, and per-epoch/per-split row counts.

## Data Flow

```
run_dir (.hydra config -> predictor)         registry -> ProbabilisticClassifier subclass
   |                                              |
   |-- epoch_NNN/ (checkpoints, full_roa npz)     |-- load_from_run(run_dir, epoch)
   |-- datasets/ (train/val split files)          |
   |-- mc_cache/ (test/cal, FM fast path)         v
   |-- cfg.data_source.{cal,test}_set_file   predict(states) -> OutcomeProbabilities
   v                                              v
 for each split: load (query_state, gt_label) -> write {split}.npz with native_probs
```

## Error Handling & Edge Cases

- Missing checkpoint or split file for an (epoch, split) -> log and skip that pair; do not abort
  the run.
- Split-file schema differs by predictor (classification rows vs start/end/label eval-states
  rows) -> handled by selecting the loader the wrapper already uses; no schema guessing in the
  writer.
- `native_probs` is the single source of truth for which arrays are written; the writer has no
  per-method branches.
- Radius/threshold mismatch on a reused MC cache -> reclassify from cached endpoints (existing
  `MCCache` behavior).

## Testing

- **Unit:** registry resolves classifier and FM; each wrapper's `native_probs` is correct;
  `predict` returns valid shapes and `[0,1]` ranges on a tiny fixture.
- **Integration:** run the script over a small synthetic run dir (1-2 epochs) and assert four
  `.npz` files per epoch, each with the arrays expected for that predictor type, and a
  `metadata.json` with matching row counts.

## Dependencies to Verify During Planning

- The dataset builder writes `train_*_dataset.txt` (and `val_*_dataset.txt`) with GT labels
  into `datasets/`.
- Train/val classification-row files parse through the existing loader on the classifier path
  (classification rows = state + label, distinct from the start/end/label eval-states schema).

## Out of Scope (YAGNI)

- Integrating the export into the engine's per-epoch eval step (chose standalone script).
- Changing `OutcomeProbabilities` or any model-agnostic plumbing.
- Row caps / subsampling for large splits.
- CSV output.
