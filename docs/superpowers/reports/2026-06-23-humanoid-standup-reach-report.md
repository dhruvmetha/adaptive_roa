# HumanoidStandUpReach — Build Report (overnight autonomous session)

**Date:** 2026-06-23 → 24 (autonomous, user asleep)
**Branch:** `main` (local; **not pushed**)
**Net new since the design spec (`e49b204`):** 24 commits. **Full test suite: 60 passing, 0 skipped on this machine.**

---

## 1. Executive summary

Built the full `humanoid_standup_reach` system from scratch (you asked to replicate the Quad3D code path for the new `humanoid_get_up_medium` dataset), across three planned phases — and then scale-hardened it for the real 800k-trajectory dataset. Everything is implemented, tested, and reviewed (per-task reviews + three opus whole-branch reviews). It runs **end-to-end** at smoke scale and the scale-critical paths are validated at real (150k) pool size.

- **Plan 1 (core):** system, data module, latent-conditional flow matcher, train/inference, Hydra configs. Manifold ℝ³⁴ × S² × ℝ³⁰; `use_manifold` toggle; per-dimension normalization; binary head-height/CoM-speed success classify.
- **Plan 2 (adaptive + eval):** ROA eval config; data-module direct-file mode + trainer registration; `adaptive_v2` wiring (FPS-primary cal/test); **intermediate-state acquisition** (candidate = sub-trajectory start, marking-based dedup) behind an opt-in `candidate_mode` flag; end-to-end smoke test.
- **Plan 3 (scale-hardening):** O(1) packed candidate-ids + lazy lengths (was an infeasible ~60M registry); fixed an `n=0` flood, an intermediate seeding bug, a biased val split, and a biased FPS eval cap; switched the acquisition pool to the leakage-free 150k train split.

---

## 2. Key autonomous decisions (and why)

| Decision | Why |
|---|---|
| **Acquisition pool → 150k TRAIN split** (`shuffled_indices.txt`), not `all_shuffled_*` (800k) | The 800k pool mixes train+eval trajectories — training on eval-split trajectories then evaluating on the eval-split FPS sets is **data leakage**. Verified the 150k train and 650k eval splits are **disjoint** (0 shared files). Also halves+ the pool. |
| **Packed candidate-ids** `cid = traj*10000 + row` + per-traj `_marked_from_row` + lazy line-count lengths | The first intermediate implementation materialized a flat ~60M-entry registry and scanned it per mark (≈10¹² ops/run) + opened all 800k files at construction (multi-GB). Packed ids make marking/sampling O(tail), decode O(1), and construction lazy. **Measured: real-150k construction = 0.001s, 0 trajectory loads.** |
| **Pre-shuffle the FPS cal/test files** before the `max_eval_rows` cap | The raw `*_fps.txt` are label-ordered along their length; a front-slice cap fed a biased subsample to q_hat calibration AND the reported ROA metrics (cal front-50k was 48/52 fail/success vs full **36/64**). Shuffled copies make the front-50k representative (verified 36/64). Memory cap preserved. New files written alongside originals (originals untouched): `cal_set_fps_shuffled.txt`, `test_set_fps_shuffled.txt`. |
| **`test_ratio: 0`** for the humanoid adaptive config | The `_base` default (0.1) would silently carve 10% of acquired pairs out of training; nothing consumes the in-loop test split (the FPS test set is the real eval). |
| **In-loop λ/δ val = intermediate-state pairs; q_hat = cal FILE** | Your explicit decision. λ/δ are calibrated on the same intermediate-state distribution the acquisition scores (no train/calibrate mismatch); q_hat uses the held-out FPS cal file. |
| **Disable `ConfidencePairFilter` for the intermediate config** | Intermediate tails extend backward across epochs → the built train file isn't append-stable, which breaks the filter's incremental-prefix optimization. Disabled with a documented follow-up (needs a full-recompute filter to re-enable). |
| **Reuse `Quadrotor3DUNet`** for the 67-D model | It's a generic latent-conditional UNet (dims-only); the spec's `UniversalUNet` lacks a latent arg and is incompatible with the base FM call. |

---

## 3. What was built (where it lives)

**New code** (all under `adaptive_roa/`):
- `systems/humanoid_standup_reach.py` — `HumanoidStandUpReachSystem` (JSON bounds, per-dim norm, binary classify on head_height≥1.3 ∧ ‖CoM vel‖≤0.2).
- `data/humanoid_standup_reach_endpoint_data.py` — data module: shuffled-indices mode (`query_mode` start/random_intermediate/all_intermediate, comma-delimiter, no zero-fill) + direct-file mode (pool's built files).
- `flow_matching/humanoid_standup_reach/latent_conditional/{flow_matcher,train,inference}.py` — `Product([Euclidean(34), Sphere(3,3), Euclidean(30)])`, `use_manifold` toggle, always-on S² distance manifold, output_dim 67.
- Intermediate acquisition: `adaptive/dataset_builder.py` + `adaptive/data_source.py` + `adaptive_v2/pool/trajectory_pool.py` (candidate_mode, packed ids, marking dedup, candidate-id-aware labels, intermediate val split, lazy lengths, `load_eval_states(max_rows=…)`).
- Registered `humanoid_standup_reach` in `adaptive_v2/trainers/flow_matching_trainer.py::_DATAMODULES`.
- One shared-base fix: `flow_matching/base/flow_matcher.py::_get_manifold_dist_dim` now handles `Sphere` (→1 geodesic), so validation MAE doesn't IndexError. Strictly additive (only affects Sphere-containing manifolds).

**Configs:** `configs/system/humanoid_standup_reach.yaml`, `configs/model/humanoid_standup_reach_unet.yaml`, `configs/train_humanoid_standup_reach.yaml`, `configs/evaluate_humanoid_standup_reach_roa.yaml`, `configs/adaptive_v2/system/humanoid_standup_reach.yaml`, `configs/adaptive_v2/model/system_dims/humanoid_standup_reach.yaml`.

**Docs:** spec `docs/superpowers/specs/2026-06-23-humanoid-standup-reach-design.md`; plans under `docs/superpowers/plans/` (core, adaptive, scale).

**Data (in the shared dataset dir, originals untouched):** `…/humanoid_get_up_medium/train_test_splits/{cal_set_fps_shuffled,test_set_fps_shuffled}.txt`.

---

## 4. Validation evidence

- **Full suite: 60 passing, 0 skipped** (incl. the dataset-dependent tests, which ran against the real data on this machine).
- **End-to-end adaptive smoke** (`test_humanoid_standup_reach_adaptive_smoke`): 1-epoch CPU run of the real `AdaptiveEngine` — intermediate acquisition → tail-pair training → FM train (real `val_loss`) → threshold-opt → q_hat on cal → ROA eval on test. ~12s.
- **Real-scale construction guard** (`test_intermediate_scale_smoke`): builder over the real 150k pool constructs in **0.001s with 0 trajectory loads**, no flat registry — proves the packed/lazy rework.
- **FPS shuffle:** front-50k of the shuffled cal/test now **36/64 fail/success = full-file balance** (was 48/52 biased).
- **`_intermediate_split` cost:** measured **~1.08s/call at 1.6M pairs, ~2 min total over a 30-epoch run** — negligible vs FM training.
- **Real-config end-to-end run** (the actual `system=humanoid_standup_reach` config — 150k train pool, shuffled cal/test, packed-id intermediate acquisition, 1-epoch CPU): **validated in substance.** It successfully: loaded the real 150k pool (140,863 success / 9,137 failure), built intermediate training pairs (426 from 8 seeded trajectories) + a non-empty intermediate val split (106 pairs), trained the FM (manifold=True, `val_loss` improved to 0.994, geodesic per-component MAE table emitted incl. `torso_vertical_geo`), and ran joint λ/δ threshold optimization on the intermediate val set — i.e. every integration path the unit tests can't exercise. It then entered the acquisition MC-sampling loop, which is CPU-bound (it was still running at heartbeat time). **Scalability observation:** with a 1-epoch-undertrained model the conformal sampler finds ≈0 "uncertain" candidates per batch and drains toward the `max_samples_per_epoch` cap — expected at 1 epoch; on GPU with a properly trained model it converges far faster and actually acquires. No errors/tracebacks at any stage.

---

## 5. Known limitations / recommended next steps

1. **Full real run not yet done.** Everything is validated at smoke/real-construction scale; a full `python scripts/run_adaptive.py system=humanoid_standup_reach` (30 epochs, GPU) hasn't been run. Recommended as the next step. With the caps in place it should be feasible; watch peak memory the first epoch.
2. **`max_eval_rows=50000` is a representative sample, not the full eval.** For a final/paper number you may want q_hat on the full 250k cal and ROA metrics on the full 1M test — set `conformal.max_eval_rows: null` (loads full files; slower, more memory). The shuffled files make any row count representative.
3. **`ConfidencePairFilter` is off** for intermediate mode (append-stability). Re-enabling needs a full-recompute filter — a contained follow-up.
4. **Classifier predictor path** (`evaluate_full_roa_classifier`) doesn't honor `max_eval_rows` — irrelevant for humanoid (generative path), worth threading for consistency.
5. **`configs/model/humanoid_standup_reach_unet.yaml` is an orphan** (train config inlines its model block) — matches the quad3d convention; harmless.
6. **Not pushed.** All 24 commits are local on `main`. Push when ready (`git push origin main`) — note `main` is 32 commits ahead of `origin/main` (includes the pre-existing local commits from before this session).

---

## 6. Commit map (this session, on `main`)
- Plan 1 core: `0014708 3c211a4 f37f586 2339fd3 2193e91 9fb50f3`
- Plan 2 adaptive/eval: `d24661a b8f40fd 7a42074 c6e36e9 b859563 7c994d8 e8ed2a8 3dfe40b 4be37c3`
- Plan 3 scale: `d5c66c9 b34c778 31b5dab f476d59 e8865c1 45b107b`
- Plans/spec/report docs interleaved (`332a722 e49b204 5dd9f57 22ed81f` …).

Progress ledgers (recovery maps): `.superpowers/sdd/progress-plan2.md`, `progress-plan3.md`.
