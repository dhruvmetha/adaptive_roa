#!/bin/bash
# Re-evaluate all systems' last epochs to get coverage_confident metric.
# All runs have MC caches so this is CPU-only (no GPU needed).
set -e

SCRIPT="python scripts/reevaluate.py"

# ── PENDULUM (radius=0.075, mc=20, batch=100000) ──────────────────────────
PEND_BASE="/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_pendulum_dhruv/outputs"
PEND_ARGS="--attractor_radius 0.075 --alpha_eval 0.1 --num_mc_samples 20 --batch_size 100000"

echo "=== PENDULUM ==="

echo "[Pend] Non-adaptive (direct)"
$SCRIPT "${PEND_BASE}/training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-28_13-15-25" \
    $PEND_ARGS --epoch 19

echo "[Pend] Adaptive direct d2=1.0 v2"
$SCRIPT "${PEND_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-03-01_11-11-14" \
    $PEND_ARGS --epoch 19

echo "[Pend] Adaptive direct d2=1.0"
$SCRIPT "${PEND_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-20_01-56-07" \
    $PEND_ARGS --epoch 19

echo "[Pend] Adaptive ranked d2=1.0"
$SCRIPT "${PEND_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-55-55" \
    $PEND_ARGS --epoch 19

echo "[Pend] Adaptive direct d2=0.75"
$SCRIPT "${PEND_BASE}/training_index_0_d2_ratio_0.75_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-25_12-11-22" \
    $PEND_ARGS --epoch 19

echo "[Pend] Adaptive direct d2=0.5"
$SCRIPT "${PEND_BASE}/training_index_0_d2_ratio_0.5_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-25_12-11-15" \
    $PEND_ARGS --epoch 19

echo "[Pend] Manifold adaptive d2=0.5"
$SCRIPT "${PEND_BASE}/training_index_0_d2_ratio_0.5_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-26_15-22-40" \
    $PEND_ARGS --epoch 19

echo "[Pend] Non-adaptive (ranked)"
$SCRIPT "${PEND_BASE}/training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-56-20" \
    $PEND_ARGS --epoch 19

# ── CARTPOLE (radius=0.2, mc=20, batch=100000) ────────────────────────────
CART_BASE="/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_cartpole_pybullet/outputs"
CART_ARGS="--attractor_radius 0.2 --alpha_eval 0.1 --num_mc_samples 20 --batch_size 100000"

echo ""
echo "=== CARTPOLE ==="

echo "[Cart] Non-adaptive"
$SCRIPT "${CART_BASE}/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_14-26-34" \
    $CART_ARGS --epoch 19

echo "[Cart] Adaptive direct d2=1.0"
$SCRIPT "${CART_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-18_14-26-32" \
    $CART_ARGS --epoch 19

echo "[Cart] Adaptive ranked d2=1.0"
$SCRIPT "${CART_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_19-22-14" \
    $CART_ARGS --epoch 19

echo "[Cart] Adaptive direct d2=0.75"
$SCRIPT "${CART_BASE}/training_index_0_d2_ratio_0.75_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-25_12-10-13" \
    $CART_ARGS --epoch 19

echo "[Cart] Adaptive direct d2=0.5"
$SCRIPT "${CART_BASE}/training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-25_12-10-05" \
    $CART_ARGS --epoch 19

# ── QUADROTOR 2D (radius=0.3, mc=10, batch=100000, cal_k=1000) ───────────
Q2D_BASE="/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs"
Q2D_ARGS="--attractor_radius 0.3 --alpha_eval 0.1 --num_mc_samples 10 --batch_size 100000 --cal_k 1000"

echo ""
echo "=== QUADROTOR 2D ==="

echo "[Q2D] Non-adaptive d2=0.0"
$SCRIPT "${Q2D_BASE}/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-17-55" \
    $Q2D_ARGS --epoch 9

echo "[Q2D] Adaptive d2=0.5"
$SCRIPT "${Q2D_BASE}/training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-17-39" \
    $Q2D_ARGS --epoch 9

echo "[Q2D] Adaptive d2=0.75"
$SCRIPT "${Q2D_BASE}/training_index_0_d2_ratio_0.75_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-16-05" \
    $Q2D_ARGS --epoch 9

echo "[Q2D] Adaptive d2=1.0"
$SCRIPT "${Q2D_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-05-32" \
    $Q2D_ARGS --epoch 9

echo "[Q2D] Adaptive d2=1.0 fixed"
$SCRIPT "${Q2D_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_12-09-02" \
    $Q2D_ARGS --epoch 9

# ── QUADROTOR 3D (radius=0.3, mc=10, batch=100000, cal_k=1000) ───────────
Q3D_BASE="/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs"
Q3D_ARGS="--attractor_radius 0.3 --alpha_eval 0.1 --num_mc_samples 10 --batch_size 100000 --cal_k 1000"

echo ""
echo "=== QUADROTOR 3D ==="

echo "[Q3D] Non-adaptive d2=0.0 (15iter)"
$SCRIPT "${Q3D_BASE}/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_11-49-54" \
    $Q3D_ARGS --epoch 14

echo "[Q3D] Adaptive d2=0.5 (15iter)"
$SCRIPT "${Q3D_BASE}/training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_11-49-55" \
    $Q3D_ARGS --epoch 14

echo "[Q3D] Adaptive d2=0.75 (15iter)"
$SCRIPT "${Q3D_BASE}/training_index_0_d2_ratio_0.75_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_11-50-16" \
    $Q3D_ARGS --epoch 14

echo "[Q3D] Adaptive d2=1.0 (15iter, fixed)"
$SCRIPT "${Q3D_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_12-01-36" \
    $Q3D_ARGS --epoch 14

echo "[Q3D] Adaptive d2=1.0 (20iter, direct)"
$SCRIPT "${Q3D_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-26_12-02-36" \
    $Q3D_ARGS --epoch 16

echo "[Q3D] Adaptive d2=1.0 (20iter, ranked)"
$SCRIPT "${Q3D_BASE}/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-26_12-02-36" \
    $Q3D_ARGS --epoch 16

echo ""
echo "=== ALL DONE ==="
