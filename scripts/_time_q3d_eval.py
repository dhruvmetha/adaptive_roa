import time, torch
from adaptive_roa.flow_matching.quadrotor_3d.latent_conditional.flow_matcher import Quadrotor3DLatentConditionalFlowMatcher
from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem
from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_fast

CKPT="/common/users/shared/pracsys/adaptive_roa_experiments/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_40_alpha_0.1_sampling_mode_ranked/2026-01-29_08-08-36/epoch_038/checkpoints/best-832-0.0129.ckpt"
TEST="/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/quadrotor3D_lqr/test_set.txt"
dev="cuda:0"
print("loading checkpoint...", flush=True)
t0=time.time()
fm=Quadrotor3DLatentConditionalFlowMatcher.load_from_checkpoint(CKPT, device=dev)
fm.eval().to(dev)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

for K in (20,):
    t0=time.time()
    m=evaluate_full_roa_fast(flow_matcher=fm, system=Quadrotor3DSystem(),
        eval_states_file=TEST, num_mc_samples=K, batch_size=30000,
        lambda_star=0.5, delta=0.1, attractor_radius=0.2, device=dev,
        output_dir=None, verbose=True)
    dt=time.time()-t0
    ld=m.get("lambda_delta",{})
    print(f"\n>>> K={K}: eval wall-time {dt:.1f}s ({dt/60:.1f} min) on 990k grid; band_F1={ld.get('f1')}", flush=True)
