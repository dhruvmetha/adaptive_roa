#!/usr/bin/env python3
"""Re-fly the zero-noise quad2d eval states and record how each flight ended.

The shipped eval (`q2_corridor_collect.py --split eval`, level 0) stores only a 0/1
label per state. Its roll() labels a flight that reaches the 1200-step horizon
without terminating as a failure, so a timeout cannot be told apart from a wall
failure in the files. This re-flies rows [lo, hi) of the eval file with the
collector's own build() and seeds, and records for each: the success flag, steps,
whether the env terminated, and the final state (file order). The flight is
deterministic (f_max 0, ambient 0), so the success flag must reproduce the stored
label; the reducer reports any disagreement.

Run inside the safe-control-gym env with SCG_REPO pointing at a checkout that has
q2_corridor_common.py:
    q2d_refly.py --eval-file eval_states.txt --lo 0 --hi 10000 --out shard_0.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SCG = os.environ["SCG_REPO"]
sys.path.insert(0, SCG)
os.chdir(SCG)          # the policy checkpoint path in q2_corridor_common is relative
from q2_corridor_common import HORIZON, build, rollout_seed  # noqa: E402
from generate_quadrotor_2d_trajectories_rl import normalize_angle  # noqa: E402

BASE_SEED, EVAL_SPLIT_ID = 20260817, 1


def fly(env, ctrl, state6, seed):
    """q2_corridor_common.roll(), also returning termination and the final state."""
    import pybullet as pb
    x, z, theta, x_dot, z_dot, theta_dot = state6
    obs, info = env.reset(seed=int(seed))
    pb.resetBasePositionAndOrientation(env.DRONE_ID, [x, 0, z], pb.getQuaternionFromEuler([0, theta, 0]),
                                       physicsClientId=env.PYB_CLIENT)
    pb.resetBaseVelocity(env.DRONE_ID, [x_dot, 0, z_dot], [0, theta_dot, 0], physicsClientId=env.PYB_CLIENT)
    env._update_and_store_kinematic_information()
    obs = env._get_observation()
    if getattr(env, "constraints", None) is not None:
        info["constraint_values"] = env.constraints.get_values(env, only_state=True)
    success, ended, steps = False, False, 0
    for steps in range(1, HORIZON + 1):
        action = ctrl.select_action(ctrl.obs_normalizer(obs), info)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success, ended = bool(info.get("goal_reached", False)), bool(terminated)
            break
    xx, xd, zz, zd, th, td = obs[:6]
    return success, steps, ended, [xx, zz, normalize_angle(th), xd, zd, td]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"{a.out} exists, skipping")
        return
    rows = np.loadtxt(a.eval_file, delimiter=",", skiprows=a.lo, max_rows=a.hi - a.lo, ndmin=2)
    env, ctrl = build(0.0, model="sine")
    res = {k: [] for k in ("success", "steps", "terminated", "final")}
    try:
        for i, r in enumerate(rows):
            s, n, t, f = fly(env, ctrl, r[:6], rollout_seed(BASE_SEED, EVAL_SPLIT_ID, a.lo + i, 0))
            for k, v in zip(res, (s, n, t, f)):
                res[k].append(v)
    finally:
        env.close()
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, index=np.arange(a.lo, a.hi), stored_label=(rows[:, 6] >= 0.5).astype(np.uint8),
             success=np.asarray(res["success"], np.uint8), steps=np.asarray(res["steps"], np.int32),
             terminated=np.asarray(res["terminated"], np.uint8), final=np.asarray(res["final"], np.float64),
             horizon=HORIZON)
    os.replace(tmp, a.out)
    print(f"[{a.lo}:{a.hi}] {int(np.sum(res['success']))} successes, "
          f"{int(np.sum(~np.asarray(res['terminated'], bool)))} not terminated", flush=True)


if __name__ == "__main__":
    main()
