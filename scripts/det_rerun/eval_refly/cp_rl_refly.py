#!/usr/bin/env python3
"""Re-fly the zero-noise PPO cartpole eval states and record how each flight ended.

The shipped set (stochastic/cartpole/gaussian_signal/safe_explorer_ppo/baseline)
stores only a 0/1 label per eval state, so a flight that reached the 150-step
horizon without terminating cannot be told apart from a wall failure. Its collector
is not in any checkout we can read, so this rebuilds the plant from the dataset
description (ctrl 15 Hz, pyb 750 Hz, quadratic cost, goal x = 0.7, entry-cut 0.05
ball over the wrapped 4-D state, walls 3.0 / 3.0 / 0.48 / 3.0, the shipped
safe_explorer_ppo checkpoint with its normalizer frozen) and checks it first:

    --validate  replays stored TRAIN flights from their starts and compares every row
    (no eval)   re-flies rows [lo, hi) of the eval file, recording success, steps,
                whether the env terminated, and the final state (file order)

Run inside the safe-control-gym env with SCG_REPO set; paths must be absolute.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from functools import partial

import numpy as np

SCG = os.environ["SCG_REPO"]
sys.path.insert(0, SCG)
os.chdir(SCG)          # the checkpoint and config paths below are repo-relative
import yaml  # noqa: E402
from safe_control_gym.utils.registration import make  # noqa: E402

CKPT = "examples/rl/models/safe_explorer_ppo/safe_explorer_ppo_model_cartpole_stab.pt"
CFG = "examples/rl/config_overrides/cartpole"
HORIZON = 150                      # 10 s at 15 Hz
TOL = 0.05
WALLS = dict(x=3.0, x_dot=3.0, theta=0.48, theta_dot=3.0)
ENV2FILE = [0, 2, 1, 3]            # env [x, x_dot, theta, theta_dot] -> file [x, theta, x_dot, theta_dot]


def build():
    task = yaml.safe_load(open(f"{CFG}/cartpole_stab.yaml"))["task_config"]
    # The override file carries only what cartpole changes; the rest is the controller's own defaults.
    algo = yaml.safe_load(open("safe_control_gym/controllers/safe_explorer/safe_ppo.yaml"))
    algo.update(yaml.safe_load(open(f"{CFG}/safe_explorer_ppo_cartpole.yaml"))["algo_config"])
    algo["training"] = False     # inference only: load() then skips the optimizer and experiment state
    tmp = tempfile.mkdtemp(prefix="cprl-")
    kw = dict(task="stabilization", ctrl_freq=15, pyb_freq=750, gui=False, randomized_init=False,
              randomized_inertial_prop=False, normalized_rl_action_space=True, action_scale=10.0,
              inertial_prop=task["inertial_prop"], episode_len_sec=10, cost="quadratic", obs_goal_horizon=0,
              task_info={"stabilization_goal": [0.7, 0], "stabilization_goal_tolerance": TOL},
              constraints=task["constraints"], done_on_out_of_bound=True, done_on_violation=False,
              terminate_on_goal=True, output_dir=tmp)
    env_func = partial(make, "cartpole", **kw)
    ctrl = make("safe_explorer_ppo", env_func, **algo, output_dir=tmp)
    ctrl.load(CKPT)
    ctrl.obs_normalizer.set_read_only()
    env = env_func()
    env.x_threshold, env.x_dot_threshold = WALLS["x"], WALLS["x_dot"]
    env.theta_threshold_radians, env.theta_dot_threshold = WALLS["theta"], WALLS["theta_dot"]
    return env, ctrl


def fly(env, ctrl, s_file, seed=0, keep=False):
    import pybullet as pb
    x, theta, x_dot, theta_dot = (float(v) for v in s_file)
    env.reset(seed=int(seed))
    # The state lives in PyBullet: it must go through the joints (cp_collect.py).
    pb.resetJointState(env.CARTPOLE_ID, 0, targetValue=x, targetVelocity=x_dot, physicsClientId=env.PYB_CLIENT)
    pb.resetJointState(env.CARTPOLE_ID, 1, targetValue=theta, targetVelocity=theta_dot, physicsClientId=env.PYB_CLIENT)
    env.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
    obs = env._get_observation()
    info = {"current_step": 0, "constraint_values": env.constraints.get_values(env, only_state=True)}
    traj = [env.state[ENV2FILE].copy()] if keep else None
    success, ended, steps = False, False, 0
    for steps in range(1, HORIZON + 1):
        obs, _, term, trunc, info = env.step(ctrl.select_action(ctrl.obs_normalizer(obs), info))
        if keep:
            traj.append(env.state[ENV2FILE].copy())
        if term or trunc:
            success, ended = bool(info.get("goal_reached", False)), bool(term)
            break
    return success, steps, ended, env.state[ENV2FILE].copy(), traj


def validate(train_npz: str, idx: np.ndarray) -> None:
    tr = np.load(train_npz)
    S, off, lab = tr["states"], tr["offsets"], tr["labels"]
    env, ctrl = build()
    worst, len_bad, lab_bad = 0.0, 0, 0
    for i in idx.tolist():
        ok, _, _, _, traj = fly(env, ctrl, tr["starts"][i], keep=True)
        mine = np.asarray(traj, np.float32)
        ref = S[off[i]:off[i + 1]]
        if len(mine) != len(ref):
            len_bad += 1
        else:
            worst = max(worst, float(np.abs(mine - ref).max()))
        lab_bad += int(ok) != int(lab[i])
    env.close()
    print(f"validate {len(idx)} train flights: length mismatches {len_bad}, label mismatches {lab_bad}, "
          f"max abs row difference {worst:.3g}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", default=None, help="train.npz to replay")
    ap.add_argument("--n", type=int, default=500, help="train flights to replay (seeded random pick)")
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.validate:
        n_all = len(np.load(a.validate)["labels"])
        validate(a.validate, np.random.default_rng(0).choice(n_all, size=a.n, replace=False))
        return
    if os.path.exists(a.out):
        print(f"{a.out} exists, skipping")
        return
    rows = np.loadtxt(a.eval_file, delimiter=",", skiprows=a.lo, max_rows=a.hi - a.lo, ndmin=2)
    env, ctrl = build()
    res = {k: [] for k in ("success", "steps", "terminated", "final")}
    try:
        for r in rows:
            s, n, t, f, _ = fly(env, ctrl, r[:4])
            for k, v in zip(res, (s, n, t, f)):
                res[k].append(v)
    finally:
        env.close()
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, index=np.arange(a.lo, a.hi), stored_label=(rows[:, 4] >= 0.5).astype(np.uint8),
             success=np.asarray(res["success"], np.uint8), steps=np.asarray(res["steps"], np.int32),
             terminated=np.asarray(res["terminated"], np.uint8), final=np.asarray(res["final"], np.float64),
             horizon=HORIZON)
    os.replace(tmp, a.out)
    print(f"[{a.lo}:{a.hi}] {int(np.sum(res['success']))} successes, "
          f"{int(np.sum(~np.asarray(res['terminated'], bool)))} not terminated", flush=True)


if __name__ == "__main__":
    main()
