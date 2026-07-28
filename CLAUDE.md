# CLAUDE.md

**This repo estimates Regions of Attraction** — which initial states a controller actually drives to
the goal — comparing generative (flow matching) / discriminative (classifier) / geometric (Part-X GP)
representations under one acquisition loop with conformal uncertainty. Systems run from a 2-D
pendulum to a 67-D humanoid. **It is NOT a flow-matching library**; it began as one, which is why
retired docs say so.

Only rules you'd get wrong *before thinking to look them up* live here. Everything else:
**`docs/INDEX.md`**.

## Names that lie
You act on a name before you form a question, so these have to be here.

- **`adaptive/` is NOT legacy.** It is the load-bearing data layer that `adaptive_v2` imports, and it
  holds the repo's only DATA_DIR write guard. Do not remove it.
- **`adaptive_v2` is NOT version 2 of `adaptive/`.** They are different layers — a loop, and the data
  layer it depends on. The `_v2` suffix implies a succession that does not exist.
- **`partial_trajs` is NOT a classifier over partial trajectories.** It is a T-step forward-dynamics
  surrogate, rolled out autoregressively and judged by the analytic `classify_attractor`.
- **There is no `src/`.** The package is `adaptive_roa/`. Any `src/...` reference is stale.
- **`../adaptive_roa` (sibling dir) is a STALE TWIN CHECKOUT of this same repo** — dead
  pre-force-push history, and its own 359-line CLAUDE.md still teaches the removed `src/` layout.
  Do not work there.
- **`../deepreach` is your fork of Bansal's HJ reachability**, with Pendulum + CartPole added. It is
  the reachability baseline, and nothing else in this repo mentions it exists.

## Silent traps
Wrong answers, no error. Nothing here fails loudly enough to teach you.

- **`DATA_DIR` = `/common/users/shared/pracsys/genMoPlan/data_trajectories` is shared, READ-ONLY**
  (owned by `st1122`). Never write or delete under it. Exactly **one** code path guards this
  (`adaptive_roa/adaptive/data_source.py` raises `ValueError`); every other write path is unguarded and will
  succeed quietly.
- **Labels flip meaning between layers.** On disk `{0,1}`; internally `{-1, 0, +1}` where **`0` is
  separatrix**; but in `evaluate_roa` predictions **`0` is failure** and `-1` is *uncertain*, and
  `-1`/`-2` are silently dropped from metrics — so F1 is computed on a retained subset and is not
  comparable across methods with different abstention rates.
- **NFS breaks DataLoader workers** (`OSError: Errno 16` cleaning `.nfs*`). The classifier trainer
  pins `num_workers=0`; the FM trainer and `adaptive_roa/data/trajectory_data.py` still default to **4**.
- **`noise_regime=noisy` still does not work — but `system=pendulum_stoch` does.** The regime flag
  only builds path strings. The stochastic path is a *separate* opt-in: `pool_format: npz` selects
  `NpzTrajectoryDataSource` in `adaptive_roa/adaptive_v2/pool/trajectory_pool.py`, reading a single
  flat `train.npz` — not the per-cell format the old note described.
- **`sampling_mode=` is cosmetic, not a selector.** Each `configs/adaptive_v2/acquisition/*.yaml`
  sets it `@package _global_`, and it is interpolated into `output_dir` and nothing else. The loop
  branches on `self.acquisition.mode` (`adaptive_roa/adaptive_v2/engine.py`), so overriding it on
  the CLI renames your output directory and changes no behaviour. Select with `acquisition=`.

## Environment & paths
- `conda activate /common/users/dm1487/envs/arcmg` · `pip install -e .`
- Paths resolve through `.env` (`NET_ID`, `DATA_DIR`, `EXP_DIR`, `SHARED_DATA_BASE`, `NOISE_REGIME`)
  and Hydra resolvers `${data_dir:}`, `${exp_dir:}`, `${shared_data_base:...}`. Never hardcode
  user-specific paths.
- **Outputs go under `EXP_DIR`** (`/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/`)
  or `outputs/`. Write only there.
- Data layout: `{shared_data_base}/{noise_regime}/{dataset}/`. Each dataset ships
  `dataset_description.json` (state_dim, manifold, goal, criteria) — read it rather than hardcoding
  system facts. It has twice been right where the code's comments were wrong.

## Commands
```bash
# Flow-matching training: adaptive_roa/flow_matching/<system>/<variant>/train.py
python adaptive_roa/flow_matching/pendulum/latent_conditional/train.py
# systems: pendulum, cartpole, mountain_car, pendulum_cartesian, humanoid_standup_reach, quadrotor_3d

# Flow-matching ROA eval (add evaluation.probabilistic=true for uncertainty)
python adaptive_roa/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa

# adaptive_v2 loop (predictor = generative | classifier | gp), config_name="default"
# NOTE: top-level `d2_ratio=` was REMOVED 2026-06-29. Select via the acquisition group:
python scripts/run_adaptive.py system=quadrotor2d predictor=classifier acquisition=direct acquisition.d2_ratio=1.0 device=cuda:0
# Part-X GP + level-set BO variant:
python scripts/run_adaptive.py system=pendulum predictor=gp acquisition=partx eval=partx device=cuda:0

# partial-trajectory T-step verifier
python adaptive_roa/partial_trajs/train.py system=pendulum

# tests (use the arcmg python)
pytest
```
- Hydra config groups live in `configs/`; override on the CLI (`system=cartpole device=cuda:0`).
- SLURM: submit via `scripts/sbatch_run.sh`. This repo runs on iLab, not Amarel.

## How to work here
- **State the exact question before reading broadly.** Breadth without a question produces
  confident wrong answers — it has, repeatedly, here.
- **Verify against code before acting on a name, a comment, or a doc.** This repo's prose has been
  wrong while its code was right at least three separate times, and twice two independent sources
  agreed with each other and were both wrong. Consensus among comments is not evidence.
- **Cite the source file for every behavioural claim.** If you can't cite it, you don't know it.
- Docs route you; **code is the evidence**. When they conflict, the code wins.
- **Never delete — archive** (`git mv` to an `archive/` dir: keeps history, unmaps it from the
  reading path). A wrong doc is worse than a missing one; a missing one sends you to the code.

## Where to look
**`docs/INDEX.md`** routes everything by intent. **`docs/archive/` is retired and untrue — never
cite it.**

## Maintaining this file
Only **silent** or **pre-conscious** failures belong here: things you'd get wrong before you thought
to look them up. Anything that fails loudly (a command errors, an import breaks) teaches you by
failing — leave it out; that is how `mem 40G` and other trivia got in here and got it wrong.
**Only `docs/INDEX.md` names files**, so moving a file breaks exactly one file. Never write a line
you have not verified against the code.
