# adaptive_roa

**Estimating Regions of Attraction for controlled dynamical systems.** Given a controller and a
goal: *which initial states does the closed-loop system actually drive to the goal?*

That set matters because you cannot deploy a controller without knowing where it works, and you
cannot compose controllers (funnels, LQR-trees) without bounding each one's basin. Classical
certification — Lyapunov/SOS, HJ reachability — is provable but doesn't scale past a few
dimensions and needs a closed-form model, which an RL policy isn't. So this repo estimates the RoA
from **simulation rollouts** instead, on systems from a 2-D pendulum to a 67-D humanoid.

Two facts drive the design: nearly all the information lives at the **boundary** (the separatrix),
and **simulation is the binding constraint**. Hence adaptive sampling near the boundary, plus
conformal prediction for calibrated uncertainty — over-claiming the RoA means the robot falls.

The research question compares **representations**, all forced through one acquisition loop and one
common contract, `estimate(states) -> (p_success, p_failure, p_invalid)`:

| Predictor | Models the RoA as |
|---|---|
| `classifier` (discriminative) | a set-membership function `p(success\|x)` — one forward pass |
| `generative` (flow matching) | an endpoint distribution — expresses multimodality and a third "invalid" outcome |
| `gp` / Part-X (geometric) | a measurable set with a **volume + credible interval** |
| `partial_trajs` | reachability under a learned T-step dynamics surrogate (~20 net calls vs ~500 sim steps) |

> **This is not a flow-matching library.** It began as one, which is why retired documents describe
> it that way. Flow matching is one representation among several.

## Start here

- **`docs/PROBLEM.md`** — the problem, why it's hard, what we've learned
- **`docs/INDEX.md`** — everything else, routed by what you're trying to do
- **`CLAUDE.md`** — environment, paths, commands, and the gotchas that bite

## Quick start

```bash
conda activate /common/users/dm1487/envs/arcmg
pip install -e .
pytest
```

Training, evaluation and the adaptive loop are documented in `CLAUDE.md`, along with the `.env`
path setup (`NET_ID`, `DATA_DIR`, `EXP_DIR`). Never hardcode user-specific paths — resolve them
through `.env`. **`DATA_DIR` is shared, read-only trajectory data: never write under it.**

## Layout

```
adaptive_roa/   the package  (there is no src/ — it was renamed)
configs/        Hydra config groups
scripts/        run_adaptive.py is the adaptive_v2 entry point
docs/           start at docs/INDEX.md;  docs/archive/ is retired and not true
tests/          pytest
```
