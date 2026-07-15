---
read_when: "you need to know what problem this repo solves — read before touching anything"
class: living
last_verified: 2026-07-15
ground_truth: [adaptive_roa/systems/, docs/research_journal/]
---

# The problem: Region-of-Attraction estimation

## In one line

Given a controller and a goal, **which initial states does the closed-loop system actually
drive to the goal?** That set is the Region of Attraction (RoA). This repo estimates it from
simulation data, as cheaply as possible, with calibrated uncertainty.

> **This is NOT a flow-matching library.** Flow matching is *one of several representations
> being compared* (see `TARGET.md`). `README.md`'s title is historical and wrong — the repo
> started as an FM implementation and grew a comparison harness around it. If you orient off
> the README you will misunderstand the entire codebase.

## Why robotics needs it

- **Safety / verification.** Start outside the RoA and the controller does not recover — the
  robot falls, the quadrotor crashes. You want to know the set *before* deploying.
- **Composition.** Funnel / LQR-tree planning chains controllers by feeding one controller's
  exit set into the next controller's RoA. **You cannot compose what you cannot bound.** This
  is the reason RoA is a set-valued question and not a per-trajectory one.
- **Certification.** A guarantee over a *set* of initial conditions, rather than evidence from
  a handful of lucky rollouts.

## Why not the classical answer

The textbook approach is a Lyapunov certificate, with sum-of-squares programming to search for
one. It yields *provable* inner approximations, but the semidefinite program grows rapidly with
state dimension and polynomial degree, so in practice it is confined to low-dimensional systems
and simple dynamics. It also needs a model in closed form — which an RL policy isn't.

This repo's system ladder runs from a **2-D pendulum** to a **67-D humanoid**, with a PyBullet
cartpole and quadrotors in between. Classical certification is not on the table at that scale.

So: **estimate the RoA from rollouts.** That converts a verification problem into a *learning*
problem with a *sampling* problem inside it — which is where all the difficulty moves.

## The two facts that dominate every design decision

**1. All of the information is at the boundary.**
The interior (obviously succeeds) and the exterior (obviously fails) are nearly free to learn.
Everything hard lives on the **separatrix** — the boundary — which in the deterministic case is
a measure-zero set that winds through phase space. Our own pendulum runs show this directly:
the unresolved states concentrate along the swing-up/energy boundary and at high |θ̇|, not in
the interior.

**2. Simulation is the binding constraint.**
Every label costs a rollout. The budget — not model capacity — is what limits the estimate.

Together these give the repo its shape: **spend the budget near the boundary** (adaptive /
active sampling), and **know how much to trust the answer** (conformal calibration). Trust
matters asymmetrically: claiming a state is in the RoA when it isn't means the robot falls, so
a *conservative* estimate beats an accurate-on-average one.

## What we have actually learned (evidence, not intuition)

- **Compute asymmetry is a result, not an implementation detail.** Classifier evaluation is one
  forward pass over the grid — cheap at any size. Flow-matching evaluation is
  `num_mc_samples_eval` ODE solves × grid size — *infeasible* on quad3d's ~990k grid at default
  MC. This alone shapes which methods are usable at scale
  (`research_journal/2026-06-21-classifier-vs-flowmatching.md`).
- **RoA difficulty ≠ state dimension.** quad2d presented a harder boundary than the
  higher-dimensional quad3d. Dimension is not the hardness axis — **boundary geometry is.**
- **Point-prediction accuracy and region-level confidence are different things.** A predictor can
  classify individual states well while set-level methods still fail to certify any region at the
  same budget — only the latter is cursed by dimension. Worth keeping distinct when reading any
  result.

## Where this is going

The deterministic problem assumes the RoA *is a set*. Under stochastic dynamics it stops being
one: the target becomes a probability field `p(success|x) ∈ [0,1]` with an irreducible
uncertainty band. That changes the object being estimated, not just the noise level — see
`TARGET.md`. **Not yet built.**

## Ground truth

Systems and success criteria: `adaptive_roa/systems/`. Evidence and results:
`docs/research_journal/`. Routing for everything else: `docs/INDEX.md`.
