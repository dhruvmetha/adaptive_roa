# Region-of-Attraction Estimation and Controller Chaining

## What This Project Does

Estimate which initial states reach the goal — the Region of Attraction (ROA) — at high dimension using only simulation rollouts. Compare learned discriminative, generative, and geometric representations. Core aim: chain controllers together by learning when one's terminal states land safely in another's basin, calibrated so probabilities are honest.

## Where We Are

**Working:** Discriminative (classifier) and flow-matching ROA estimators trained at multiple scales, 2D pendulum through 67D humanoid. Conformal prediction calibrates per-state uncertainty. Adaptive loop samples efficiently near decision boundary.

**Developing:** Epistemic/aleatoric uncertainty decomposition (guides sampling cost-benefit). Geometric Part-X GP and partial-trajectory forward-dynamics surrogates. Multi-controller graph search; prototype swing-up → catch achieves 38% → 99% composite success on pendulum.

**Open:** End-to-end calibrated chain-success probability. Generative model's honest handling of high-dimensional multimodal terminal distributions. Composition under heterogeneous controller families.

## What To Judge Papers Against

**Core:** Does it estimate/certify ROA at dimension ≥6D? Does it calibrate uncertainty in learned sets? Does it address controller composition via learned edge probabilities?

**Supporting:** Does it separate epistemic (sampling-reducible) from aleatoric (irreducible) uncertainty? Does it work with dynamics surrogates instead of analytic models?

**Known gaps:** Reachability certification (classical) maxes at ~6D. RL chaining (DSG, GSC) learns composition uncalibrated. Generative-model uncertainty either assumes Gaussian or requires ensembles. Formal composition (Žikelić, Neary) gives end-to-end guarantees but via certificates not classifiers, and stays low-D.
