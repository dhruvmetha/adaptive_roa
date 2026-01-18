# Adaptive Sampling (with Conformal Uncertainty): Problem Statement

## Background

This repo trains *flow-matching* surrogate models that predict the distribution of trajectory endpoints for dynamical systems (e.g., Pendulum, CartPole). Those endpoint predictions are used to answer a *region-of-attraction (ROA)* question:

> For an initial state \(x_0\), will the true system trajectory converge to the target attractor (SUCCESS) or not (FAILURE)?

Running the true simulator to label new initial states is expensive. The “adaptive” work here is about collecting (or selecting) *as little trajectory data as possible* while still building a reliable ROA classifier.

---

## Problem Statement

### Given

- A state space \(\mathcal{X}\) (often a product manifold with circular coordinates).
- An expensive simulator \(S\) that maps an initial state to an endpoint (or full trajectory):
  - \(x_T = S(x_0)\)
- A task label derived from the endpoint using a system-specific rule (implemented as `system.classify_attractor`):
  - \(y(x_0) \in \{-1, +1\}\) where \(+1=\) SUCCESS and \(-1=\) FAILURE.
- A learned probabilistic surrogate model \(M_\theta\) (a flow matcher) that can sample endpoints \(\hat{x}_T \sim p_\theta(\,\cdot\,|x_0)\) via latent sampling.
- A need to account for *model uncertainty* near the ROA boundary (the separatrix), where confident “success vs failure” decisions are most error-prone.

### Decide

Design a sequential data-acquisition policy (an “adaptive sampler”) that chooses which new initial states to add to the training set each epoch, using the current model’s uncertainty signal, so that:

- The ROA classifier improves quickly as a function of *data used* (i.e., number of training trajectories).
- The number of expensive simulator calls is minimized (or, in “pre-collected pool” mode, the number of *trajectories included/used* is minimized for a target quality).
- Reported uncertainty is calibrated, so “I don’t know” corresponds to a statistically meaningful abstention region.

Equivalently, you can view this as either:

- **Budgeted**: with a fixed simulator budget \(B\), maximize ROA classification quality while maintaining coverage.
- **Data-minimal**: find the smallest \(N\) trajectories such that coverage \(\ge 1-\alpha\) and performance targets (e.g., F1 and/or low unknown/separatrix rate) are met.

---

## Conformal Prediction as the Uncertainty Mechanism

The surrogate’s latent sampling is used to estimate a success probability:

- \(p_{\text{success}}(x_0) \approx \mathbb{P}(y=+1\mid x_0)\) (and optionally \(p_{\text{failure}}\) for two-sided rules).

Conformal prediction then turns these probability estimates into a *set-valued predictor* with a finite-sample coverage target:

- Produce a prediction set \(\Gamma(x_0) \subseteq \{-1, 0, +1\}\) where \(0\) is “UNKNOWN/abstain”.
- Guarantee (under standard conformal assumptions) \( \mathbb{P}\big(y(x_0)\in \Gamma(x_0)\big) \ge 1-\alpha \).

Operationally in this codebase, a point is treated as **uncertain** if its conformal prediction set is *not a singleton* (i.e., the model is not confident). These uncertain points are the ones prioritized for new simulation / inclusion.

---

## What “Adaptive” Means in This Repo

At a high level, an adaptive epoch is:

1. **Train** the flow matcher on the current training trajectories.
2. **Fit** a conformal predictor:
   - Optimize a decision boundary parameter (typically \(\lambda^\*\), optionally \(\delta^\*\)) with a loss that trades off misclassification vs abstention.
   - Calibrate a conformal threshold \(q_{\hat{}}\) to target coverage \(1-\alpha\).
3. **Propose candidates** (either by sampling the state space, or by drawing the next batch from a pre-collected trajectory pool).
4. **Split candidates** into:
   - **D1**: always added (provides fresh calibration/training signal),
   - **D2**: selectively added based on conformal uncertainty.
5. **Select** only the uncertain subset of D2 (skip confident candidates) and add/simulate them.
6. **Evaluate** on a held-out test set and/or a full ROA label set for monitoring.

This is the concrete mechanism behind plots like “F1 vs dataset size” and “Separatrix% vs dataset size”.

### Note on “Data Used” in Plots

Some adaptive run scripts record metrics (e.g., full ROA F1 / separatrix%) **before** adding the newly selected samples for the *next* epoch, but log `train_trajectories` **after** adding those samples. For post-hoc “metric vs dataset size” plots, the x-axis is therefore computed as:

- `data_used = train_trajectories - n_d1_added - n_d2_added`

See `scripts/plot_data_vs_metrics.py` and `scripts/compare_data_vs_metrics.py`.

---

## Success Criteria (How We Judge “Better”)

The adaptive method is intended to improve the *data-efficiency* of ROA classification. Common evaluation signals in this repo include:

- **F1** on confident predictions (treating abstentions as “unknown” rather than forcing a label).
- **Separatrix / unknown rate**: fraction of points where the predictor abstains.
- **Coverage**: fraction of cases where the true label lies in the conformal prediction set (target \(1-\alpha\)).
- **Savings** vs a non-adaptive baseline: how many simulations/trajectories were skipped because the model was confident.

---

## Experiments & Ablations: Comparison Axes

This section contextualizes the main experiment “axes” (things we sweep) and maps them onto the code and logged artifacts (e.g., `epoch_XXX/results.json`, `epoch_XXX/full_roa_evaluation.json`).

### Axis Glossary (what each knob actually controls)

- **\(N_{\text{train}}\) / training dataset size**: number of trajectories in the training split used to train the flow matcher for the current epoch (often `len(train_split)`).
- **\(K_{\text{add}}\) / per-epoch add budget**: number of trajectories incorporated into training at the end of an epoch (targeted by `adaptive_data_max` in balanced sampling, or variable under fixed sampling).
- **\(K_{\text{eval}}\) / per-epoch uncertainty-evaluation budget**: number of candidate trajectories *scored* by the current model to decide uncertainty (this drives compute cost and can be much larger than \(K_{\text{add}}\) in balanced sampling; see `n_total_sampled`).
- **\(\rho\) / “uncertain percent”**:
  - **target \(\rho\)**: `d2_ratio` (fraction of the per-epoch add budget intended to be uncertainty-filtered),
  - **realized \(\rho\)**: fraction of actually-added points that were uncertain (can differ from target if uncertain points are rare; balanced sampling back-fills with certain points).
- **Pool policy**: whether “skipped” candidates remain available for future epochs (balanced) or are consumed/removed by the sequential pointer (fixed sampling).

1. **Dataset sizes (non-adaptive baselines)**  
   Sweep the *training dataset size* \(N\) (number of trajectories used to train the flow matcher), e.g. \(N \in \{50, 100, 150, \ldots, 1000\}\). This is the natural x-axis for “F1 vs dataset size” and “Separatrix% vs dataset size”.

2. **Fixed thresholds with multiple \(\delta\) values**  
   When using *fixed* thresholds, \(\delta\) controls the width of the abstention/unknown band around a fixed \(\lambda=0.5\):
   - One-sided: UNKNOWN if \(p_{\text{success}} \in [0.5-\delta,\; 0.5+\delta]\)
   - Two-sided (CartPole-style): confident only if \(p_{\text{success}} > 0.5+\delta\) **or** \(p_{\text{failure}} > 0.5+\delta\)  
   A sweep like \(\delta \in \{0.1, 0.2, \ldots, 0.5\}\) (note: configs/code require \(\delta < 0.5\)) directly measures the abstention–error trade-off. The commonly used “notebook” threshold \(0.6\) corresponds to \(\delta=0.1\) (since \(0.5+\delta=0.6\)).

3. **Adaptive sampling: initial dataset size** (`initial_train_size`, per system)  
   Initial training size can be held at 50 for some systems (e.g., pendulum/cartpole configs), but may need to be larger for others (e.g., mountain car uses 500 in `configs/adaptive_mountain_car.yaml`). This ablation probes how much “bootstrap” data is needed before uncertainty signals become useful.

4. **Adaptive sampling: iteration dataset size / add budget** (`adaptive_data_max` or `samples_per_epoch`)  
   Control how many new trajectories are incorporated per epoch, e.g., 50 vs 100. In the pre-collected pool scripts (`src/adaptive/run_adaptive_*.py`), this is typically:
   - `sampling_strategy: balanced_uncertain` → per-epoch additions are fixed by `adaptive_data_max`
   - `sampling_strategy: fixed` → candidates are `samples_per_epoch` and additions depend on uncertainty filtering

5. **Adaptive sampling: “uncertain percent” / uncertainty mix** (`d2_ratio`, all systems)  
   In balanced sampling, `d2_ratio` is the target fraction of the per-epoch add budget routed through uncertainty filtering (D2) vs random additions (D1): \(0=\) all random, \(1=\) all uncertainty-filtered. Sweeps like 50% and 75% correspond to `d2_ratio ∈ {0.5, 0.75}`.

6. **Conformal prediction: \(\alpha\) values** (`conformal.alpha`, typically one system)  
   \(\alpha\) sets the nominal coverage target \(1-\alpha\). Sweeping \(\alpha\) tests the calibration–abstention trade-off: lower \(\alpha\) (higher target coverage) typically increases abstention/unknown rate.

7. **Cold start vs warm start** (`warm_start`, typically one system)  
   In `src/adaptive/run_adaptive_*.py`, warm start resumes model weights from the previous epoch’s best checkpoint; cold start trains from scratch each epoch. This separates “data helps” from “optimizer/initialization helps” (and is also a compute/time ablation).

8. **Conformal adaptive vs variance adaptive (heuristic)** (all systems)  
   - *Conformal adaptive*: select points based on conformal uncertainty (e.g., `ConformalPredictor.select_uncertain`, which uses calibrated `q_hat`). In the pre-collected scripts, this is the selection mechanism in the `sampling_strategy: fixed` branch.
   - *Threshold adaptive* (also used in this repo): treat points as uncertain if they fall in the \(\lambda^\*\pm\delta^\*\) “unknown band” (no `q_hat`), as in `src/adaptive/balanced_sampler.py`.
   - *Variance adaptive*: select points when the MC dispersion is high (e.g., high variance/entropy of \(p_{\text{success}}\) from multiple latent samples), using a variance threshold or top‑k rule. This can emulate “100% uncertain” if the threshold is set low, but provides no formal coverage guarantee.

9. **Threshold fixed vs threshold optimization** (`conformal.optimize_mode`, all systems)  
   - Fixed: choose \(\lambda,\delta\) by hand (or use the notebook-style fixed threshold, e.g., 0.6).
   - Optimized: learn \(\lambda^\*\) (optimize_mode=`lambda`) or \(\delta^\*\) (optimize_mode=`delta` with fixed \(\lambda=0.5\)), then calibrate `q_hat` for coverage.

10. **Adaptive vs non-adaptive datasets** (all systems)  
    Compare uncertainty-filtered dataset growth to non-adaptive controls at *matched dataset size and/or matched simulation budget*, e.g.:
    - Non-adaptive: `d2_ratio=0` (all random additions) or “add everything”
    - Adaptive: `d2_ratio>0` with uncertainty filtering

11. **Generative vs non-generative successor prediction** (all systems)  
    - Generative successor prediction: latent-conditional flow matcher + MC sampling → \(p_{\text{success}}(x)\) and uncertainty.
    - Non-generative successor prediction: deterministic endpoint predictor (or single-sample successor) → requires a separate heuristic for abstention/uncertainty.

12. **Successor prediction vs classification** (all systems)  
    - Successor prediction: learn \(x_0 \mapsto \hat{x}_T\) (or a distribution over endpoints), then derive ROA labels via `system.classify_attractor`.
    - Classification: learn \(x_0 \mapsto y\) directly (a pure classifier baseline).

### Additional axes (present in the code, easy to miss)

13. **Sampling strategy & pool retention** (`sampling_strategy`, all systems with pre-collected pools)  
    Two strategies in `src/adaptive/run_adaptive_*.py` change the meaning of “skipping”:
    - `sampling_strategy: fixed` (sequential pointer): confident D2 candidates are *skipped and consumed* (the pointer advances past them), so they cannot be revisited later.
    - `sampling_strategy: balanced_uncertain` (explicit `used_indices`): confident candidates are *discarded but kept in the pool* unless they are actually added to training.
    This affects both fairness of comparisons and the effective candidate distribution over epochs.

14. **Uncertainty definition used for *selection*** (all systems)  
    There are (at least) two different uncertainty signals used in selection:
    - **Calibrated conformal set uncertainty** (`ConformalPredictor.select_uncertain`): uses `q_hat` (coverage-driven).
    - **Threshold-band uncertainty** (`src/adaptive/balanced_sampler.py`): treats points inside \(\lambda^\*\pm\delta^\*\) as uncertain (does *not* use `q_hat`).
    If you want a clean ablation, treat this as an explicit axis: *selection via q_hat* vs *selection via λ±δ* (vs future variance heuristics).

15. **Conformal loss trade-off \(w\)** (`conformal.w`, all systems)  
    \(w\) controls the objective optimized for \(\lambda^\*\) / \(\delta^\*\):  
    \(\text{Loss} = w \cdot \text{MisclassRate} + (1-w)\cdot \text{UnknownRate}\).  
    This is a direct axis for “accuracy vs abstention”, and interacts strongly with any \(\delta\) sweep or `optimize_mode=delta`.

16. **Calibration split ratio** (`conformal.calibration_ratio`, all systems)  
    Controls how much labeled training data is held out for calibrating `q_hat` vs optimizing \(\lambda^\*\)/\(\delta^\*\). This impacts both the stability of the conformal guarantee and the quality of the decision boundary.

17. **Decision rule: one-sided vs two-sided** (`conformal.decision_rule`, system-dependent)  
    Whether uncertainty/thresholding depends only on \(p_{\text{success}}\) (one-sided) or also on \(p_{\text{failure}}\) (two-sided, CartPole-style).

18. **MC sampling fidelity vs compute** (`conformal.num_mc_samples`, `num_mc_samples_eval`, `mc_batch_size`)  
    Number of latent samples used to estimate \(p_{\text{success}}\)/\(p_{\text{failure}}\) during fitting and evaluation. This is an axis for estimation noise and runtime.

19. **Labeling “physics” threshold** (`attractor_radius`, system-dependent)  
    The radius used in `system.classify_attractor` changes the ground-truth label assignment from endpoints, and therefore changes both training targets and evaluation difficulty. It’s effectively part of the task definition.

20. **Evaluation protocol** (all systems)  
    - **Which dataset**: “test set” as implemented in the adaptive pool scripts is a subset of the training indices (overlap), whereas “full ROA” evaluation scores against all labeled start states.
    - **Which threshold scheme**: `full_roa_evaluation.json` contains both \(\lambda^\*\pm\delta\) metrics and “notebook-style” fixed thresholds (e.g., 0.6).

### More sensitivity axes (real, but usually “secondary”)

21. **Optimization search space & resolution** (`lambda_grid_size`, `delta_grid_size`, `delta_min`, `delta_max`)  
    These do not change the *definition* of uncertainty, but can change the learned \(\lambda^\*\)/\(\delta^\*\) (and therefore both abstention and accuracy) due to finite grid resolution and bounded search ranges.

22. **Candidate-search compute limits** (`batch_size_sampling`, `max_samples_per_epoch`)  
    In balanced uncertain sampling, these parameters cap how hard the algorithm tries to *find* enough uncertain points. If the cap is too low, the run will back-fill with “certain” points and the realized uncertainty mix will deviate from the target.

23. **Randomness & pool ordering** (`seed`, `shuffled_indices_file`)  
    Since candidates are drawn in the order of `shuffled_indices.txt`, the shuffle (and seed used to create it) is an implicit axis that can affect learning curves—especially at small dataset sizes.

24. **Flow-matcher capacity & training/inference budget** (`latent_dim`, model size, `trainer.max_epochs`, `flow_matching.num_integration_steps`)  
    These are “model axes” rather than “adaptive axes”, but they can dominate performance and interact with uncertainty: better-calibrated uncertainty estimates typically require sufficient model capacity and enough training.

### LaTeX checklist (paper-friendly)

```tex
\begin{enumerate}
    \item \textbf{Comparison axes}
    \begin{enumerate}
        \item \textbf{Dataset sizes (non-adaptive baselines)}: 50, 100, 150, \ldots, 1000
        \quad(\textit{x-axis = \# training trajectories used to train the successor/flow matcher})

        \item \textbf{Fixed thresholds, multiple $\delta$}: 0.1, 0.2, \ldots, 0.5
        \quad(\textit{fixed $\lambda=0.5$; unknown band is $[0.5-\delta,\,0.5+\delta]$})

        \item \textbf{Adaptive sampling -- initial dataset size}: 50
        \quad(\textit{config: \texttt{initial\_train\_size}; may be system-dependent})

        \item \textbf{Adaptive sampling -- per-iteration add budget}: 50, 100
        \quad(\textit{config: \texttt{adaptive\_data\_max} or \texttt{samples\_per\_epoch}})

        \item \textbf{Adaptive sampling -- uncertain percent}: 50\%, 75\%
        \quad(\textit{config: \texttt{d2\_ratio}; fraction of per-epoch adds that are uncertainty-filtered})

        \item \textbf{Conformal prediction -- $\alpha$ values}
        \quad(\textit{config: \texttt{conformal.alpha}; sweep on one system if compute-limited})

        \item \textbf{Adaptive cold start vs warm start}
        \quad(\textit{config: \texttt{warm\_start}; resume from previous epoch checkpoint vs retrain from scratch})

        \item \textbf{Conformal adaptive vs variance adaptive (variance threshold)}
        \quad(\textit{conformal uses calibrated sets; variance is a heuristic, no guarantees})

        \item \textbf{Threshold fixed vs threshold optimization}
        \quad(\textit{config: \texttt{conformal.optimize\_mode} in \{lambda, delta\}})

        \item \textbf{Adaptive vs non-adaptive datasets}
        \quad(\textit{match dataset size / budget; e.g. \texttt{d2\_ratio}=0 vs >0})

        \item \textbf{Generative vs non-generative successor prediction}
        \quad(\textit{latent/MC successor vs deterministic successor})

        \item \textbf{Successor prediction vs classification}
        \quad(\textit{predict endpoints then label vs predict labels directly})

        \item \textbf{Sampling strategy / pool retention (fixed vs balanced)}
        \quad(\textit{are skipped points revisited later or consumed by the pointer?})

        \item \textbf{Uncertainty definition for selection (q\_hat vs $\lambda\pm\delta$ vs variance)}
        \quad(\textit{calibrated conformal sets vs threshold-band heuristic vs MC-variance heuristic})

        \item \textbf{Conformal trade-off weight $w$}
        \quad(\textit{misclassification vs unknown/abstention in $\lambda^\*/\delta^\*$ optimization})

        \item \textbf{Calibration split ratio}
        \quad(\textit{config: \texttt{conformal.calibration\_ratio}})

        \item \textbf{MC sample count / compute budget}
        \quad(\textit{config: \texttt{num\_mc\_samples}, \texttt{num\_mc\_samples\_eval}})

        \item \textbf{Attractor radius (label definition)}
        \quad(\textit{config: \texttt{attractor\_radius}; changes what counts as success/failure})

        \item \textbf{Evaluation protocol}
        \quad(\textit{full ROA vs subset; $\lambda^\*\pm\delta$ vs fixed 0.6 thresholds})

        \item \textbf{Optimization grid resolution / search bounds}
        \quad(\textit{\texttt{lambda\_grid\_size}, \texttt{delta\_grid\_size}, \texttt{delta\_min/max}})

        \item \textbf{Candidate-search compute cap}
        \quad(\textit{\texttt{batch\_size\_sampling}, \texttt{max\_samples\_per\_epoch}})

        \item \textbf{Shuffle / seed}
        \quad(\textit{\texttt{seed} and \texttt{shuffled\_indices\_file} define candidate order})

        \item \textbf{Model capacity / training budget}
        \quad(\textit{\texttt{latent\_dim}, model size, \texttt{trainer.max\_epochs}, \texttt{num\_integration\_steps}})
    \end{enumerate}
\end{enumerate}
```

---

## Where This Lives in the Code

- Conformal prediction: `src/conformal/`
- Adaptive sampling (core loop + analysis): `src/adaptive/`
- Plotting “data used vs metrics”: `scripts/plot_data_vs_metrics.py`, `scripts/compare_data_vs_metrics.py`
- Full adaptive run scripts (system-specific): `src/adaptive/run_adaptive_*.py`
