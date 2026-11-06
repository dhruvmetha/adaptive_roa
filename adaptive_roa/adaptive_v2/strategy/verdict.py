"""det_v2 acquisition (method section 5): q_A or U_H over the next unused starts.

Each round scores the first ``n_candidates`` unused trajectory starts in pool order
(all of them when fewer remain), excluding the round's uniform picks, and takes
the batch by ``selection``: ``topk`` (the main rule) or ``proportional``.

Scoring is batched: final-state models draw K endpoints for ``batch_size`` states
per model call (prediction_store.mc_predict, as evaluation does), and the BNN
classifier draws K posterior samples of p_s per batch. A uniform arm is this
strategy at d2_ratio = 0, which the engine never calls.

The scores go to the epoch directory as acquisition_scores.npz (the engine pops
``diagnostics["_arrays"]``), so each round's ranking can be inspected later.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from adaptive_roa.adaptive_v2.eval import prediction_store as ps
from adaptive_roa.adaptive_v2.strategy import verdict_scores as vs
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState

SCORES = ("q_a", "u_h")
SELECTIONS = ("topk", "proportional")


class VerdictAcquisitionStrategy:
    def __init__(self, cfg: Any):
        self.score = str(cfg.score)
        if self.score not in SCORES:
            raise ValueError(f"acquisition.score must be one of {SCORES}, got {self.score!r}")
        self.selection = str(cfg.get("selection", "topk"))
        if self.selection not in SELECTIONS:
            raise ValueError(f"acquisition.selection must be one of {SELECTIONS}, got {self.selection!r}")
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.get("n_candidates", 50000))
        self.batch_size = int(cfg.get("batch_size", 2048))
        if self.batch_size < 2048:
            raise ValueError(f"acquisition.batch_size={self.batch_size}: det_v2 scores at least 2,048 states per call")
        self.num_mc_samples = int(cfg.num_mc_samples)
        self.seed = int(cfg.get("seed", 0))
        self.verbose = bool(cfg.get("verbose", True))
        self.mode = "uniform" if self.d2_ratio <= 0 else self.score

    # ------------------------------------------------------------------ scores
    def _candidate_scores(self, states: np.ndarray, probability_backend: Any, is_classifier: bool,
                          tau: tuple[float, float] | None, round_seed: int) -> dict[str, np.ndarray]:
        model = probability_backend.model_handle
        system = probability_backend.system
        device = probability_backend.device
        out: dict[str, np.ndarray] = {}
        if not is_classifier:
            labels, _ = ps.mc_predict(model, system, states, self.num_mc_samples,
                                      probability_backend.attractor_radius,
                                      batch_size=self.batch_size, device=device)
            c = ps.counts_from_labels(labels)
            n_s, n_f, n_i = (c[k].astype(np.int64) for k in ("n_success", "n_failure", "n_invalid"))
            k = float(self.num_mc_samples)
            out.update(n_success=n_s, n_failure=n_f, n_invalid=n_i)
            if tau is not None:
                out.update(vs.verdict_probabilities(n_s, n_f, n_i, *tau))
            out["u_h"] = vs.u_h(n_s / k, n_f / k, n_i / k)
            return out
        # Classifier: p_s from the handle (the marginal threshold opt and evaluation read).
        p = np.asarray(probability_backend.estimate(states).p_success, dtype=float)
        out["p_success"] = p
        out["u_h"] = vs.u_h(p)
        if tau is not None:
            n_marg = int(getattr(model, "n_marginal_samples", 1))
            if n_marg > 1 and hasattr(model, "p_success_draws"):
                draws = model.p_success_draws(states, self.num_mc_samples, seed=round_seed,
                                              batch_size=self.batch_size)
            else:
                draws = p[None, :]          # MLP: one draw, so q_A is the indicator of A
            out.update(vs.classifier_verdict_probabilities(draws, *tau))
        return out

    def _embed(self, system: Any, states: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(states, dtype=np.float32))
        with torch.no_grad():
            return system.embed_state_for_model(system.normalize_state(x)).cpu().numpy().astype(float)

    # ---------------------------------------------------------------- select
    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        if target_count <= 0:
            return AcquisitionResult(diagnostics={"skipped_reason": "target_count_zero"})
        tau = None
        if getattr(threshold_state, "tau_s_star", None) is not None:
            tau = (float(threshold_state.tau_s_star), float(threshold_state.tau_f_star))
        if self.score == "q_a" and tau is None:
            raise ValueError("acquisition.score=q_a needs tau* from threshold=j_sel")

        t0 = time.time()
        states, indices = pool.sample_candidates_without_marking(self.n_candidates, exclude=exclude or set())
        indices = list(indices)
        if len(indices) == 0:
            return AcquisitionResult(diagnostics={"skipped_reason": "pool_exhausted"})
        states = np.asarray(states, dtype=np.float32)
        builder = pool.dataset_builder
        round_seed = int(np.random.SeedSequence([self.seed, len(builder.used_indices)]).generate_state(1)[0])
        is_classifier = str(threshold_backend.predictor_type) == "classifier"
        sc = self._candidate_scores(states, probability_backend, is_classifier, tau, round_seed)
        t_score = time.time() - t0
        u = sc["q_a"] if self.score == "q_a" else sc["u_h"]

        system = probability_backend.system
        rolled = np.asarray(builder.data_source.get_start_states(sorted(builder.used_indices)), dtype=np.float32) \
            if builder.used_indices else np.zeros((0, states.shape[1]), dtype=np.float32)

        def distance(pos: np.ndarray) -> np.ndarray:
            if len(rolled) == 0:
                return np.full(len(pos), np.inf)
            return vs.nearest_distance(self._embed(system, states[pos]), self._embed(system, rolled))

        if self.selection == "topk":
            picks, sel_diag = vs.select_topk(u, target_count, distance)
        else:
            rng = np.random.default_rng(round_seed)
            picks, sel_diag = vs.select_proportional(u, target_count, rng)
        selected = [indices[i] for i in picks]
        mask = np.zeros(len(indices), dtype=bool)
        mask[picks] = True

        diag: dict[str, Any] = {
            "score": self.score, "selection": self.selection, "num_mc_samples": self.num_mc_samples,
            "tau_s_star": None if tau is None else tau[0], "tau_f_star": None if tau is None else tau[1],
            "n_candidates": len(indices), "n_selected": len(selected),
            "n_positive_score": int(np.sum(u > 0)),
            "score_mean": float(np.mean(u)), "score_max": float(np.max(u)),
            "score_selected_min": float(np.min(u[picks])) if len(picks) else None,
            "score_selected_mean": float(np.mean(u[picks])) if len(picks) else None,
            "score_seconds": round(t_score, 2),
            **sel_diag,
        }
        if "q_a" in sc:
            h = vs.verdict_entropy_bits(sc)
            diag.update(verdict_entropy_mean=float(np.mean(h)),
                        verdict_entropy_selected_mean=float(np.mean(h[picks])) if len(picks) else None,
                        q_a_mean=float(np.mean(sc["q_a"])))
        arrays = {"candidate_indices": np.asarray(indices, dtype=np.int64), "selected": mask,
                  "score": np.asarray(u, dtype=np.float32)}
        for key in ("q_s", "q_f", "q_a", "u_h", "p_success"):
            if key in sc:
                arrays[key] = np.asarray(sc[key], dtype=np.float32)
        for key in ("n_success", "n_failure", "n_invalid"):
            if key in sc:
                arrays[key] = np.asarray(sc[key], dtype=np.int16)
        diag["_arrays"] = arrays
        if self.verbose:
            print(f"    [{self.score}/{self.selection}] {len(indices)} candidates in {t_score:.1f}s, "
                  f"{diag['n_positive_score']} with score > 0; took {len(selected)} "
                  f"(score >= {diag['score_selected_min']})")
        return AcquisitionResult(
            d1_indices=[],
            d2_indices=selected,
            n_candidates_evaluated=len(indices),
            n_certain_discarded=len(indices) - len(selected),
            n_invalid_added=0,
            diagnostics=diag,
        )
