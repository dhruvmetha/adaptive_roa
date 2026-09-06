from __future__ import annotations

from typing import Any

from adaptive_roa.adaptive_v2.types import AcquisitionResult
from adaptive_roa.partx.acquisition import select_pool_indices
from adaptive_roa.partx.bounds import roa_volume_bound
from adaptive_roa.partx.tree import PartitionTree


class PartXAcquisitionStrategy:
    mode = "partx"

    # Part-X partitions the state space by the SIGN of a robustness function and
    # acquires on the straddle of its zero level set. That requires a predictor
    # whose latent IS the robustness / label probability. A predictor that models
    # the final state instead (gp_reg, fm_*, anything on the endpoint-MC backend)
    # has no such latent, and pairing one with this strategy is a category error,
    # not a configuration choice.
    _LATENT_CONTRACT = (
        "Part-X requires a predictor that models the robustness score / label "
        "probability, not the final state. Valid: predictor=gp (GP classifier). "
        "Invalid: predictor=gp_reg or any endpoint/final-state model, whose "
        "probability backend has no latent_posterior to partition on."
    )

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.d2_ratio = float(cfg.d2_ratio)
        if self.d2_ratio <= 0.0:
            # engine.py computes n_d2_target = samples_per_epoch - int(
            # samples_per_epoch * (1 - d2_ratio)), so d2_ratio=0 makes
            # need_d2_acquisition False and select() is NEVER called. The arm
            # then draws its whole budget uniformly while its config still says
            # acquisition=partx, which is how gaussian_torque/cp_med_partx came
            # to sit in the Part-X family as a pure uniform baseline for 12
            # epochs with no acquisition block in any results.json.
            raise ValueError(
                f"acquisition=partx with d2_ratio={self.d2_ratio} never runs: the "
                "engine skips d2 acquisition entirely, so the arm is a uniform "
                "baseline mislabelled as Part-X. Use d2_ratio>0, or select a "
                "different acquisition."
            )
        self.beta = float(cfg.get("beta", 1.96))
        self.allocation = str(cfg.get("allocation", "per_region_volume"))
        self.n_candidates = int(cfg.get("n_candidates", 50000))
        self._tree_cfg = cfg.get("tree", {})
        self._bounds_cfg = cfg.get("bounds", {})
        self.tree: PartitionTree | None = None

    def select(self, pool, probability_backend, threshold_backend,
               threshold_state, target_count, exclude=None) -> AcquisitionResult:
        if target_count <= 0:
            return AcquisitionResult(diagnostics={"skipped_reason": "target_count_zero"})

        system = probability_backend.system
        latent_fn = getattr(probability_backend, "latent_posterior", None)
        if not callable(latent_fn):
            raise TypeError(
                f"{type(probability_backend).__name__} exposes no latent_posterior. "
                + self._LATENT_CONTRACT
            )
        if self.tree is None:
            self.tree = PartitionTree(
                system,
                branching_factor=int(self._tree_cfg.get("branching_factor", 2)),
                delta=float(self._tree_cfg.get("delta", 0.05)),
                alpha=float(self._tree_cfg.get("alpha", 0.05)),
                m_class=int(self._tree_cfg.get("m_class", 64)),
                max_leaves=self._tree_cfg.get("max_leaves", None),
            )
        # 1) refine tree against the current GP posterior
        self.tree.refine(latent_fn)

        # 2) score available pool candidates, select within unresolved leaves
        cand_states, cand_indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude
        )
        m, s2 = latent_fn(cand_states)
        selected, acq_diag = select_pool_indices(
            self.tree, cand_states, cand_indices, m, s2,
            target_count=target_count, beta=self.beta, allocation=self.allocation,
        )

        # 3) region bounds for reporting
        bounds = roa_volume_bound(
            self.tree, latent_fn,
            R=int(self._bounds_cfg.get("R", 200)),
            M=int(self._bounds_cfg.get("M", 64)),
        )
        diagnostics = {
            **acq_diag,
            "n_leaves": len(self.tree.leaves()),
            "n_remaining_leaves": len(self.tree.remaining_leaves()),
            "roa_volume": bounds["volume"],
            "roa_volume_ci": [bounds["ci_low"], bounds["ci_high"]],
        }
        # expose diagnostics to the evaluator without changing the engine signature
        try:
            probability_backend.model_handle.partx_diag = diagnostics
            probability_backend.model_handle.partx_tree = self.tree
        except Exception:
            pass

        return AcquisitionResult(
            d2_indices=list(selected),
            n_candidates_evaluated=len(cand_indices),
            n_certain_discarded=len(cand_indices) - len(selected),
            n_invalid_added=0,
            diagnostics=diagnostics,
        )
