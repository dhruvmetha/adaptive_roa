from __future__ import annotations

from typing import Any

from adaptive_roa.adaptive_v2.types import AcquisitionResult
from adaptive_roa.partx.acquisition import select_pool_indices
from adaptive_roa.partx.bounds import roa_volume_bound
from adaptive_roa.partx.tree import PartitionTree


class PartXAcquisitionStrategy:
    mode = "partx"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.d2_ratio = float(cfg.d2_ratio)
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

        system = getattr(pool, "system", None) or probability_backend.system
        latent_fn = probability_backend.latent_posterior
        if self.tree is None:
            self.tree = PartitionTree(
                system,
                branching_factor=int(self._tree_cfg.get("branching_factor", 2)),
                delta=float(self._tree_cfg.get("delta", 0.05)),
                alpha=float(self._tree_cfg.get("alpha", 0.05)),
                m_class=int(self._tree_cfg.get("m_class", 64)),
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
        return AcquisitionResult(
            d2_indices=list(selected),
            n_candidates_evaluated=len(cand_indices),
            n_certain_discarded=len(cand_indices) - len(selected),
            n_invalid_added=0,
            diagnostics=diagnostics,
        )
