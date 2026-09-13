"""Per-member probabilities from a deep ensemble of scalar-outcome flow matchers.

Subclasses the single-model backend so `estimate`, `estimate_both`, the readout
guard and `p_invalid` are inherited unchanged: the handle answers
`p_success_mc` and `p_success_exact` with the ensemble mean, so the parent's
code is already correct for an ensemble. Only `estimate_members` is new.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from adaptive_roa.adaptive_v2.probability.outcome_fm import OutcomeFMProbabilityBackend


class EnsembleOutcomeFMProbabilityBackend(OutcomeFMProbabilityBackend):
    """Exposes the member axis that `DecompositionAcquisitionStrategy` needs.

    `member_sample_size` is None: under the `exact` readout each member's p is a
    continuous success mass of its learned flow map with no Monte-Carlo noise
    inside it, so the K-sample binomial debiasing that the endpoint-MC backends
    need would subtract a bias that is not there.

    Under the `mc` readout each member IS a K-sample estimate and BALD does
    carry the usual ~(1/2K)(1-1/M) upward bias. That is accepted rather than
    corrected, because the shipped arm runs `exact`; `bind_model` refuses a
    readout the model does not share, so the two cannot silently diverge.
    """

    member_sample_size: int | None = None

    def __init__(self, cfg: Any, system: Any, device: str):
        super().__init__(cfg, system, device)
        self.n_members = 0

    def bind_model(self, model_handle: Any) -> None:
        # Parent enforces that the readout NAME agrees between model and backend.
        super().bind_model(model_handle)
        n = int(getattr(model_handle, "n_members", 0))
        if n < 2:
            raise ValueError(
                f"ensemble outcome-FM backend needs at least 2 members, got {n}. A "
                "1-member 'ensemble' has no epistemic signal and would silently "
                "score BALD = 0 everywhere."
            )
        self._reject_readout_parameter_mismatch(model_handle)
        self.n_members = n

    def _reject_readout_parameter_mismatch(self, model_handle: Any) -> None:
        """Matching readout NAMES are not enough; the parameters must match too.

        Calibration and evaluation reach p through `model(x) -> logits`, which
        goes to `OutcomeFlowMatcher.predict_p_success` and uses the MEMBER's own
        `num_ode_steps` plus the flow matcher's DEFAULT grid_size=33 and
        bisect_iters=20. Acquisition reaches p through this backend, which passes
        `probability.*`. Equal names with unequal parameters therefore still
        computes two different numbers, silently, with both looking reasonable.

        Measured divergence, max|sigmoid(model(x)) - backend.estimate(x)|:
            shipped config (both 50 steps, grid 33, bisect 20)    2.6e-08
            model 10 steps against backend 50                     3.4e-04
            bisect_iters 4                                        1.5e-03
        The latter two are the order of the Brier gaps this ablation measures
        (probability/outcome_fm.yaml:27-32), so they would be a readout artefact
        masquerading as a result.
        """
        members = list(getattr(model_handle, "members", []))
        if not members:
            return
        member_steps = {int(getattr(m, "num_ode_steps", self.num_ode_steps)) for m in members}
        if member_steps != {self.num_ode_steps}:
            raise ValueError(
                f"num_ode_steps mismatch: probability.num_ode_steps={self.num_ode_steps} "
                f"but the members use {sorted(member_steps)}. Calibration and evaluation "
                f"would integrate the flow differently from acquisition. Set "
                f"predictor.outcome_fm.num_ode_steps to match."
            )

    @torch.no_grad()
    def estimate_members(self, start_states: np.ndarray, verbose: bool = False) -> np.ndarray:
        """[N, D] raw states -> [M, N] per-member success probabilities."""
        model = self._require_model()
        p = model.member_p_success(self._as_tensor(start_states))
        if verbose:
            print(f"    [EnsembleOutcomeFM] {p.shape[1]} states, M={p.shape[0]}, "
                  f"readout={self.readout}")
        return p.detach().cpu().numpy().astype(np.float64)
