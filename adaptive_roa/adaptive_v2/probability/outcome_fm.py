"""Probability backend for the scalar-outcome flow matcher.

Two readouts, both computed, because the ablation needs to tell model
miscalibration apart from sampling noise:

``mc``     K-sample fraction. Identical procedure to ``EndpointMCProbabilityBackend``,
           so a gap against endpoint FM is attributable to the target, not the
           readout. Quantised to 1/K.
``exact``  Success mass of the learned 1D flow map, computed by bracketing and
           bisecting the crossing in source space. Continuous, no MC noise, and
           cheaper than the MC estimate (grid + bisect < K integrations at K=100).

``estimate`` returns whichever ``readout`` names -- that is what the pipeline
calibrates and acquires on -- while ``estimate_both`` exposes the pair for
scoring. p_invalid is identically zero: a scalar outcome flow has no notion of
"failed to reach an attractor", which is itself part of the finding, since
endpoint-MC's p_invalid carried real signal at high noise.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class OutcomeFMProbabilityBackend:
    """Probability backend over a 1D outcome flow (MC and/or exact readout)."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.num_mc_samples = int(cfg.get("num_mc_samples", 100))
        self.num_ode_steps = int(cfg.get("num_ode_steps", 100))
        self.grid_size = int(cfg.get("grid_size", 33))
        self.bisect_iters = int(cfg.get("bisect_iters", 20))
        self.chunk_size = int(cfg.get("chunk_size", 4096))
        readout = str(cfg.get("readout", "exact")).lower()
        if readout not in {"mc", "exact"}:
            raise ValueError(f"readout must be 'mc' or 'exact', got {readout!r}")
        self.readout = readout
        self.system = system
        self.device = device
        self.model_handle: Any = None
        # Fraction of points whose flow map was NOT monotone on the last exact
        # call. Surfaced so a silent fallback to quadrature cannot be mistaken
        # for a clean bisection result.
        self.last_nonmonotone_fraction: float | None = None

    def bind_model(self, model_handle: Any) -> None:
        # The threshold/calibration path reaches p through `model(x) -> logits`
        # (ClassifierProbabilityEstimator), while eval reaches it through this
        # backend. If the two readouts differ, calibration optimises against a
        # different quantity than evaluation reports -- silently, and with both
        # numbers looking reasonable. Refuse the configuration instead.
        model_readout = getattr(model_handle, "forward_readout", None)
        if model_readout is not None and model_readout != self.readout:
            raise ValueError(
                f"readout mismatch: probability.readout={self.readout!r} but the model's "
                f"forward_readout={model_readout!r}. Calibration would use one and evaluation "
                f"the other. Set predictor.outcome_fm.forward_readout to match."
            )
        self.model_handle = model_handle

    def _as_tensor(self, start_states: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(start_states, dtype=np.float32), device=self.device
        )

    def _require_model(self) -> Any:
        if self.model_handle is None:
            raise RuntimeError("Probability backend used before bind_model")
        return self.model_handle

    def estimate_both(self, start_states: np.ndarray) -> dict[str, np.ndarray]:
        """Both readouts plus the monotonicity diagnostic, for scoring."""
        model = self._require_model()
        states = self._as_tensor(start_states)

        p_mc = model.p_success_mc(
            states,
            num_samples=self.num_mc_samples,
            num_steps=self.num_ode_steps,
            chunk_size=self.chunk_size,
        )
        p_exact, monotone = model.p_success_exact(
            states,
            num_steps=self.num_ode_steps,
            grid_size=self.grid_size,
            bisect_iters=self.bisect_iters,
            chunk_size=self.chunk_size,
        )
        self.last_nonmonotone_fraction = float((~monotone).to(torch.float64).mean().item())
        return {
            "p_success_mc": p_mc.detach().cpu().numpy().astype(np.float64),
            "p_success_exact": p_exact.detach().cpu().numpy().astype(np.float64),
            "monotone": monotone.detach().cpu().numpy(),
        }

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        model = self._require_model()
        states = self._as_tensor(start_states)

        if self.readout == "mc":
            p = model.p_success_mc(
                states,
                num_samples=self.num_mc_samples,
                num_steps=self.num_ode_steps,
                chunk_size=self.chunk_size,
            )
        else:
            p, monotone = model.p_success_exact(
                states,
                num_steps=self.num_ode_steps,
                grid_size=self.grid_size,
                bisect_iters=self.bisect_iters,
                chunk_size=self.chunk_size,
            )
            self.last_nonmonotone_fraction = float(
                (~monotone).to(torch.float64).mean().item()
            )

        p_success = p.detach().cpu().numpy().astype(np.float64)
        return OutcomeProbabilities(
            p_success=p_success,
            p_failure=1.0 - p_success,
            p_invalid=np.zeros_like(p_success),
        )
