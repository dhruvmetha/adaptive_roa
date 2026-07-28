"""Model handle binding a final-state predictor to the endpoint-MC backend.

Satisfies the contract ``ProbabilityEstimator`` and ``compute_endpoint_prediction_error``
already rely on, so the Bayesian final-state arms route through the existing
conformal, threshold, and evaluation machinery unchanged.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import torch


class _ManifoldDistanceShim:
    """Minimal ``distance_manifold`` stand-in exposing ``dist``.

    full_roa.py guards get_manifold_component_names() behind
    hasattr(model, "distance_manifold"), so supplying only one of the pair means
    the per-component error stats are silently skipped (or crash). This shim
    delegates to the head so both consumers agree.
    """

    def __init__(self, head):
        self._head = head

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RAW-in, RAW-out -- deliberately NOT normalized.

        Its only caller is full_roa.py:621,649, which passes raw predicted and
        raw actual endpoints and writes the result to
        ``full_roa["endpoint_errors"]``. That call site hands the FM family's
        ``distance_manifold.dist`` raw states too, so keeping this raw is what
        makes ``endpoint_errors`` comparable across the two families.
        Normalizing here would fix nothing and would instead reintroduce the
        same cross-family unit mismatch in a different JSON field.

        Note this is NOT the same convention as
        ``FinalStateModelHandle.compute_manifold_distance_per_component``, which
        normalizes because its FM counterpart does. The two conventions are a
        property of the two CALL SITES, not an inconsistency here.
        """
        return self._head.distance_per_component(x, y)


class FinalStateModelHandle:
    """Predicts a distribution over x_T and samples ONE endpoint per call.

    Determinism here is the OPPOSITE of ``OutcomeModelHandle``: the estimator
    calls ``predict_endpoint`` K times on the same batch and the spread across
    those calls IS the outcome probability. Every call therefore draws a fresh
    weight sample AND a fresh head sample. Seeding this handle collapses every
    arm to p in {0, 1}.

    KNOWN ESTIMATOR ASYMMETRY (ensemble arm)
    ----------------------------------------
    ``_params`` calls ``posterior.forward_sample``, which for an ensemble draws
    ONE member uniformly at random per call. Over K calls the marginal is
    therefore a K-draws-WITH-REPLACEMENT mixture, whose member weights are a
    multinomial rather than the exact 1/M.

    The outcome family does NOT do this: ``EnsemblePosterior`` overrides
    ``predictive_logit_samples`` to ENUMERATE all M members exactly
    (``posteriors.py:166-179``), precisely because sampling the atoms is a
    systematic bias rather than noise that averages out. That override is
    unavailable here: this handle's contract is one fresh draw per call, so
    there is no point at which K draws are visible together to be replaced by an
    enumeration.

    Consequence, stated so it is not invisible in results: ``bnn_ensemble_reg``
    and its outcome-family sibling are NOT the same estimator quality even at
    equal M. At the shipped K=10, M=5 the member-weight standard deviation is
    sqrt(p(1-p)/K) = 0.126 against an exact p = 0.2, i.e. ~63% relative. See
    ``FinalStateTrainer._resolve_num_mc_samples`` for the guard and its warning.
    """

    def __init__(self, posterior, head, system: Any, device: str = "cpu"):
        self.posterior = posterior
        self.head = head
        self.system = system
        self.device = device
        self.training = False
        self.distance_manifold = _ManifoldDistanceShim(head)

    def eval(self):
        self.posterior.eval()
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.posterior.train(mode)
        self.training = bool(mode)
        return self

    def to(self, device):
        self.device = device
        self.posterior.to(device)
        return self

    def _params(self, states: torch.Tensor) -> torch.Tensor:
        embedded = self.system.embed_state_for_model(self.system.normalize_state(states))
        return self.posterior.forward_sample(embedded)

    def predict_endpoint(self, states) -> torch.Tensor:
        """[B, state_dim] raw -> [B, state_dim] raw. Fresh sample every call."""
        if torch.is_tensor(states):
            x = states.detach().to(dtype=torch.float32)
            out_device = states.device
        else:
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            out_device = self.device
        x = x.to(next(self.posterior.parameters()).device)
        with torch.no_grad():
            endpoints = self.head.sample(self._params(x))
        return endpoints.to(out_device)

    def get_manifold_component_names(self) -> List[str]:
        return list(self.head.component_names)

    def compute_manifold_distance_per_component(self, predicted, true) -> torch.Tensor:
        """Per-component geodesic error in NORMALIZED coordinates: [B, n_comp].

        MUST agree with the flow-matching family's identically-named method,
        ``flow_matching/base/flow_matcher.py:1156-1161``, which normalizes both
        arguments before taking the distance. Both families are called from the
        same unguarded site (``adaptive/endpoint_evaluation.py:116``) and the
        result is written to the SHARED ``artifacts_v2.json`` key
        ``endpoint_error``, which ``scripts/compile_adaptive_metrics.py:82-92``
        reads POSITIONALLY into one cross-run dataframe. The two families also
        return byte-identical ``component_names``, so a name-based join succeeds
        silently and nothing downstream can detect a unit mismatch.

        Computing this on raw coordinates instead inflated pendulum's
        ``angular_velocity`` component by exactly the velocity bound, 6.28x,
        against the FM arms it is tabulated beside.
        """
        return self.head.distance_per_component(
            self.system.normalize_state(predicted), self.system.normalize_state(true)
        )
