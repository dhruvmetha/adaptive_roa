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

    def __init__(self, posterior, head, system: Any, device: str = "cpu",
                 reuse_body_features: bool = False):
        self.posterior = posterior
        self.head = head
        self.system = system
        self.device = device
        self.training = False
        self.distance_manifold = _ManifoldDistanceShim(head)
        # Opt-in (predictor.final_state.reuse_body_features). The MC loops call
        # predict_endpoint K times on the SAME batch tensor; for a deterministic
        # or last-layer Laplace posterior the body output is identical on every
        # one of those calls, so it is computed on the first and reused. Only the
        # head draws (weight noise, component pick, endpoint noise) run per call,
        # in the order they always ran, so the endpoints are bit-identical.
        self.reuse_body_features = bool(reuse_body_features)
        self._body_cache: tuple | None = None

    def eval(self):
        self.posterior.eval()
        self.training = False
        self._body_cache = None
        return self

    def train(self, mode: bool = True):
        self.posterior.train(mode)
        self.training = bool(mode)
        self._body_cache = None
        return self

    def to(self, device):
        self.device = device
        self.posterior.to(device)
        self._body_cache = None
        return self

    def _params(self, states: torch.Tensor) -> torch.Tensor:
        embedded = self.system.embed_state_for_model(self.system.normalize_state(states))
        return self.posterior.forward_sample(embedded)

    def _split_posterior(self):
        """``(body, head_draw)`` with ``head_draw(body(e)) == forward_sample(e)``, or None.

        Only for posteriors whose body is a fixed function in eval mode. Exact
        types, so a subclass that changes forward_sample is never split.
        """
        from adaptive_roa.predictors.posteriors import (
            DeterministicPosterior, LastLayerLaplacePosterior,
        )
        post = self.posterior
        if post.training:
            return None
        if type(post) is DeterministicPosterior:
            return post.net, (lambda feats: feats)
        if type(post) is LastLayerLaplacePosterior:
            return post.body, post.sample_from_features
        return None

    def _params_reusing_body(self, states, x: torch.Tensor):
        """``_params(x)``, with the body output reused while ``states`` is the cached batch.

        The key holds a reference to the caller's tensor (so its id cannot be
        recycled) plus its version counter, which every in-place write bumps,
        including writes through a view or to the base; and the version of every
        posterior parameter, so a weight update also invalidates the cache.
        """
        split = self._split_posterior()
        if split is None or not torch.is_tensor(states):
            return self._params(x)
        try:
            version = states._version
            versions = tuple(p._version for p in self.posterior.parameters())
        except RuntimeError:
            # Inference-mode tensors carry no version counter; never cache them.
            self._body_cache = None
            return self._params(x)
        body, head_draw = split
        c = self._body_cache
        if (c is not None and c[0] is states and c[1] == version
                and c[2] == versions and c[3] == x.device):
            feats = c[4]
        else:
            embedded = self.system.embed_state_for_model(self.system.normalize_state(x))
            feats = body(embedded)
            self._body_cache = (states, version, versions, x.device, feats)
        return head_draw(feats)

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
            if self.reuse_body_features:
                params = self._params_reusing_body(states, x)
            else:
                params = self._params(x)
            endpoints = self.head.sample(params)
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
