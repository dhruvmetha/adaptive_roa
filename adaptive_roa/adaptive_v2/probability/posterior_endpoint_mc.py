"""Endpoint-MC probabilities with the weight draw PINNED across the K samples.

This is the member-exposing backend the final-state (endpoint-regressing) BNN
arms need in order to acquire on BALD.

Why it cannot go through the model handle
-----------------------------------------
``FinalStateModelHandle.predict_endpoint`` draws a fresh weight sample on EVERY
call, by documented contract (``final_state_handle.py:48-52``); the spread
across K calls is exactly what makes the handle's marginal a posterior
predictive. That is right for ``estimate()`` and wrong for a decomposition: K
draws taken that way mix epistemic and aleatoric variation, so
``epistemic_bald`` over them scores sampling noise while looking perfectly
reasonable. There is no point inside the handle at which one weight sample is
visible across several head draws.

So this backend reaches past the handle to ``posterior`` and ``head`` directly,
draws the weights ONCE per member, and takes the K head samples under it.

Why it subclasses rather than replaces
--------------------------------------
``estimate()``, ``sample_endpoints()``, ``p_invalid`` and the whole calibration
and evaluation path are inherited untouched, so an arm that swaps
``endpoint_mc`` for this one changes nothing except that acquisition gains a
surface it did not have. Nothing downstream of the predictor moves, which is
what keeps a BALD arm comparable to its own uniform twin.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from adaptive_roa.adaptive_v2.probability.endpoint_mc import EndpointMCProbabilityBackend


class PosteriorEndpointMCProbabilityBackend(EndpointMCProbabilityBackend):
    """Per-member endpoint-MC probabilities from a weight posterior.

    ``estimate_members`` returns ``[S, N]``: one row per posterior draw (or per
    ensemble member), each entry a K-sample binomial estimate of p(success).
    ``member_sample_size`` reports K so ``epistemic_bald_debiased`` can remove
    the ~(1/2K)(1-1/M) upward bias that K carries.
    """

    def __init__(self, cfg: Any, system: Any, device: str):
        super().__init__(cfg, system, device)
        self.n_posterior_samples = int(cfg.get("n_posterior_samples", 64))
        if self.n_posterior_samples < 2:
            raise ValueError(
                f"n_posterior_samples={self.n_posterior_samples} gives no epistemic "
                "signal; BALD would score 0 everywhere. Use >= 2 (default 64)."
            )
        self.chunk_size = int(cfg.get("chunk_size", 4096))
        # Set to what the posterior actually returned on the last call, which is
        # S for a Gaussian and the member count for one that enumerates.
        self.n_members = 0

    @property
    def member_sample_size(self) -> int:
        """K. Each p_m is a K-sample binomial estimate, so it carries noise."""
        return self.num_mc_samples

    # ------------------------------------------------------------------ binding

    def bind_model(self, model_handle: Any) -> None:
        super().bind_model(model_handle)
        posterior = self._posterior()
        for attr in ("forward_all_members", "forward_samples"):
            if hasattr(posterior, attr):
                break
        else:
            raise TypeError(
                f"{type(posterior).__name__} exposes neither forward_all_members() nor "
                "forward_samples(); a BALD arm needs a posterior it can draw several "
                "predictive atoms from."
            )
        self._reject_posteriors_without_spread(posterior)

    @staticmethod
    def _reject_posteriors_without_spread(posterior: Any) -> None:
        """Refuse anything whose atoms are all the same point.

        Both cases below produce S identical rows, hence BALD = 0 everywhere,
        hence an arm that selects by an arbitrary tie-break while reporting an
        adaptive configuration. That failure is completely silent in the
        artifacts, so it is refused at bind time instead.
        """
        # Imported here rather than at module scope: adaptive_roa.predictors
        # pulls in torch model code that this module otherwise does not need.
        from adaptive_roa.predictors.posteriors import DeterministicPosterior

        if isinstance(posterior, DeterministicPosterior):
            raise ValueError(
                "a deterministic posterior has no epistemic spread, so every BALD "
                "score would be 0 and the arm would select by tie-break while "
                "reporting an adaptive config. Use an mfvi/ensemble/laplace arm."
            )
        n = getattr(posterior, "n_members", None)
        if n is not None and int(n) < 2:
            raise ValueError(
                f"ensemble posterior needs at least 2 members, got {int(n)}. A "
                "1-member 'ensemble' has no epistemic signal and would score 0."
            )

    def _posterior(self) -> Any:
        if self.model_handle is None:
            raise RuntimeError("probability backend used before bind_model")
        return getattr(self.model_handle, "posterior", self.model_handle)

    def _head(self) -> Any:
        head = getattr(self.model_handle, "head", None)
        if head is None:
            raise RuntimeError(
                f"{type(self.model_handle).__name__} has no `head`; this backend "
                "needs the predictive head to draw endpoints under a fixed weight "
                "sample."
            )
        return head

    def _device(self, posterior: Any) -> torch.device:
        try:
            return next(posterior.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device(self.device)

    # ------------------------------------------------------------------ members

    @torch.no_grad()
    def estimate_members(self, start_states: np.ndarray, verbose: bool = False) -> np.ndarray:
        """[N, D] raw states -> [S, N] per-member success probabilities."""
        posterior = self._posterior()
        head = self._head()
        device = self._device(posterior)

        states = torch.as_tensor(np.asarray(start_states), dtype=torch.float32, device=device)
        n_points = int(states.shape[0])
        if n_points == 0:
            return np.zeros((0, 0), dtype=np.float64)

        chunks: list[np.ndarray] = []
        for lo in range(0, n_points, self.chunk_size):
            chunks.append(self._members_for_chunk(posterior, head, states[lo:lo + self.chunk_size]))

        out = np.concatenate(chunks, axis=1)
        self.n_members = int(out.shape[0])
        if verbose:
            print(f"    [PosteriorEndpointMC] {n_points} states, M={out.shape[0]}, "
                  f"K={self.num_mc_samples}")
        return out

    def _members_for_chunk(self, posterior: Any, head: Any, states: torch.Tensor) -> np.ndarray:
        """One chunk of candidates: [B, D] -> [S, B].

        Chunking redraws weights per chunk for a CONTINUOUS posterior, so
        "member s" is a different weight vector in chunk 1 than in chunk 2. That
        is safe, and deliberately so: ``epistemic_bald`` is a per-point
        functional, H(mean_s p_s[n]) - mean_s H(p_s[n]), which needs only the S
        values AT A GIVEN POINT to come from S distinct posterior draws. It
        never compares member s at one point against member s at another. An
        enumerating posterior is consistent across chunks anyway, since
        ``forward_all_members`` is deterministic.
        """
        embedded = self.system.embed_state_for_model(self.system.normalize_state(states))

        if hasattr(posterior, "forward_all_members"):
            # Finite support: enumerate it. Sampling M atoms with replacement
            # instead gives a multinomial weight vector rather than an exact
            # 1/M, which is a systematic bias, not noise that averages out.
            params = posterior.forward_all_members(embedded)            # [M, B, P]
        else:
            params = posterior.forward_samples(embedded, self.n_posterior_samples)  # [S, B, P]

        n_members, batch = int(params.shape[0]), int(states.shape[0])
        out = np.empty((n_members, batch), dtype=np.float64)
        for m in range(n_members):
            member_params = params[m]        # pinned for the whole K loop below
            hits = torch.zeros(batch, dtype=torch.float64)
            for _ in range(self.num_mc_samples):
                endpoint = head.sample(member_params)
                label = self.system.classify_attractor(endpoint, self.attractor_radius)
                hits += (label == 1).double().cpu()
            out[m] = (hits / float(self.num_mc_samples)).numpy()
        return out
