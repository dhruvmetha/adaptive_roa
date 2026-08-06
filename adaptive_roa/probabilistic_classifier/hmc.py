"""Export wrappers for the two HMC reference arms.

The checkpoint carries its own architecture metadata (hidden_dims, activation,
input/output dims), so the net is rebuilt from the checkpoint rather than from
the run config -- there is no path where a config edit silently produces a
wrong-shaped model.

That self-description removes the failure mode the sibling wrappers guard
against with an explicit ``load_state_dict(strict=False)`` key check (a
changed ``n_members`` or the wrong ``posterior_kind`` leaving whole
sub-networks at random init): here the skeleton is built from the very same
dict the flat sample vectors were written into, so architecture drift between
"what was trained" and "what gets rebuilt" cannot happen. What CAN still
happen is a checkpoint whose ``samples`` column count silently disagrees with
that freshly-built skeleton's own parameter count -- e.g. a hand-edited or
truncated checkpoint, or a ``build_bayesian_mlp`` change between when the file
was written and when it is loaded. ``HMCPosterior._forward_with`` would slice
``theta`` against each parameter in turn and only fail once it runs out of
elements (or silently ignores a surplus), which surfaces far from the actual
cause. ``_load_hmc`` checks the flat dimension up front and raises loudly
instead, matching the sibling wrappers' "never load a mismatched skeleton
silently" contract in the terms this checkpoint format actually has.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.final_state_handle import FinalStateModelHandle
from adaptive_roa.predictors.handles import OutcomeModelHandle
from adaptive_roa.predictors.heads import FinalStateHead
from adaptive_roa.predictors.hmc.posterior import HMCPosterior
from .base import ProbabilisticClassifier
from .endpoint_mc import endpoint_mc_probabilities
from .registry import register_probabilistic_classifier


def _load_hmc(run_dir, epoch, device):
    path = Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints" / "best-hmc.ckpt"
    if not path.exists():
        raise FileNotFoundError(f"no HMC checkpoint at {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_bayesian_mlp(
        input_dim=int(ck["input_dim"]), hidden_dims=list(ck["hidden_dims"]),
        output_dim=int(ck["output_dim"]), posterior="deterministic",
        activation=str(ck["activation"]),
    )
    # Same skeleton-vs-payload check the sibling wrappers make via
    # load_state_dict's missing/unexpected keys, expressed for a flat sample
    # vector instead of a named state dict: the draw dimension must equal the
    # rebuilt net's own trainable-parameter count exactly.
    expected_dim = sum(p.numel() for p in net.parameters() if p.requires_grad)
    samples = ck["samples"]
    actual_dim = int(samples.shape[-1]) if samples.ndim else -1
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"checkpoint {path} carries {actual_dim}-dim samples but the net "
            f"rebuilt from its own recorded metadata (hidden_dims="
            f"{list(ck['hidden_dims'])}, activation={ck['activation']!r}, "
            f"input_dim={ck['input_dim']}, output_dim={ck['output_dim']}) has "
            f"{expected_dim} trainable parameters. Loading it would inject "
            f"samples into a mismatched skeleton."
        )
    return HMCPosterior(net, samples).eval().to(device), ck


@register_probabilistic_classifier
class HMCProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "classifier"
    predictor_name = "hmc"
    native_probs = ("p_success",)

    def __init__(self, handle, system, device):
        self.handle = handle
        self.system = system
        self.device = device

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        out = []
        with torch.no_grad():
            for i in range(0, len(states), 8192):
                x = torch.as_tensor(states[i:i + 8192], dtype=torch.float32,
                                    device=self.device)
                out.append(torch.sigmoid(self.handle(x).view(-1)).double().cpu().numpy())
        p = np.concatenate(out) if out else np.zeros(0)
        return OutcomeProbabilities(p_success=p, p_failure=1.0 - p,
                                    p_invalid=np.zeros_like(p))

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        posterior, ck = _load_hmc(run_dir, epoch, device)
        if ck["head"] != "outcome":
            raise ValueError(f"checkpoint head is {ck['head']!r}, expected 'outcome'")
        handle = OutcomeModelHandle(posterior, system).eval().to(device)
        return cls(handle, system, device)


@register_probabilistic_classifier
class HMCRegProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "generative"
    predictor_name = "hmc_reg"
    native_probs = ("p_success", "p_failure", "p_invalid")

    def __init__(self, handle, system, device, attractor_radius, num_mc_samples):
        self.handle = handle
        self.system = system
        self.device = device
        self.attractor_radius = attractor_radius
        self.num_mc_samples = num_mc_samples

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        return endpoint_mc_probabilities(
            self.handle, self.system, states,
            attractor_radius=self.attractor_radius,
            num_mc_samples=self.num_mc_samples,
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        prob = cfg.get("probability", {})
        if "attractor_radius" not in prob:
            raise KeyError(
                f"run config at {run_dir} has no probability.attractor_radius; "
                "refusing to guess a radius, which would relabel every endpoint."
            )
        posterior, ck = _load_hmc(run_dir, epoch, device)
        if ck["head"] != "final_state":
            raise ValueError(f"checkpoint head is {ck['head']!r}, expected 'final_state'")
        head = FinalStateHead(system, beta=0.0)
        handle = FinalStateModelHandle(posterior, head, system).eval().to(device)
        return cls(handle, system, device,
                   attractor_radius=float(prob["attractor_radius"]),
                   num_mc_samples=int(prob.get("num_mc_samples", 10)))
