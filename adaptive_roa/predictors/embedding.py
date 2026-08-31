"""Inverse of ``system.embed_state_for_model``.

A Gaussian-process regressor cannot learn an angle directly: theta = +/-pi is a
seam, and the pendulum's failure attractors sit exactly on it. Nor can it learn a
quaternion directly, since -q is the same rotation as q. So the GP regresses to
the EMBEDDED representation -- (sin, cos) per angle, the 4-vector per rotation --
and this decoder maps a sample back to a raw state.

``extract_circular_state`` in flow_matching/utils is pendulum-specific (a
hardcoded 3 -> 2 mapping) and has no call sites; this is the generic form.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from adaptive_roa.predictors.manifold_likelihood import (
    canonicalize_quaternion,
    wrap_angle,
)

# Embedded width consumed per component, keyed by manifold type.
_EMBED_WIDTH = {"Real": lambda dim: int(dim), "SO2": lambda dim: 2, "SO3": lambda dim: 4}


class EmbeddedStateDecoder:
    """Maps ``embed_state_for_model(normalize_state(x))`` back to raw ``x``."""

    def __init__(self, system):
        self.system = system
        self._parts: List[Tuple[str, int, int, int, int]] = []

        embed_offset = 0
        state_offset = 0
        for comp in system.manifold_components:
            width_fn = _EMBED_WIDTH.get(comp.manifold_type)
            if width_fn is None:
                raise ValueError(
                    f"no embedding decoder for manifold component "
                    f"{comp.manifold_type!r}; expected one of {sorted(_EMBED_WIDTH)}"
                )
            width = width_fn(comp.dim)
            self._parts.append((comp.manifold_type, embed_offset, width, state_offset, comp.dim))
            embed_offset += width
            state_offset += comp.dim

        self.embed_dim = embed_offset
        self.state_dim = state_offset

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """[B, embed_dim] -> [B, state_dim] in RAW coordinates."""
        out = []
        for kind, e0, width, _s0, dim in self._parts:
            chunk = embedded[..., e0:e0 + width]
            if kind == "Real":
                out.append(chunk)
            elif kind == "SO2":
                # atan2 over an unnormalized direction: magnitude is irrelevant and
                # there is no seam, which is the point of regressing in this space.
                out.append(wrap_angle(torch.atan2(chunk[..., 0:1], chunk[..., 1:2])))
            else:  # SO3
                out.append(canonicalize_quaternion(chunk))
        normalized = torch.cat(out, dim=-1)
        return self.system.denormalize_state(normalized)
