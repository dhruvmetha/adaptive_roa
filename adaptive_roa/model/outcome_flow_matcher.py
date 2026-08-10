"""Scalar-outcome flow matching: the generative x binary-target cell.

The adaptive pipeline already has three of the four cells in the
(target) x (how the predictive distribution is formed) factorial:

    cross-entropy point estimate  x  outcome      -> ClassifierModule
    Gaussian NLL / Bayesian       x  final state  -> FinalStateHead + posterior
    flow matching                 x  final state  -> EndpointMCProbabilityBackend

Flow matching only ever appears in the *final-state* column, so the campaign's
headline comparison (endpoint-FM vs classifier, debiased Brier 6-43x apart yet
sAUROC identical to 3-4 dp) varies the target AND the machinery at once. This
module supplies the missing cell so the two factors can be separated.

Geometry. The flow space is R^1 with two sharp anchors, ``-1`` = failure and
``+1`` = success. Anchors at +-1 rather than {0,1} put the decision threshold at
0, which is the mean of the ``N(0,1)`` source -- so an uninformative model
returns p=0.5 rather than an arbitrary value, and the readout threshold does not
have to be calibrated separately.

Backbone. Deliberately the SAME MLP the classifier uses, over the same embedded
state, with the same hidden dims. That is what makes the contrast clean: against
the classifier this varies *only* the loss (velocity regression vs weighted
cross-entropy) on byte-identical data with identical capacity. Against endpoint
FM it varies only the target.

Known asymmetry, recorded rather than papered over: ``ClassifierModule`` counters
class imbalance with ``pos_weight``; velocity regression has no equivalent knob,
so on a heavily imbalanced system the two arms are not matched on that axis. The
pendulum runs this is built for are 39-48% success, where ``pos_weight`` ~ 1 and
the asymmetry is immaterial. It would NOT be immaterial on quad2d (~8% success).
"""
from __future__ import annotations

from typing import Any, List

import lightning.pytorch as pl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}

# Source-space integration bounds for the exact readout. N(0,1) mass outside
# +-5 is 5.7e-7, far below any Brier resolution we can measure, so truncating
# there costs nothing and bounds the bracket search.
_X0_LIMIT = 5.0


class _TimeEmbedding(nn.Module):
    """Fourier features for t. A bare scalar t makes the velocity net fit the
    time axis poorly near the anchors, where the field changes fastest."""

    def __init__(self, num_freqs: int = 4):
        super().__init__()
        self.num_freqs = int(num_freqs)

    @property
    def dim(self) -> int:
        return 1 + 2 * self.num_freqs

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        feats = [t]
        for k in range(1, self.num_freqs + 1):
            feats.append(torch.sin(math.pi * k * t))
            feats.append(torch.cos(math.pi * k * t))
        return torch.cat(feats, dim=-1)


class OutcomeVelocityMLP(nn.Module):
    """(embedded state, x_t, t) -> scalar velocity."""

    def __init__(
        self,
        condition_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        num_time_freqs: int = 4,
    ):
        super().__init__()
        act_cls = _ACTIVATIONS.get(str(activation).lower())
        if act_cls is None:
            raise ValueError(
                f"unknown activation {activation!r}; expected one of {sorted(_ACTIVATIONS)}"
            )
        self.time_embedding = _TimeEmbedding(num_time_freqs)
        input_dim = int(condition_dim) + 1 + self.time_embedding.dim

        dims = [input_dim] + [int(h) for h in hidden_dims]
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, condition: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """condition [B, C], x_t [B], t [B] -> velocity [B]."""
        feats = torch.cat([condition, x_t.view(-1, 1), self.time_embedding(t)], dim=-1)
        return self.net(feats).view(-1)


class OutcomeFlowMatcher(pl.LightningModule):
    """Conditional FM from N(0,1) to the anchor implied by the binary label.

    ``forward`` takes RAW states and embeds internally, matching
    ``ClassifierModule``'s contract so both arms share one embedding code path.
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        system: Any,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        num_ode_steps: int = 100,
        forward_readout: str = "exact",
        forward_num_samples: int = 100,
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.system = system  # plain attr; system methods are device-agnostic
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.num_ode_steps = int(num_ode_steps)
        if str(forward_readout).lower() not in {"mc", "exact"}:
            raise ValueError(f"forward_readout must be 'mc' or 'exact', got {forward_readout!r}")
        self.forward_readout = str(forward_readout).lower()
        self.forward_num_samples = int(forward_num_samples)

    # ---------------------------------------------------------------- embedding

    def embed(self, raw_states: torch.Tensor) -> torch.Tensor:
        normalized = self.system.normalize_state(raw_states)
        return self.system.embed_state_for_model(normalized)

    def velocity(self, raw_states: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Training-time call: velocity of the flow at (x_t, t) given the state."""
        return self.velocity_net(self.embed(raw_states), x_t, t)

    def forward(self, raw_states: torch.Tensor) -> torch.Tensor:
        """RAW states -> logit(p_success), matching ``ClassifierModule``'s contract.

        This is what makes the arm a drop-in for every discriminative code path
        -- threshold optimisation, conformal calibration, and
        ``ClassifierProbabilityEstimator`` all call ``model(x) -> logits`` and
        recover ``sigmoid(logit(p)) = p`` unchanged. Without it the classifier
        family tag would route calibration through a path this module cannot
        answer, and Route B would need engine surgery after all.

        Note this is NOT the training forward; training calls ``velocity``.
        """
        p = self.predict_p_success(raw_states)
        # logit is unbounded at 0/1 and the exact readout genuinely returns
        # values within 1e-7 of the tails, so clamp before the transform.
        p = p.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(p / (1.0 - p)).to(torch.float32).view(-1, 1)

    @torch.no_grad()
    def predict_p_success(self, raw_states: torch.Tensor) -> torch.Tensor:
        if self.forward_readout == "mc":
            return self.p_success_mc(
                raw_states, num_samples=self.forward_num_samples, num_steps=self.num_ode_steps
            )
        p, _ = self.p_success_exact(raw_states, num_steps=self.num_ode_steps)
        return p

    # ---------------------------------------------------------------- training

    @staticmethod
    def _anchor(label: torch.Tensor) -> torch.Tensor:
        """label in {0,1} (or {-1,+1}) -> anchor in {-1,+1}."""
        return torch.where(label > 0.5, 1.0, -1.0)

    def _step(self, batch, stage: str):
        raw = batch["inputs"]
        label = batch["label"].float().view(-1)

        x1 = self._anchor(label)
        x0 = torch.randn_like(x1)
        t = torch.rand_like(x1)
        x_t = (1.0 - t) * x0 + t * x1
        target_v = x1 - x0  # linear path => constant velocity along the segment

        pred_v = self.velocity(raw, x_t, t)
        loss = F.mse_loss(pred_v, target_v)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

    # ---------------------------------------------------------------- flow map

    @torch.no_grad()
    def flow_map(
        self,
        condition: torch.Tensor,
        x0: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Integrate dx/dt = v(x,t | condition) from t=0 to t=1.

        ``condition`` [N, C] embedded states, ``x0`` [N] source samples aligned
        with it. Midpoint (RK2) rather than Euler: the velocity field is steep
        near the anchors and Euler at a comparable step count visibly biases the
        crossing point, which is precisely what the exact readout measures.
        """
        steps = int(num_steps if num_steps is not None else self.num_ode_steps)
        dt = 1.0 / steps
        x = x0.clone()
        for i in range(steps):
            t0 = torch.full_like(x, i * dt)
            k1 = self.velocity_net(condition, x, t0)
            k2 = self.velocity_net(condition, x + 0.5 * dt * k1, t0 + 0.5 * dt)
            x = x + dt * k2
        return x

    # ---------------------------------------------------------------- readouts

    @torch.no_grad()
    def p_success_mc(
        self,
        raw_states: torch.Tensor,
        num_samples: int = 100,
        num_steps: int | None = None,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """K-sample MC fraction. Matches how endpoint-MC forms p, so a difference
        against endpoint FM is attributable to the target rather than the readout.
        Quantised to 1/K -- see ``p_success_exact`` for the unquantised estimate.
        """
        out = []
        for start in range(0, raw_states.shape[0], chunk_size):
            block = raw_states[start:start + chunk_size]
            cond = self.embed(block)                                  # [b, C]
            b = cond.shape[0]
            cond_rep = cond.repeat_interleave(num_samples, dim=0)     # [b*K, C]
            x0 = torch.randn(b * num_samples, device=cond.device, dtype=cond.dtype)
            x1 = self.flow_map(cond_rep, x0, num_steps).view(b, num_samples)
            out.append((x1 > 0).to(torch.float64).mean(dim=1))
        return torch.cat(out)

    @torch.no_grad()
    def p_success_exact(
        self,
        raw_states: torch.Tensor,
        num_steps: int | None = None,
        grid_size: int = 33,
        bisect_iters: int = 20,
        chunk_size: int = 4096,
        fallback_grid_size: int = 1025,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact success mass under the learned flow, with no MC noise.

        In 1D the ODE is a deterministic map Psi: x0 -> x1, so
        ``p = P(Psi(x0) > 0)``. When Psi is monotone this is ``1 - Phi(x0*)`` for
        the unique root x0*, found by bracketing on a coarse grid and bisecting.

        Monotonicity is a property of the *learned* field, not a guarantee, so it
        is checked rather than assumed: points whose grid shows more than one sign
        change fall back to grid quadrature over the same evaluations. Returns
        ``(p_success, monotone_mask)``; the caller is expected to report the
        non-monotone fraction rather than let it pass silently.
        """
        device = raw_states.device
        grid = torch.linspace(-_X0_LIMIT, _X0_LIMIT, int(grid_size), device=device)
        normal = torch.distributions.Normal(0.0, 1.0)

        p_out, mono_out = [], []
        for start in range(0, raw_states.shape[0], chunk_size):
            block = raw_states[start:start + chunk_size]
            cond = self.embed(block)                                  # [b, C]
            b, g = cond.shape[0], grid.shape[0]

            cond_rep = cond.repeat_interleave(g, dim=0)               # [b*g, C]
            x0_grid = grid.repeat(b).to(cond.dtype)
            psi = self.flow_map(cond_rep, x0_grid, num_steps).view(b, g)

            sign_pos = psi > 0
            changes = (sign_pos[:, 1:] != sign_pos[:, :-1]).sum(dim=1)
            monotone = changes <= 1

            # --- quadrature fallback, valid for any number of crossings -------
            weights = torch.exp(normal.log_prob(grid))
            p_quad = (sign_pos.to(torch.float64) * weights.to(torch.float64)).sum(dim=1) \
                / weights.to(torch.float64).sum()

            # --- bisection on the single bracketed crossing --------------------
            # Degenerate cases (no crossing) are handled by the p_quad fallback,
            # which returns ~0 or ~1 correctly when every grid point agrees.
            has_cross = changes == 1
            p_exact = p_quad.clone()
            if bool(has_cross.any()):
                idx = torch.nonzero(has_cross).view(-1)
                first = torch.argmax(
                    (sign_pos[idx, 1:] != sign_pos[idx, :-1]).to(torch.int8), dim=1
                )
                lo = grid[first]
                hi = grid[first + 1]
                cond_sub = cond[idx]
                for _ in range(int(bisect_iters)):
                    mid = 0.5 * (lo + hi)
                    val = self.flow_map(cond_sub, mid.to(cond.dtype), num_steps)
                    # Keep the half-interval that still straddles the crossing.
                    lo_pos = self._grid_value_positive(sign_pos[idx], first)
                    go_hi = (val > 0) == lo_pos
                    lo = torch.where(go_hi, mid, lo)
                    hi = torch.where(go_hi, hi, mid)
                root = 0.5 * (lo + hi)
                # Psi increasing at the crossing => success is the upper tail.
                upper_is_success = ~self._grid_value_positive(sign_pos[idx], first)
                tail = 1.0 - normal.cdf(root.double())
                p_exact[idx] = torch.where(upper_is_success, tail, 1.0 - tail)

            # Quadrature on the BRACKETING grid resolves x0 to 10/(g-1) ~ 0.31,
            # i.e. up to ~0.12 in p -- five orders coarser than the bisection
            # path's 1.9e-6, and far coarser than the effects being measured. A
            # fallback that silently degrades precision that much is worse than
            # no fallback, so non-monotone rows are re-integrated on a fine grid.
            # They are rare (0% observed on trained fields), so the cost is small.
            folded = ~monotone
            if bool(folded.any()) and fallback_grid_size > grid_size:
                idx = torch.nonzero(folded).view(-1)
                fine = torch.linspace(-_X0_LIMIT, _X0_LIMIT, int(fallback_grid_size),
                                      device=device)
                fw = torch.exp(normal.log_prob(fine)).to(torch.float64)
                sub = cond[idx]
                fine_p = []
                # Chunk over points: b*1025 rows at once can dwarf the main pass.
                step = max(1, chunk_size // int(fallback_grid_size))
                for s in range(0, sub.shape[0], step):
                    blk = sub[s:s + step]
                    rep = blk.repeat_interleave(fine.shape[0], dim=0)
                    x0f = fine.repeat(blk.shape[0]).to(cond.dtype)
                    psif = self.flow_map(rep, x0f, num_steps).view(blk.shape[0], -1)
                    fine_p.append(((psif > 0).to(torch.float64) * fw).sum(dim=1) / fw.sum())
                p_quad[idx] = torch.cat(fine_p)

            # Never claim more certainty than the integration domain supports.
            # With no crossing inside x0 in [-5, 5] the quadrature returns EXACTLY
            # 0 or 1, but all that was established is that any disagreeing mass
            # lies in the truncated tails -- Phi(-5) = 2.9e-7. Reporting a hard 0
            # or 1 makes log-score and KL infinite whenever the truth is interior,
            # which is exactly the situation at the noisy levels. Clamping to the
            # tail mass is the honest bound and is far below any effect measured
            # here, so it cannot manufacture a result either.
            tail = float(normal.cdf(torch.tensor(-_X0_LIMIT, dtype=torch.float64)))
            merged = torch.where(monotone, p_exact, p_quad).clamp(tail, 1.0 - tail)
            p_out.append(merged)
            mono_out.append(monotone)

        return torch.cat(p_out), torch.cat(mono_out)

    @staticmethod
    def _grid_value_positive(sign_pos: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
        """Sign of Psi at the LOW end of each row's bracketing interval."""
        return sign_pos.gather(1, first.view(-1, 1)).view(-1)

    # ------------------------------------------------------------- checkpoints

    @classmethod
    def from_checkpoint(cls, ckpt_path, system: Any, **kwargs) -> "OutcomeFlowMatcher":
        """Rebuild from a Lightning checkpoint, inferring the net shape from it.

        The constructor takes a `system` and a net that Lightning cannot
        reconstruct, so callers have to rebuild by hand. Doing that with
        hard-coded hidden dims plus `strict=False` is a silent-failure trap: if
        the arm's config ever changes width, every weight key mismatches, the
        load quietly does nothing, and the analysis proceeds on an untrained
        network that still returns plausible probabilities near 0.5.

        Shapes come from the checkpoint and the load is STRICT, so a mismatch
        raises instead of producing confident nonsense.
        """
        import re

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        pat = re.compile(r"^velocity_net\.net\.(\d+)\.weight$")
        layers = sorted(
            ((int(m.group(1)), state[k].shape) for k in state if (m := pat.match(k))),
            key=lambda t: t[0],
        )
        if not layers:
            raise ValueError(f"{ckpt_path}: no velocity_net weights found")

        # Linear shapes are (out, in); hidden dims are every layer's output but
        # the last, and the input width fixes the time-embedding frequency count.
        hidden = [int(s[0]) for _, s in layers[:-1]]
        in_dim = int(layers[0][1][1])
        cond_dim = int(system.embed_state_for_model(
            system.normalize_state(torch.zeros(1, int(system.state_dim)))
        ).shape[-1])
        # in_dim = cond_dim + 1 (x_t) + (1 + 2*num_freqs)
        num_freqs = (in_dim - cond_dim - 2) // 2
        if num_freqs < 0 or cond_dim + 2 + 2 * num_freqs != in_dim:
            raise ValueError(
                f"{ckpt_path}: input width {in_dim} is inconsistent with condition dim "
                f"{cond_dim}; the checkpoint was trained on a different system"
            )

        model = cls(
            velocity_net=OutcomeVelocityMLP(cond_dim, hidden, num_time_freqs=num_freqs),
            system=system,
            **kwargs,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if k.startswith("velocity_net.")]
        if real_missing or unexpected:
            raise RuntimeError(
                f"{ckpt_path}: state_dict mismatch (missing={real_missing}, unexpected={unexpected})"
            )
        model.eval()
        return model
