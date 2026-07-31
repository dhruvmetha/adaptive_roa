"""Growing a sparse GP's inducing set across warm-started adaptive epochs.

Both GP arms size their inducing set as ``min(n_inducing, n)`` on the COLD path
only. Under ``warm_start: true`` that means epoch 0's dataset permanently caps
model capacity: ``configs/adaptive_v2/experiment/partx_pendulum.yaml`` starts at
``initial_train_size: 50`` and grows to 500 over 10 epochs, so a GP configured
for 128 inducing points would stay locked at 50 for the whole run while its pool
grew tenfold. Nothing downstream reports capacity, so the effect is a quietly
under-fit model rather than a failure.

The helpers here let a warm start TOP UP the inducing set instead of either
skipping the rebuild (capacity cap) or rebuilding from scratch (throwing away
the learned posterior every epoch).
"""
from __future__ import annotations

import torch


def hyperparameter_state(model: torch.nn.Module) -> dict:
    """The learned kernel/mean parameters, i.e. everything that is NOT the
    inducing points or the variational distribution."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if k.startswith("mean_module.") or k.startswith("covar_module.")}


def copy_hyperparameters(model: torch.nn.Module, hyper: dict) -> None:
    """Write ``hyper`` back into ``model`` in place.

    Deliberately not ``load_state_dict(..., strict=False)``: gpytorch's
    ``VariationalStrategy`` installs a load hook that indexes
    ``state_dict[keys[0]]``, so a partial dict with no entries under the strategy
    prefix raises IndexError.
    """
    dst = dict(model.named_parameters())
    dst.update(dict(model.named_buffers()))
    with torch.no_grad():
        for k, v in hyper.items():
            dst[k].copy_(v)


def append_inducing_points(old_z: torch.Tensor, X: torch.Tensor, n_new: int) -> torch.Tensor:
    """Append ``n_new`` rows drawn at random from ``X`` to an inducing set.

    Handles both layouts in this codebase: ``[m, d]`` (single-output
    ``GPClassifier``) and ``[num_tasks, m, d]`` (multitask ``GPRegressor``, whose
    tasks share one inducing set replicated along the batch dim).
    """
    if n_new <= 0:
        return old_z
    perm = torch.randperm(X.size(0), device=X.device)[:n_new]
    new_z = X[perm].detach().clone().to(dtype=old_z.dtype, device=old_z.device)
    if old_z.dim() == 3:
        new_z = new_z.unsqueeze(0).expand(old_z.size(0), -1, -1)
    return torch.cat([old_z, new_z], dim=-2)


def embed_whitened_variational(
    old_mean: torch.Tensor, old_chol: torch.Tensor, n_new: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extend a whitened variational posterior to a larger inducing set, LOSSLESSLY.

    gpytorch's ``VariationalStrategy`` is WHITENED: the learned parameters are
    ``(m, S)`` of ``q(eps) = N(m, S)`` with ``u = L eps`` and ``L = chol(K_zz)``.
    Its ``prior_distribution`` is therefore the standard normal ``N(0, I)`` (see
    ``variational_strategy.py``), which is what a freshly built model is
    initialized to on its first forward pass.

    Cholesky factorization is incremental: for
    ``K_full = [[A, B], [B^T, C]]`` the factor is
    ``L_full = [[chol(A), 0], [B^T chol(A)^-T, chol(C - B^T A^-1 B)]]``, i.e. the
    leading block of the enlarged ``L`` is EXACTLY the old ``L``. So writing the
    old ``(m, S)`` into the leading block and giving the appended points the
    whitened prior ``N(0, I)`` (with zero cross-covariance) leaves the marginal
    posterior over the ORIGINAL inducing values bit-for-bit unchanged, and gives
    the new inducing values exactly the prior conditional ``p(u_new | u_old)``.
    Nothing learned is discarded and nothing is invented.

    ``old_mean``  is ``[..., m]``   -> returns ``[..., m + n_new]``
    ``old_chol``  is ``[..., m, m]`` -> returns ``[..., m + n_new, m + n_new]``
    """
    if n_new <= 0:
        return old_mean, old_chol
    m = int(old_mean.size(-1))
    total = m + int(n_new)

    mean = old_mean.new_zeros(*old_mean.shape[:-1], total)
    mean[..., :m] = old_mean

    chol = old_chol.new_zeros(*old_chol.shape[:-2], total, total)
    chol[..., :m, :m] = old_chol
    eye = torch.eye(int(n_new), dtype=chol.dtype, device=chol.device)
    chol[..., m:, m:] = eye
    return mean, chol
