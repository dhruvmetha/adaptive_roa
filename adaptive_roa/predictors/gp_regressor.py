"""Multi-output sparse variational GP over embedded endpoints.

One independent SVGP per output task, sharing an inducing-point set, wrapped in
gpytorch's IndependentMultitaskVariationalStrategy. Trained with a minibatched
ELBO because the adaptive pool reaches 12k-17k rows on the quadrotors and the
existing single-output GP's full-batch loop does not scale across tasks.

Sampling goes through the LIKELIHOOD, not the latent function: the spec requires
the arm's predictive spread to include the observation-noise floor, or the GP
gets an aleatoric-free advantage over the Bayesian NN arms.

Sampling is also CHUNKED; see ``_predictive_chunk`` for why that is a
correctness requirement rather than a memory optimization.
"""
from __future__ import annotations

import math

import gpytorch
import torch

from adaptive_roa.predictors.gp_inducing import (
    append_inducing_points,
    copy_hyperparameters,
    embed_whitened_variational,
    hyperparameter_state,
)


class MultitaskSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, num_tasks: int, kernel: str = "matern52"):
        # inducing_points: [num_tasks, n_inducing, input_dim]
        batch_shape = torch.Size([num_tasks])
        vd = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape
        )
        base = gpytorch.variational.VariationalStrategy(
            self, inducing_points, vd, learn_inducing_locations=True
        )
        super().__init__(
            gpytorch.variational.IndependentMultitaskVariationalStrategy(base, num_tasks=num_tasks)
        )
        d = inducing_points.size(-1)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        base_kernel = (
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d, batch_shape=batch_shape)
            if kernel == "matern52"
            else gpytorch.kernels.RBFKernel(ard_num_dims=d, batch_shape=batch_shape)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class GPRegressor:
    """Fit/sample wrapper with its own state_dict schema."""

    def __init__(self, num_tasks: int, input_dim: int, n_inducing: int = 128,
                 kernel: str = "matern52", n_iters: int = 300, lr: float = 0.01,
                 batch_size: int = 1024, eval_every: int = 10, device: str = "cpu"):
        self.num_tasks = int(num_tasks)
        # Recorded at construction, then RECONCILED with reality: `fit` checks it
        # against the feature width it is handed, and `load_state_dict` refreshes
        # it (and `n_inducing`) from the checkpoint's inducing_shape. The export
        # wrapper legitimately constructs with a placeholder input_dim=1 and then
        # loads, so a constructor-time check is not possible.
        self.input_dim = int(input_dim)
        self.n_inducing = int(n_inducing)
        self.kernel = kernel
        self.n_iters = int(n_iters)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.eval_every = max(int(eval_every), 1)
        self.device = device
        self.model = None
        self.likelihood = None
        self.last_batches_per_iter = 0
        self.best_val_nll = None

    def _build(self, inducing: torch.Tensor):
        self.model = MultitaskSVGP(inducing, self.num_tasks, self.kernel).to(self.device)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)
        self.input_dim = int(inducing.size(-1))

    @property
    def _base_strategy(self):
        return self.model.variational_strategy.base_variational_strategy

    def inducing_count(self) -> int:
        """How many inducing points the live model actually has."""
        if self.model is None:
            return 0
        return int(self._base_strategy.inducing_points.size(-2))

    def _grow_inducing(self, X: torch.Tensor, target: int) -> int:
        """Top up a warm-started model's inducing set to ``target`` points.

        Rebuilding from scratch would throw away the learned posterior every
        epoch; skipping the rebuild (the previous behaviour) caps capacity at
        epoch 0's dataset size forever. This does neither: it keeps the kernel
        hyperparameters, the likelihood, every existing inducing LOCATION, and
        (losslessly, see ``embed_whitened_variational``) the learned variational
        posterior, and appends new inducing points drawn from the grown pool.
        """
        n_new = int(target) - self.inducing_count()
        if n_new <= 0:
            return 0

        strat = self._base_strategy
        old_z = strat.inducing_points.detach().clone()
        vd = strat._variational_distribution
        old_mean = vd.variational_mean.detach().clone()
        old_chol = vd.chol_variational_covar.detach().clone()
        # Everything except the inducing points and the variational parameters,
        # i.e. the learned kernel/mean hyperparameters.
        hyper = hyperparameter_state(self.model)
        lik = {k: v.detach().clone() for k, v in self.likelihood.state_dict().items()}

        self._build(append_inducing_points(old_z, X, n_new))
        copy_hyperparameters(self.model, hyper)
        self.likelihood.load_state_dict(lik)

        strat = self._base_strategy
        vd = strat._variational_distribution
        mean, chol = embed_whitened_variational(old_mean, old_chol, n_new)
        with torch.no_grad():
            vd.variational_mean.copy_(mean.to(vd.variational_mean))
            vd.chol_variational_covar.copy_(chol.to(vd.chol_variational_covar))
        # Suppress gpytorch's lazy prior re-initialization on the next forward,
        # which would overwrite the block we just wrote.
        strat.variational_params_initialized.fill_(1)
        return n_new

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            X_val: torch.Tensor | None = None, Y_val: torch.Tensor | None = None) -> "GPRegressor":
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.as_tensor(Y, dtype=torch.float32).to(self.device)
        n = X.size(0)

        # Checked BEFORE the build so it covers the cold path too, where
        # `_build` would otherwise silently overwrite `input_dim` from the data
        # and make the constructor argument unfalsifiable.
        if X.size(-1) != self.input_dim:
            raise ValueError(
                f"GPRegressor was built for input_dim={self.input_dim} but was handed "
                f"features of width {X.size(-1)}. A silent mismatch here would train "
                f"the kernel's ARD lengthscales against the wrong coordinates."
            )
        if Y.size(-1) != self.num_tasks:
            raise ValueError(
                f"GPRegressor has num_tasks={self.num_tasks} but targets are "
                f"{Y.size(-1)}-dimensional."
            )

        if self.model is None or self.likelihood is None:
            n_ind = min(self.n_inducing, n)
            perm = torch.randperm(n)[:n_ind]
            inducing = X[perm].clone().unsqueeze(0).repeat(self.num_tasks, 1, 1)
            self._build(inducing)
        else:
            # Warm start -- keep the loaded variational distribution, kernel
            # hyperparameters and inducing locations, and continue optimizing
            # them against the (now larger) training set. num_data below is
            # recomputed from the current n, so the ELBO scaling stays correct.
            # The pool grows every adaptive epoch, so top the inducing set up to
            # the configured budget instead of leaving capacity pinned to epoch
            # 0's dataset size.
            added = self._grow_inducing(X, min(self.n_inducing, n))
            if added:
                print(f"Warm start: grew inducing set by {added} to "
                      f"{self.inducing_count()} (pool n={n})")

        self.model.train()
        self.likelihood.train()
        opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()), lr=self.lr
        )
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=n)

        use_val = X_val is not None and Y_val is not None and len(X_val) > 0
        best_state, self.best_val_nll = None, None

        bs = min(self.batch_size, n)
        self.last_batches_per_iter = (n + bs - 1) // bs
        for it in range(self.n_iters):
            order = torch.randperm(n, device=X.device)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                opt.zero_grad()
                loss = -mll(self.model(X[idx]), Y[idx])
                loss.backward()
                opt.step()
            # Model selection on the held-out predictive NLL, matching the
            # criterion the BNN final-state arms early-stop on (val_nll), so all
            # of the final-state arms are selected the same way. Also evaluated
            # on the LAST iteration so the terminal state is always a candidate.
            if use_val and ((it + 1) % self.eval_every == 0 or it == self.n_iters - 1):
                val_nll = self.val_nll(X_val, Y_val)
                if self.best_val_nll is None or val_nll < self.best_val_nll:
                    self.best_val_nll = val_nll
                    best_state = {
                        "model": {k: v.detach().clone()
                                  for k, v in self.model.state_dict().items()},
                        "likelihood": {k: v.detach().clone()
                                       for k, v in self.likelihood.state_dict().items()},
                    }
                self.model.train()
                self.likelihood.train()

        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            self.likelihood.load_state_dict(best_state["likelihood"])
            print(f"GP regressor: restored best-val_nll state (val_nll={self.best_val_nll:.6g})")
        self.eval()
        return self

    def val_nll(self, X_val: torch.Tensor, Y_val: torch.Tensor) -> float:
        """Mean per-point, per-task MARGINAL predictive NLL on held-out data.

        Deliberately the diagonal predictive rather than the joint ``log_prob``:
        the joint's log-determinant goes through stochastic Lanczos quadrature
        above ``max_cholesky_size`` and would make the selection criterion noisy
        (and batch-size dependent) exactly the way ``sample`` was.
        """
        X_val = torch.as_tensor(X_val, dtype=torch.float32).to(self.device)
        Y_val = torch.as_tensor(Y_val, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mu = self.mean(X_val)
            var = self.predictive_variance(X_val).clamp_min(1e-12)
            nll = 0.5 * (torch.log(2.0 * math.pi * var) + (Y_val - mu).pow(2) / var)
        return float(nll.mean())

    def _predictive(self, X: torch.Tensor):
        self.model.eval()
        self.likelihood.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.likelihood(self.model(X))

    def _chunk_rows(self) -> int:
        """Query rows per predictive call, so ``rows * num_tasks`` stays within
        gpytorch's exact-Cholesky regime.

        gpytorch swaps a joint MVN's exact Cholesky root for a rank-100
        truncated Lanczos root decomposition once the JOINT dimension exceeds
        ``max_cholesky_size`` (default 800). For a multitask model the joint
        dimension is ``rows * num_tasks``, so the budget MUST be divided by
        num_tasks or quadrotor3d (13 tasks) still trips it at 62 rows.

        This is a correctness bound, not a memory one: past the threshold the
        low-rank root drastically under-disperses the draws (measured empirical
        variance ~4-8% of the analytic predictive variance at production batch
        sizes), which would have silently removed the arm's aleatoric noise
        floor, made p_success a function of how the caller batched its queries,
        and biased endpoint_error low against the sibling arms. Raising
        ``max_cholesky_size`` instead is not an option -- an exact Cholesky of a
        26624-square joint (quadrotor3d at batch 2048) is not affordable.
        """
        budget = int(gpytorch.settings.max_cholesky_size.value())
        return max(1, budget // max(int(self.num_tasks), 1))

    def _chunked(self, X: torch.Tensor, fn, cat_dim: int = 0) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        n = X.size(0)
        step = self._chunk_rows()
        if n <= step:
            return fn(X)
        return torch.cat([fn(X[i:i + step]) for i in range(0, n, step)], dim=cat_dim)

    def sample(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Predictive draws WITH observation noise: [num_samples, N, num_tasks].

        Chunked over query rows (see ``_chunk_rows``). Chunking makes the draws
        independent ACROSS chunks, which is exactly right for every consumer
        here: each endpoint is classified on its own, so only the per-point
        marginals matter, and those are what chunking makes correct.
        """
        num_samples = int(num_samples)
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if X.size(0) == 0:
            return torch.zeros(num_samples, 0, self.num_tasks, device=X.device)
        with torch.no_grad():
            return self._chunked(
                X, lambda x: self._predictive(x).sample(torch.Size([num_samples])), cat_dim=1
            )

    def mean(self, X: torch.Tensor) -> torch.Tensor:
        return self._chunked(X, lambda x: self._predictive(x).mean)

    def predictive_variance(self, X: torch.Tensor) -> torch.Tensor:
        return self._chunked(X, lambda x: self._predictive(x).variance)

    def latent_variance(self, X: torch.Tensor) -> torch.Tensor:
        """Latent-function variance, EXCLUDING observation noise. Diagnostic only --
        never sample from this path; see the module docstring."""
        def one(x):
            self.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                return self.model(x).variance
        return self._chunked(X, one)

    def eval(self):
        if self.model is not None:
            self.model.eval()
            self.likelihood.eval()
        return self

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device)
            self.likelihood.to(device)
        return self

    def state_dict(self):
        inducing = self._base_strategy.inducing_points
        return {
            "model": self.model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
            "inducing_shape": tuple(inducing.shape),
            "num_tasks": self.num_tasks,
            "kernel": self.kernel,
        }

    def load_state_dict(self, sd):
        if self.model is None:
            self.num_tasks = int(sd["num_tasks"])
            self.kernel = sd["kernel"]
            self._build(torch.zeros(sd["inducing_shape"]))
        self.model.load_state_dict(sd["model"])
        self.likelihood.load_state_dict(sd["likelihood"])
        # The checkpoint is authoritative about the architecture it carries; the
        # export wrapper constructs with a placeholder input_dim and relies on
        # this to stop `input_dim` misreporting the loaded model. `n_inducing`
        # is deliberately NOT overwritten: it is the CONFIGURED target that
        # `fit` grows towards, not a property of the checkpoint. Use
        # `inducing_count()` for what the live model actually has.
        self.input_dim = int(tuple(sd["inducing_shape"])[-1])
        self.eval()
        return self
