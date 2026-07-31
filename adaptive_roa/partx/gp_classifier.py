from __future__ import annotations

import gpytorch
import numpy as np
import torch

from adaptive_roa.predictors.gp_inducing import (
    append_inducing_points,
    copy_hyperparameters,
    embed_whitened_variational,
    hyperparameter_state,
)


class _VarGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, kernel: str):
        vd = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        vs = gpytorch.variational.VariationalStrategy(
            self, inducing_points, vd, learn_inducing_locations=True
        )
        super().__init__(vs)
        d = inducing_points.size(-1)
        self.mean_module = gpytorch.means.ConstantMean()
        base = (
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
            if kernel == "matern52"
            else gpytorch.kernels.RBFKernel(ard_num_dims=d)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class GPClassifier:
    """Variational sparse GP classifier. Latent f is the robustness."""

    def __init__(self, system, n_inducing=128, kernel="matern52",
                 n_iters=300, lr=0.1, device="cpu"):
        self.system = system
        self.n_inducing = int(n_inducing)
        self.kernel = kernel
        self.n_iters = int(n_iters)
        self.lr = float(lr)
        self.device = device
        self.model = None
        self.likelihood = None

    def _features(self, X_raw: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(X_raw), dtype=torch.float32)
        feats = self.system.embed_state_for_model(self.system.normalize_state(x))
        return feats.to(self.device)

    def inducing_count(self) -> int:
        """How many inducing points the live model actually has."""
        if self.model is None:
            return 0
        return int(self.model.variational_strategy.inducing_points.size(-2))

    def _grow_inducing(self, X: torch.Tensor, target: int) -> int:
        """Top up a warm-started model's inducing set to ``target`` points.

        Under ``warm_start: true`` the previous code skipped the rebuild
        entirely, so epoch 0's dataset size capped capacity for the whole run:
        ``configs/adaptive_v2/experiment/partx_pendulum.yaml`` starts at
        ``initial_train_size: 50`` and grows to 500, leaving a GP configured for
        128 inducing points locked at 50 while its pool grew tenfold. Rebuilding
        from scratch instead would discard the learned posterior every epoch.
        This keeps the kernel hyperparameters, every existing inducing LOCATION
        and (losslessly, see ``embed_whitened_variational``) the learned
        variational posterior, and appends new points drawn from the grown pool.
        """
        n_new = int(target) - self.inducing_count()
        if n_new <= 0:
            return 0

        strat = self.model.variational_strategy
        old_z = strat.inducing_points.detach().clone()
        vd = strat._variational_distribution
        old_mean = vd.variational_mean.detach().clone()
        old_chol = vd.chol_variational_covar.detach().clone()
        hyper = hyperparameter_state(self.model)

        self.model = _VarGP(
            append_inducing_points(old_z, X, n_new), self.kernel
        ).to(self.device)
        copy_hyperparameters(self.model, hyper)

        strat = self.model.variational_strategy
        vd = strat._variational_distribution
        mean, chol = embed_whitened_variational(old_mean, old_chol, n_new)
        with torch.no_grad():
            vd.variational_mean.copy_(mean.to(vd.variational_mean))
            vd.chol_variational_covar.copy_(chol.to(vd.chol_variational_covar))
        # Suppress gpytorch's lazy prior re-initialization on the next forward,
        # which would overwrite the block just written.
        strat.variational_params_initialized.fill_(1)
        return n_new

    def fit(self, X_raw: np.ndarray, y01: np.ndarray) -> "GPClassifier":
        X = self._features(X_raw)
        y = torch.as_tensor(np.asarray(y01), dtype=torch.float32).to(self.device)
        n_ind = min(self.n_inducing, X.size(0))
        if self.model is None or self.likelihood is None:
            # Inducing points: random subset of the training features
            # (deterministic under seed).
            perm = torch.randperm(X.size(0))[:n_ind]
            self.model = _VarGP(X[perm].clone(), self.kernel).to(self.device)
            self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
        else:
            # Warm start -- keep the loaded variational distribution, kernel
            # hyperparameters and inducing locations, and continue optimizing
            # them against the (now larger) training set. num_data below is
            # recomputed from the current y, so the ELBO scaling stays correct.
            # The pool grows every adaptive epoch, so top the inducing set up to
            # the configured budget rather than leaving capacity pinned to
            # epoch 0's dataset size.
            added = self._grow_inducing(X, n_ind)
            if added:
                print(f"Warm start: grew inducing set by {added} to "
                      f"{self.inducing_count()} (pool n={X.size(0)})")
        self.model.train(); self.likelihood.train()
        opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()), lr=self.lr
        )
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y.size(0))
        for _ in range(self.n_iters):
            opt.zero_grad()
            out = self.model(X)
            loss = -mll(out, y)
            loss.backward()
            opt.step()
        self.eval()
        return self

    def _latent(self, X_raw):
        self.model.eval(); self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f = self.model(self._features(X_raw))
            return f.mean.cpu().numpy(), f.variance.cpu().numpy()

    def latent_posterior(self, X_raw):
        return self._latent(X_raw)

    def p_success(self, X_raw):
        m, s2 = self._latent(X_raw)
        # Integrated probit predictive probability.
        from scipy.stats import norm
        return norm.cdf(m / np.sqrt(1.0 + s2))

    def eval(self):
        if self.model is not None:
            self.model.eval(); self.likelihood.eval()
        return self

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device); self.likelihood.to(device)
        return self

    def state_dict(self):
        inducing = self.model.variational_strategy.inducing_points
        return {
            "model": self.model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
            "inducing_shape": tuple(inducing.shape),
            "kernel": self.kernel,
        }

    def load_state_dict(self, sd):
        if self.model is None or self.likelihood is None:
            inducing = torch.zeros(sd["inducing_shape"])
            self.model = _VarGP(inducing, sd["kernel"]).to(self.device)
            self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
        # Sync the attribute with what was actually built. Without this, loading
        # a matern52 checkpoint into GPClassifier(kernel="rbf") yields a Matern
        # module that reports "rbf", and since both kernels expose identical
        # state-dict keys the NEXT epoch's load succeeds silently against a
        # genuinely different kernel. GPRegressor.load_state_dict does this too.
        self.kernel = sd["kernel"]
        self.model.load_state_dict(sd["model"])
        self.likelihood.load_state_dict(sd["likelihood"])
        self.eval()
        return self
