from __future__ import annotations

import gpytorch
import numpy as np
import torch


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

    def fit(self, X_raw: np.ndarray, y01: np.ndarray) -> "GPClassifier":
        X = self._features(X_raw)
        y = torch.as_tensor(np.asarray(y01), dtype=torch.float32).to(self.device)
        n_ind = min(self.n_inducing, X.size(0))
        # Inducing points: random subset of the training features (deterministic under seed).
        perm = torch.randperm(X.size(0))[:n_ind]
        inducing = X[perm].clone()
        if self.model is None or self.likelihood is None:
            self.model = _VarGP(inducing, self.kernel).to(self.device)
            self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
        # else: warm start -- keep the loaded variational distribution, kernel
        # hyperparameters and inducing locations, and continue optimizing them
        # against the (now larger) training set. num_data below is recomputed
        # from the current y, so the ELBO scaling stays correct.
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
        self.model.load_state_dict(sd["model"])
        self.likelihood.load_state_dict(sd["likelihood"])
        self.eval()
