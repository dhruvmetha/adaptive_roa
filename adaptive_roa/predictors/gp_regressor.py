"""Multi-output sparse variational GP over embedded endpoints.

One independent SVGP per output task, sharing an inducing-point set, wrapped in
gpytorch's IndependentMultitaskVariationalStrategy. Trained with a minibatched
ELBO because the adaptive pool reaches 12k-17k rows on the quadrotors and the
existing single-output GP's full-batch loop does not scale across tasks.

Sampling goes through the LIKELIHOOD, not the latent function: the spec requires
the arm's predictive spread to include the observation-noise floor, or the GP
gets an aleatoric-free advantage over the Bayesian NN arms.
"""
from __future__ import annotations

import gpytorch
import torch


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
                 batch_size: int = 1024, device: str = "cpu"):
        self.num_tasks = int(num_tasks)
        self.input_dim = int(input_dim)
        self.n_inducing = int(n_inducing)
        self.kernel = kernel
        self.n_iters = int(n_iters)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.device = device
        self.model = None
        self.likelihood = None
        self.last_batches_per_iter = 0

    def _build(self, inducing: torch.Tensor):
        self.model = MultitaskSVGP(inducing, self.num_tasks, self.kernel).to(self.device)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "GPRegressor":
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.as_tensor(Y, dtype=torch.float32).to(self.device)
        n = X.size(0)
        n_ind = min(self.n_inducing, n)
        perm = torch.randperm(n)[:n_ind]
        inducing = X[perm].clone().unsqueeze(0).repeat(self.num_tasks, 1, 1)
        self._build(inducing)

        self.model.train()
        self.likelihood.train()
        opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()), lr=self.lr
        )
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=n)

        bs = min(self.batch_size, n)
        self.last_batches_per_iter = (n + bs - 1) // bs
        for _ in range(self.n_iters):
            order = torch.randperm(n, device=X.device)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                opt.zero_grad()
                loss = -mll(self.model(X[idx]), Y[idx])
                loss.backward()
                opt.step()
        self.eval()
        return self

    def _predictive(self, X: torch.Tensor):
        self.model.eval()
        self.likelihood.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.likelihood(self.model(X))

    def sample(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Predictive draws WITH observation noise: [num_samples, N, num_tasks]."""
        with torch.no_grad():
            return self._predictive(X).sample(torch.Size([int(num_samples)]))

    def mean(self, X: torch.Tensor) -> torch.Tensor:
        return self._predictive(X).mean

    def predictive_variance(self, X: torch.Tensor) -> torch.Tensor:
        return self._predictive(X).variance

    def latent_variance(self, X: torch.Tensor) -> torch.Tensor:
        """Latent-function variance, EXCLUDING observation noise. Diagnostic only --
        never sample from this path; see the module docstring."""
        self.model.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.model(X).variance

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
        inducing = self.model.variational_strategy.base_variational_strategy.inducing_points
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
        self.eval()
        return self
