"""Task 3 tests: ClassifierMLP + ClassifierModule."""
import torch
import torch.nn.functional as F

from adaptive_roa.model.classifier_mlp import ClassifierMLP, ClassifierModule


class _IdSystem:
    """Trivial system: no normalization, identity embedding."""

    def normalize_state(self, x):
        return x

    def embed_state_for_model(self, x):
        return x


def test_forward_shape():
    module = ClassifierModule(ClassifierMLP(input_dim=4, hidden_dims=[16, 16]), _IdSystem())
    out = module(torch.randn(8, 4))
    assert out.shape == (8, 1)


def test_learns_separable_problem():
    torch.manual_seed(0)
    module = ClassifierModule(ClassifierMLP(input_dim=2, hidden_dims=[32, 32]), _IdSystem(), lr=1e-2)
    opt = torch.optim.Adam(module.parameters(), lr=1e-2)

    X = torch.randn(256, 2)
    y = (X[:, 0] > 0).float()

    first_loss = None
    for _ in range(150):
        opt.zero_grad()
        logits = module(X).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=module.pos_weight)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()

    preds = (torch.sigmoid(module(X).view(-1)) > 0.5).float()
    acc = (preds == y).float().mean().item()
    assert loss.item() < first_loss * 0.5
    assert acc > 0.9
