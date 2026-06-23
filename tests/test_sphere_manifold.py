"""Independent verification of the FB FM Sphere/Product used by HumanoidStandUpReach.

Confirms the ambient-3 tangent convention (singularity-free), so the flow matcher's
model output_dim is 67 and the distance manifold returns 65 components.
"""
import pytest
import torch
from flow_matching.utils.manifolds import Product, Euclidean, Sphere


def _unit_sphere_block(x):
    x = x.clone()
    n = x[:, 34:37].norm(dim=1, keepdim=True).clamp(min=1e-8)
    x[:, 34:37] = x[:, 34:37] / n
    return x


def _humanoid_product():
    return Product(
        input_dim=67,
        manifolds=[(Euclidean(), 34, 34), (Sphere(), 3, 3), (Euclidean(), 30, 30)],
    )


def test_product_rejects_reduced_sphere_tangent():
    with pytest.raises(ValueError, match="Sphere manifold must have state_dim == tangent_dim"):
        Product(input_dim=67, manifolds=[(Euclidean(), 34, 34), (Sphere(), 3, 2), (Euclidean(), 30, 30)])


def test_product_accepts_ambient_sphere_and_dist_has_65_components():
    m = _humanoid_product()
    a = _unit_sphere_block(torch.randn(8, 67))
    b = _unit_sphere_block(torch.randn(8, 67))
    d = m.dist(a, b)
    assert d.shape == (8, 65)


def test_bare_sphere_dist_is_one_value_per_pair():
    s = Sphere()
    x = torch.randn(8, 3); x = x / x.norm(dim=1, keepdim=True)
    y = torch.randn(8, 3); y = y / y.norm(dim=1, keepdim=True)
    assert s.dist(x, y).shape == (8, 1)


def test_proju_is_orthogonal_to_point():
    s = Sphere()
    x = torch.randn(8, 3); x = x / x.norm(dim=1, keepdim=True)
    v = torch.randn(8, 3)
    pv = s.proju(x, v)
    assert pv.shape == (8, 3)
    assert torch.allclose((x * pv).sum(dim=1), torch.zeros(8), atol=1e-5)


def test_expmap_preserves_unit_norm():
    s = Sphere()
    x = torch.randn(8, 3); x = x / x.norm(dim=1, keepdim=True)
    u = s.proju(x, torch.randn(8, 3) * 0.3)
    y = s.expmap(x, u)
    assert torch.allclose(y.norm(dim=1), torch.ones(8), atol=1e-5)


def test_log_exp_roundtrip_small_tangent():
    s = Sphere()
    x = torch.randn(8, 3); x = x / x.norm(dim=1, keepdim=True)
    u = s.proju(x, torch.randn(8, 3) * 0.1)  # small to avoid antipodal ambiguity
    u_back = s.logmap(x, s.expmap(x, u))
    assert torch.allclose(u, u_back, atol=1e-4)
