import numpy as np
from adaptive_roa.partx.eval import merge_region_bounds


def test_merge_region_bounds():
    base = {"coverage": 0.9, "f1": 0.8}
    diag = {"roa_volume": 0.42, "roa_volume_ci": [0.4, 0.44], "n_leaves": 12}
    out = merge_region_bounds(base, diag)
    assert out["coverage"] == 0.9
    assert out["partx_roa_volume"] == 0.42
    assert out["partx_roa_volume_ci"] == [0.4, 0.44]
    assert out["partx_n_leaves"] == 12


def test_merge_region_bounds_none_diag():
    base = {"coverage": 0.9}
    assert merge_region_bounds(base, None) == {"coverage": 0.9}
