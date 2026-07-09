import numpy as np
from adaptive_roa.partx.trainer import load_xy


def test_load_xy_signed_scheme(tmp_path):
    f = tmp_path / "train.txt"
    # (theta, theta_dot, label): 1=success, -1=failure, 0=separatrix, -2=invalid
    f.write_text("0.0 0.0 1\n2.1 0.0 -1\n1.0 0.0 0\n0.5 0.0 -2\n")
    X, y = load_xy(str(f), state_dim=2)
    assert X.shape == (2, 2)           # separatrix + invalid dropped
    assert list(y) == [1, 0]           # success->1, failure->0


def test_load_xy_binary_scheme(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("0.0 0.0 1\n2.1 0.0 0\n")   # {0,1} scheme: keep all
    X, y = load_xy(str(f), state_dim=2)
    assert X.shape == (2, 2)
    assert list(y) == [1, 0]
