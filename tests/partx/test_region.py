import numpy as np
from adaptive_roa.partx.region import Region


def _box():
    return Region(low=np.array([-np.pi, -8.0]), high=np.array([np.pi, 8.0]),
                  norm_scale=np.array([2 * np.pi, 16.0]))


def test_contains_and_volume():
    r = _box()
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert list(r.contains(pts)) == [True, False]
    assert np.isclose(r.volume(), 1.0)          # full support -> normalized volume 1


def test_longest_dim_and_subdivide():
    r = _box()
    assert r.longest_dim() == 0                  # both normalized to 1.0 -> ties to 0
    children = r.subdivide(branching_factor=2)
    assert len(children) == 2
    assert np.isclose(sum(c.volume() for c in children), r.volume())
    # split is along longest normalized dim at the midpoint by default
    assert np.isclose(children[0].high[0], 0.0) and np.isclose(children[1].low[0], 0.0)
