"""Task 4 tests: classification dataset writer + DATA_DIR safety guard."""
import os

import numpy as np
import pytest

from adaptive_roa.adaptive.data_source import TrajectoryDataSource
from adaptive_roa.utils.env_config import get_data_dir


class _StubSource:
    """Provides only build_endpoint_dataset, the sole dependency of the method."""

    def build_endpoint_dataset(self, indices, mode):
        starts = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=float)
        ends = np.zeros_like(starts)
        labels = np.array([1, -1, 1], dtype=np.int64)  # internal: success, failure, success
        return starts, ends, labels


def test_writes_state_and_binary_label(tmp_path):
    out = tmp_path / "cls.txt"
    n = TrajectoryDataSource.save_classification_dataset(_StubSource(), [0], str(out), "train")

    assert n == 3
    data = np.loadtxt(str(out))
    assert data.shape == (3, 3)  # 2 state cols + 1 label col
    np.testing.assert_allclose(data[:, :2], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    # internal {1, -1, 1} -> binary {1.0, 0.0, 1.0}
    np.testing.assert_array_equal(data[:, -1], [1.0, 0.0, 1.0])


def test_refuses_to_write_under_data_dir():
    bad = os.path.join(get_data_dir(), "should_never_be_written.txt")
    with pytest.raises(ValueError):
        TrajectoryDataSource.save_classification_dataset(_StubSource(), [0], bad, "train")
    assert not os.path.exists(bad)
