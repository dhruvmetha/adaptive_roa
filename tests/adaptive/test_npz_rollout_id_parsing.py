"""Shuffled-indices files ship in two formats; both must load.

stochastic/cartpole/noisy_torque writes the collector's filenames
('sequence_115038.txt') where every other family writes bare integers.
np.loadtxt(dtype=int64) raised on those and killed two launched runs 8s in.

Verified against the real file before this parser was written: the embedded
number is the npz row -- it forms an exact permutation of 0..116241 and gives
shuffled_labels[i] == labels[perm[i]] at 1.000000, against 0.821 for direct
indexing. The mapping is load-bearing, not an ordering coincidence.
"""

import numpy as np
import pytest

from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource


def _write(tmp_path, lines, name="idx.txt"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_plain_integer_format(tmp_path):
    out = NpzTrajectoryDataSource._parse_rollout_ids(_write(tmp_path, ["3", "1", "2", "0"]))
    np.testing.assert_array_equal(out, [3, 1, 2, 0])


def test_sequence_filename_format(tmp_path):
    out = NpzTrajectoryDataSource._parse_rollout_ids(
        _write(tmp_path, ["sequence_115038.txt", "sequence_61759.txt", "sequence_0.txt"]))
    np.testing.assert_array_equal(out, [115038, 61759, 0])


def test_mixed_and_blank_lines_are_tolerated(tmp_path):
    out = NpzTrajectoryDataSource._parse_rollout_ids(
        _write(tmp_path, ["7", "", "sequence_9.txt", "  ", "11"]))
    np.testing.assert_array_equal(out, [7, 9, 11])


def test_dtype_is_int64_so_it_can_index_offsets(tmp_path):
    out = NpzTrajectoryDataSource._parse_rollout_ids(_write(tmp_path, ["sequence_5.txt"]))
    assert out.dtype == np.int64


def test_an_unparseable_line_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="cannot read a rollout id"):
        NpzTrajectoryDataSource._parse_rollout_ids(_write(tmp_path, ["not_an_id_at_all"]))


def test_empty_file_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="no rollout ids"):
        NpzTrajectoryDataSource._parse_rollout_ids(_write(tmp_path, [""]))
