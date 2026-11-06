"""Atomic writes honour the umask, like a plain open() (dm1487's bug report, 2026-11-06).

tempfile.mkstemp creates its file 0600 and os.replace keeps that mode, so under the
harness's umask 002 every artifacts_v2.json, results.json and partx_state.pkl came out
unreadable to the other account in login-bekris, which breaks cross-account resume
and scoring.
"""
import os
import stat
from types import SimpleNamespace

import pytest


@pytest.fixture
def umask_002():
    old = os.umask(0o002)
    yield
    os.umask(old)


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def test_engine_atomic_json_is_group_readable(tmp_path, umask_002):
    from adaptive_roa.adaptive_v2.engine import _atomic_write_json
    p = tmp_path / "artifacts_v2.json"
    _atomic_write_json(p, {"a": 1})
    assert _mode(p) == 0o664
    _atomic_write_json(p, {"a": 2})                 # an overwrite too
    assert _mode(p) == 0o664
    assert sorted(x.name for x in tmp_path.iterdir()) == ["artifacts_v2.json"]


def test_partx_state_is_group_readable(tmp_path, umask_002):
    from adaptive_roa.partx_faithful.state import load_state, save_state
    p = tmp_path / "partx_state.pkl"
    save_state({"x": 1}, p, "h")
    assert _mode(p) == 0o664 and load_state(p, "h") == {"x": 1}


def test_suffix_builder_state_is_group_readable(tmp_path, umask_002):
    from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder
    b = AdaptiveDatasetBuilder.__new__(AdaptiveDatasetBuilder)
    b.candidate_mode = "suffix"
    b._suffix = SimpleNamespace(to_dict=lambda: {"used": [1, 2]})
    p = tmp_path / "builder_state.json"
    b.save_state(str(p))
    assert _mode(p) == 0o664


def test_mode_follows_a_stricter_umask(tmp_path):
    from adaptive_roa.adaptive_v2.engine import _atomic_write_json
    old = os.umask(0o027)
    try:
        _atomic_write_json(tmp_path / "r.json", {})
    finally:
        os.umask(old)
    assert _mode(tmp_path / "r.json") == 0o640
