"""_atomic_write_json must never leave a truncated artifact on disk.

engine.py writes epoch_*/artifacts_v2.json (and its siblings) through this
helper specifically because the `general` account preempts jobs silently.
A plain `open(path, "w")` truncates the destination before writing a single
byte, so a preemption mid-write leaves an empty or half-written file that
launcher.py's completed_epochs() could miscount as a finished epoch. os.replace
is atomic within a filesystem: any reader sees either the complete old file
or the complete new one, never a mix of the two.
"""
import json

from adaptive_roa.adaptive_v2.engine import _atomic_write_json


def test_happy_path_writes_valid_parseable_json(tmp_path):
    # An atomic write that silently landed in the wrong place would still
    # pass a "no corrupt file" check while breaking everything downstream --
    # so assert on the actual destination path and its exact content.
    target = tmp_path / "artifacts_v2.json"
    _atomic_write_json(target, {"epoch": 3, "ok": True}, indent=2)
    assert target.exists()
    assert json.loads(target.read_text()) == {"epoch": 3, "ok": True}


def test_no_temp_file_left_behind_on_success(tmp_path):
    target = tmp_path / "artifacts_v2.json"
    _atomic_write_json(target, {"epoch": 1})
    assert list(tmp_path.iterdir()) == [target]


def test_old_file_survives_a_write_that_never_completes(tmp_path, monkeypatch):
    # Simulate a preemption that kills the process while json.dump is still
    # writing into the temp file, before os.replace ever runs.
    target = tmp_path / "artifacts_v2.json"
    target.write_text(json.dumps({"epoch": 0}))

    def _boom(*a, **k):
        raise OSError("simulated preemption mid-write")

    monkeypatch.setattr(json, "dump", _boom)
    try:
        _atomic_write_json(target, {"epoch": 1})
    except OSError:
        pass

    # The old file must still be intact and parseable -- never truncated or
    # partially overwritten by the interrupted new write.
    assert json.loads(target.read_text()) == {"epoch": 0}
    # And no leftover temp file from the aborted write.
    assert list(tmp_path.iterdir()) == [target]


def test_no_temp_file_left_behind_when_there_was_no_old_file(tmp_path, monkeypatch):
    target = tmp_path / "artifacts_v2.json"

    def _boom(*a, **k):
        raise OSError("simulated preemption mid-write")

    monkeypatch.setattr(json, "dump", _boom)
    try:
        _atomic_write_json(target, {"epoch": 0})
    except OSError:
        pass

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
