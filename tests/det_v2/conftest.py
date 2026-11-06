"""The `world` fixture: a stub det_v2 world per test (det_v2_world.py)."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from det_v2_world import World  # noqa: E402


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    subprocess.run(["chmod", "-R", "u+w", str(tmp_path)], check=False)
