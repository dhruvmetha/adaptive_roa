import re
from pathlib import Path


def test_engine_calibrates_qhat_for_partx():
    src = Path("adaptive_roa/adaptive_v2/engine.py").read_text()
    # The q_hat calibration branch must include "partx".
    assert re.search(r'acquisition_mode in \("conformal", "partx"\)', src)
