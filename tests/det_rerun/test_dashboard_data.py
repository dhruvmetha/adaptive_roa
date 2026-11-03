"""The dashboard's FPR-capped 110k/100k quad3d settings are the 200k runs' first 11 epochs."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location("dashboard_data", ROOT / "scripts/det_rerun/dashboard_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_read_off_takes_epochs_0_to_10_of_the_200k_cells():
    dd = _mod()
    rows = [{"setting": "q3d200", "cell": "c", "epoch": e, "budget": 200000, "n_epochs": 20, "status": "running"}
            for e in range(13)] + [{"setting": "pen", "cell": "p", "epoch": 0, "budget": 600, "n_epochs": 11,
                                     "status": "running"}]
    out = dd.read_off(rows)
    assert [r["epoch"] for r in out] == list(range(11))
    assert all(r["setting"] == "q3d110" and r["budget"] == 110000 and r["n_epochs"] == 11 for r in out)
    assert all(r["status"] == "finished" for r in out)                    # 13 epochs banked >= 11
    assert all(r["setting"] == "q3d200" for r in rows if r["cell"] == "c")   # source rows untouched
    assert dd.read_off([dict(r, epoch=0) for r in rows[:1]])[0]["status"] == "running"
