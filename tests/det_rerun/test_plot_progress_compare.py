"""--compare renders the method figure and the FM-regime figure as a pair."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pp(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("plot_progress", ROOT / "scripts/det_rerun/plot_progress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path)
    cell = ([10.0, 20.0], [[0.5, 0.4], [0.8, 0.85], [20.0, 15.0], [5.0, 4.0], [15.0, 11.0], [0.1, 0.08],
                           [0.2, 0.15]], True)
    data = {s: {("fm_det", m): cell for m in ("uniform", "direct", "ranked", "entropy")} for s in mod.SETTINGS}
    data["pen"][("det_mlp_clf", "ranked")] = cell
    monkeypatch.setattr(mod, "load", lambda: data)
    monkeypatch.setattr(sys, "argv", ["plot_progress.py", "--compare"])
    return mod, data


def test_compare_prints_both_figures_then_nothing(pp, capsys, tmp_path):
    mod, _ = pp
    mod.main()
    printed = capsys.readouterr().out.split()
    assert printed == [str(tmp_path / "compare_ranked.png"), str(tmp_path / "compare_fm_regimes.png")]
    assert all(Path(p).stat().st_size > 0 for p in printed)
    mod.main()
    assert capsys.readouterr().out == ""


def test_change_in_one_figure_reprints_the_pair(pp, capsys, tmp_path):
    mod, data = pp
    mod.main()
    capsys.readouterr()
    xs, ys, _ = data["q2d"][("fm_det", "entropy")]       # FM-regime figure only
    data["q2d"][("fm_det", "entropy")] = (xs, ys, False)
    mod.main()
    assert len(capsys.readouterr().out.split()) == 2
