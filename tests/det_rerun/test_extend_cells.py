"""q3drl200 continues q3drl100: manifest rows, and copying a finished cell without touching it."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/det_rerun"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/det_rerun/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_q3drl200_rows_run_22_epochs_in_their_own_dirs():
    mm = _load("make_manifest")
    rows = [r for r in mm.cells(99512) if r["system"] == "q3drl200"]
    assert len(rows) == 21 and mm.EXTENDS["q3drl200"] == "q3drl100"
    for r in rows:
        assert " n_epochs=22 " in r["overrides"] and "samples_per_epoch=9000" in r["overrides"]
        assert f"/q3drl200/{r['name']}" in r["overrides"]
    neural = [r for r in rows if r["arm"] != "partx_faithful"]
    assert all("'prediction_store.endpoint_epochs=[0,5,10,15,21]'" in r["overrides"] for r in neural)
    # 10,000 + 21 x 9,000 trajectories at the start of the last epoch.
    assert 10000 + (mm.N_EPOCHS["q3drl200"] - 1) * 9000 == 199000
    assert not any("n_epochs" in r["overrides"] for r in mm.cells(99512) if r["system"] == "q3drl100")


def _source(tmp, n_banked=11):
    src = tmp / "q3drl100" / "dr_q3drl100_fm_ranked_s99512"
    (src / ".hydra").mkdir(parents=True)
    cfg = {"n_epochs": 11, "output_dir": "/scratch/st1122/det_rerun_out/q3drl100/dr_q3drl100_fm_ranked_s99512",
           "data_dir": "${data_dir:}", "eval": {"binary_outcomes": None},
           "prediction_store": {"endpoint_epochs": [0, 5, 10]}}
    (src / ".hydra" / "config.yaml").write_text(yaml.safe_dump(cfg))
    (src / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    (src / ".hydra" / "overrides.yaml").write_text(yaml.safe_dump(
        ["system=quadrotor3d_rl", "prediction_store.endpoint_epochs=[0,5,10]", f"output_dir={cfg['output_dir']}"]))
    (src / "dataset_builder_state.json").write_text(json.dumps({"train_indices": list(range(5))}))
    for e in range(n_banked):
        (src / f"epoch_{e:03d}").mkdir()
        (src / f"epoch_{e:03d}" / "artifacts_v2.json").write_text("{}")
    (src / "final_results.json").write_text("{}")
    return src


def _row(tmp):
    out = tmp / "q3drl200" / "dr_q3drl200_fm_ranked_s99512"
    return {"name": out.name, "system": "q3drl200", "arm": "fm_det", "overrides":
            "system=quadrotor3d_rl +experiment=det_rerun seed=99512 samples_per_epoch=9000 n_epochs=22 "
            f"'prediction_store.endpoint_epochs=[0,5,10,15,21]' output_dir={out}"}, out


def test_extend_copies_patches_and_registers(tmp_path):
    ec = _load("extend_cells")
    src = _source(tmp_path)
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    launched = tmp_path / "launched.tsv"
    launched.write_text("dr_q3drl100_fm_ranked_s99512\tdirect\twesteros:1\t1\t20261101_0000\tlog\tabc1234\n")
    row, out = _row(tmp_path)
    ec.extend(row, "q3drl200", "q3drl100", launched)
    assert sorted(p.name for p in out.glob("epoch_*")) == [f"epoch_{e:03d}" for e in range(11)]
    assert not (out / "final_results.json").exists() and (out / "dataset_builder_state.json").exists()
    for f in (out / ".hydra" / "config.yaml", out / "resolved_config.yaml"):
        cfg = OmegaConf.load(f)
        assert cfg.output_dir == str(out) and cfg.n_epochs == 22 and cfg.eval.binary_outcomes is False
        assert list(cfg.prediction_store.endpoint_epochs) == [0, 5, 10, 15, 21]
    assert "${data_dir:}" in (out / ".hydra" / "config.yaml").read_text()   # interpolations stay unresolved
    ov = yaml.safe_load((out / ".hydra" / "overrides.yaml").read_text())
    assert f"output_dir={out}" in ov and "n_epochs=22" in ov and not any("q3drl100" in o for o in ov)
    assert json.loads((out / "extended_from.json").read_text())["commit"] == "abc1234"
    assert launched.read_text().splitlines()[-1].split("\t") == [
        row["name"], "copied", "-", "-", launched.read_text().splitlines()[-1].split("\t")[4], "-", "abc1234"]
    assert {p: p.read_bytes() for p in src.rglob("*") if p.is_file()} == before   # source untouched
    # A second call leaves the copy alone and registers nothing.
    n = len(launched.read_text().splitlines())
    assert "left as is" in ec.extend(row, "q3drl200", "q3drl100", launched)
    assert len(launched.read_text().splitlines()) == n


def test_finished_amarel_mirror_copies_without_its_marker(tmp_path):
    ec = _load("extend_cells")
    src = _source(tmp_path)
    (src / ".amarel_mirror").write_text("")
    launched = tmp_path / "launched.tsv"
    launched.write_text("dr_q3drl100_fm_ranked_s99512\tamarel\t--partition=gpu\t63400000\t20261101_0000\tlog\t45dc0ce\n")
    row, out = _row(tmp_path)
    ec.extend(row, "q3drl200", "q3drl100", launched)
    assert not (out / ".amarel_mirror").exists() and (src / ".amarel_mirror").exists()


def test_extend_refuses_an_unfinished_source(tmp_path):
    ec = _load("extend_cells")
    _source(tmp_path, n_banked=10)
    launched = tmp_path / "launched.tsv"
    launched.write_text("dr_q3drl100_fm_ranked_s99512\tdirect\twesteros:1\t1\t20261101_0000\tlog\tabc1234\n")
    row, out = _row(tmp_path)
    with pytest.raises(SystemExit, match="not finished"):
        ec.extend(row, "q3drl200", "q3drl100", launched)
    assert not out.exists()
