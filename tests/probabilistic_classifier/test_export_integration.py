import json
from pathlib import Path

import numpy as np
import torch
import yaml

from adaptive_roa.model.classifier_mlp import ClassifierMLP, ClassifierModule
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.probabilistic_classifier.export import export_run


def _save_clf_ckpt(epoch_dir, system):
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    mlp = ClassifierMLP(input_dim=input_dim, hidden_dims=[8], output_dim=1, dropout=0.0)
    module = ClassifierModule(mlp=mlp, system=system, pos_weight=torch.tensor(1.0), lr=1e-3, weight_decay=1e-5)
    ckpt_dir = Path(epoch_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save({"state_dict": module.state_dict()}, ckpt_dir / "best.ckpt")


def _write_clf_rows(path, n, sd):
    rows = np.random.randn(n, sd)
    labels = np.random.randint(0, 2, size=(n, 1))
    np.savetxt(path, np.hstack([rows, labels]), delimiter=" ")


def _write_eval_states(path, n, sd):
    start = np.random.randn(n, sd)
    end = np.random.randn(n, sd)
    labels = np.random.randint(0, 2, size=(n, 1))
    np.savetxt(path, np.hstack([start, end, labels]), delimiter=",")


def test_export_run_classifier_writes_four_splits(tmp_path):
    system = PendulumSystem()
    sd = int(system.state_dim)
    run_dir = tmp_path / "run"
    (run_dir / "datasets").mkdir(parents=True)
    _save_clf_ckpt(run_dir / "epoch_000", system)
    _write_clf_rows(run_dir / "datasets" / "train_classification_dataset.txt", 6, sd)
    _write_clf_rows(run_dir / "datasets" / "val_classification_dataset.txt", 4, sd)
    cal_file = tmp_path / "cal_set.txt"
    test_file = tmp_path / "test_set.txt"
    _write_eval_states(cal_file, 5, sd)
    _write_eval_states(test_file, 7, sd)

    cfg = {
        "predictor": "classifier",
        "system": {"_target_": "adaptive_roa.systems.pendulum.PendulumSystem"},
        "classifier": {"hidden_dims": [8], "dropout": 0.0},
        "data_source": {"cal_set_file": str(cal_file), "test_set_file": str(test_file)},
    }
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "out"
    export_run(str(run_dir), str(out_dir), device="cpu")

    ep = out_dir / "epoch_000"
    for split, n in [("train", 6), ("val", 4), ("cal", 5), ("test", 7)]:
        d = np.load(ep / f"{split}.npz")
        assert d["query_state"].shape == (n, sd)
        assert d["gt_label"].shape == (n,)
        assert set(np.unique(d["gt_label"])).issubset({-1, 1})
        assert "p_success" in d.files
        assert "p_failure" not in d.files  # classifier native_probs = (p_success,)
        assert np.all((d["p_success"] >= 0.0) & (d["p_success"] <= 1.0))
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["predictor"] == "classifier"
    assert meta["native_probs"] == ["p_success"]
