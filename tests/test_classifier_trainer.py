"""Task 6 test: ClassifierTrainer end-to-end smoke (CPU, 2 epochs)."""
import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.classifier_trainer import ClassifierTrainer


class _FakeSystem:
    """Identity normalize/embed; 4D state."""

    @property
    def state_dim(self):
        return 4

    def normalize_state(self, x):
        return x

    def embed_state_for_model(self, x):
        return x


def _write(path, X, y):
    np.savetxt(str(path), np.hstack([X, y.reshape(-1, 1)]), fmt="%.6f")


def test_trainer_smoke(tmp_path):
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((64, 4))
    y_train = (x_train[:, 0] > 0).astype(float)
    x_val = rng.standard_normal((16, 4))
    y_val = (x_val[:, 0] > 0).astype(float)
    train_f, val_f = tmp_path / "train.txt", tmp_path / "val.txt"
    _write(train_f, x_train, y_train)
    _write(val_f, x_val, y_val)

    cfg = OmegaConf.create({
        "device": "cpu", "batch_size": 16, "num_workers": 0,
        "classifier": {"hidden_dims": [16, 16], "lr": 1e-2, "max_epochs": 2,
                       "patience": 5, "weight_decay": 1e-5},
        "lightning_trainer": {"gradient_clip_val": 1.0, "log_every_n_steps": 1, "enable_progress_bar": False},
    })

    trainer = ClassifierTrainer(cfg, _FakeSystem(), system_name="fake")
    module = trainer.fit({"train": str(train_f), "val": str(val_f)}, str(tmp_path / "out"))

    out = module(torch.randn(5, 4))
    assert out.shape == (5, 1)
