"""Task 5 tests: AdaptiveClassificationDataModule."""
import numpy as np
import torch

from adaptive_roa.data.adaptive_classification_data import AdaptiveClassificationDataModule


def _write(path, states, labels):
    np.savetxt(str(path), np.hstack([states, labels.reshape(-1, 1)]), fmt="%.6f")


def test_batches_and_pos_weight(tmp_path):
    # 8 failures (0), 2 successes (1) -> pos_weight = n_neg/n_pos = 8/2 = 4
    x_train = np.random.randn(10, 6)
    y_train = np.array([0.0] * 8 + [1.0] * 2)
    x_val = np.random.randn(4, 6)
    y_val = np.array([0.0, 1.0, 0.0, 1.0])

    train_f = tmp_path / "train.txt"
    val_f = tmp_path / "val.txt"
    _write(train_f, x_train, y_train)
    _write(val_f, x_val, y_val)

    dm = AdaptiveClassificationDataModule(str(train_f), str(val_f), batch_size=4, num_workers=0)
    dm.setup()

    assert abs(dm.pos_weight - 4.0) < 1e-6
    batch = next(iter(dm.train_dataloader()))
    assert batch["inputs"].shape == (4, 6)
    assert batch["inputs"].dtype == torch.float32
    assert batch["label"].dtype == torch.float32
