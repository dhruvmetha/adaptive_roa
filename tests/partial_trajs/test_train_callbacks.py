"""The non-adaptive trainer checkpoints but does NOT early-stop.

These partial-trajectory runs are not adaptive: they train the full ``max_epochs``
(no ``EarlyStopping``), keeping only the best-val + last checkpoints.
"""
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf

from adaptive_roa.partial_trajs.train import build_callbacks


def test_build_callbacks_checkpoints_without_early_stopping(tmp_path):
    callbacks = build_callbacks(OmegaConf.create({}), tmp_path)
    assert any(isinstance(cb, ModelCheckpoint) for cb in callbacks), "expected a ModelCheckpoint"
    assert not any(isinstance(cb, EarlyStopping) for cb in callbacks), "must not early-stop"


def test_build_callbacks_checkpoint_dir_and_best_last(tmp_path):
    (cb,) = [c for c in build_callbacks(OmegaConf.create({}), tmp_path) if isinstance(c, ModelCheckpoint)]
    assert cb.dirpath == str(tmp_path)
    assert cb.monitor == "val_loss"
    assert cb.save_last is True
