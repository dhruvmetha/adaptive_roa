"""Task 6 test: ClassifierTrainer end-to-end smoke (CPU, 2 epochs)."""
import numpy as np
import pytest
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


def _imbalanced(tmp_path, pos_frac=0.1, n=200, seed=0):
    """A fixture whose data-derived pos_weight is clearly != 1.0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = np.zeros(n)
    y[: max(1, int(n * pos_frac))] = 1.0
    rng.shuffle(y)
    path = tmp_path / f"data_{seed}.txt"
    _write(path, X, y)
    return str(path), float((y == 0).sum()) / float((y == 1).sum())


def _clf_cfg(**classifier_overrides):
    clf = {"hidden_dims": [16, 16], "lr": 1e-2, "max_epochs": 1,
           "patience": 5, "weight_decay": 1e-5}
    clf.update(classifier_overrides)
    return OmegaConf.create({
        "device": "cpu", "batch_size": 16, "num_workers": 0, "classifier": clf,
        "lightning_trainer": {"gradient_clip_val": 1.0, "log_every_n_steps": 1},
    })


def test_an_explicit_pos_weight_overrides_the_data_derived_one(tmp_path):
    """The reference tier pins pos_weight = 1.0 so every arm targets the SAME
    untempered posterior HMC references. This trainer hardcoded
    `data_module.pos_weight` and read no config key at all, so the tier's
    `classifier` arm silently trained against a CLASS-TEMPERED likelihood while
    HMC referenced the unweighted one -- a mismatch invisible in every artifact.
    """
    train_f, data_derived = _imbalanced(tmp_path, seed=0)
    val_f, _ = _imbalanced(tmp_path, n=80, seed=1)
    files = {"train": train_f, "val": val_f}
    assert data_derived != pytest.approx(1.0), "fixture must be imbalanced"

    default = ClassifierTrainer(_clf_cfg(), _FakeSystem(), "fake").fit(
        files, str(tmp_path / "default")
    )
    override = ClassifierTrainer(_clf_cfg(pos_weight=1.0), _FakeSystem(), "fake").fit(
        files, str(tmp_path / "override")
    )

    assert default.pos_weight.item() == pytest.approx(data_derived)
    assert override.pos_weight.item() == pytest.approx(1.0)
    assert default.pos_weight.item() != pytest.approx(override.pos_weight.item())


def test_the_activation_is_configurable_and_defaults_to_relu(tmp_path):
    """The tier's contract is "[50,50] tanh for every arm", and ReLU's
    non-differentiability is exactly what degrades leapfrog's local error to
    O(eps). A hardcoded ReLU here put this arm on a different backbone than the
    tier claimed, with nothing to say so."""
    train_f, _ = _imbalanced(tmp_path, seed=0)
    val_f, _ = _imbalanced(tmp_path, n=80, seed=1)
    files = {"train": train_f, "val": val_f}

    default = ClassifierTrainer(_clf_cfg(), _FakeSystem(), "fake").fit(
        files, str(tmp_path / "relu")
    )
    tanh = ClassifierTrainer(_clf_cfg(activation="tanh"), _FakeSystem(), "fake").fit(
        files, str(tmp_path / "tanh")
    )

    kinds = lambda m: [type(x).__name__ for x in m.mlp.net]
    assert "ReLU" in kinds(default) and "Tanh" not in kinds(default)
    assert "Tanh" in kinds(tanh) and "ReLU" not in kinds(tanh)


def test_the_reference_tier_pins_the_classifier_arm_too():
    """The tier file must actually carry the keys this trainer now reads. It
    listed `classifier.hidden_dims` alone, which composed cleanly and left the
    arm on a tempered likelihood and a ReLU backbone."""
    tier = OmegaConf.load("configs/adaptive_v2/experiment/reference_tier.yaml")
    clf = tier.predictor.classifier
    assert list(clf.hidden_dims) == [50, 50]
    assert clf.activation == "tanh"
    assert float(clf.pos_weight) == 1.0
