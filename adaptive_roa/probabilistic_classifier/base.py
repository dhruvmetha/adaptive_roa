from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class ProbabilisticClassifier(ABC):
    """Uniform wrapper over a probability-producing model.

    Subclasses declare ``predictor_type`` (matches the Hydra ``predictor``
    config value) and ``native_probs`` (the OutcomeProbabilities field names
    the method natively produces; these are the arrays the export writes).
    """

    predictor_type: str = ""
    predictor_name: str = ""
    native_probs: tuple[str, ...] = ()

    @abstractmethod
    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        """Compute probabilities for ``states`` via live model inference."""
        raise NotImplementedError

    def predict_cached(
        self, run_dir: str, epoch: int, split: str, states: np.ndarray
    ) -> Optional[OutcomeProbabilities]:
        """Return cached probabilities for (epoch, split) if available, else None.

        Base implementation has no cache. Subclasses override when a fast path
        (e.g. a precomputed MC cache) exists.
        """
        return None

    @classmethod
    @abstractmethod
    def load_from_run(
        cls, run_dir: str, epoch: int, cfg, system, device: str = "cuda"
    ) -> "ProbabilisticClassifier":
        """Load the model for ``epoch`` from a training run directory."""
        raise NotImplementedError
