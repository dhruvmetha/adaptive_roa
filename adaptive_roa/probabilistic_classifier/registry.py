from __future__ import annotations

from typing import Type

from .base import ProbabilisticClassifier

_REGISTRY: dict[str, Type[ProbabilisticClassifier]] = {}


def register_probabilistic_classifier(cls: Type[ProbabilisticClassifier]):
    """Class decorator: register ``cls`` under its ``predictor_type``."""
    if not cls.predictor_type:
        raise ValueError(f"{cls.__name__} must set a non-empty predictor_type")
    _REGISTRY[cls.predictor_type] = cls
    return cls


def get_probabilistic_classifier_class(
    predictor_type: str,
) -> Type[ProbabilisticClassifier]:
    if predictor_type not in _REGISTRY:
        raise KeyError(
            f"No probabilistic classifier registered for predictor="
            f"{predictor_type!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[predictor_type]
