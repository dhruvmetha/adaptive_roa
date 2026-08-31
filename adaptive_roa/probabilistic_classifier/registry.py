from __future__ import annotations

from typing import Type

from .base import ProbabilisticClassifier

_REGISTRY: dict[str, Type[ProbabilisticClassifier]] = {}


def register_probabilistic_classifier(cls: Type[ProbabilisticClassifier]):
    """Register ``cls`` under its arm name, and under its family tag as an alias.

    ``predictor_type`` is a family tag ("classifier"/"generative") shared by
    several arms, so it cannot be the primary key -- five outcome arms would
    collide and the export would load the wrong checkpoint class. The family
    alias is kept so runs written before ``predictor.name`` existed still
    resolve; first registration wins so a later arm cannot steal the alias.
    """
    if not cls.predictor_type:
        raise ValueError(f"{cls.__name__} must set a non-empty predictor_type")
    if not cls.predictor_name:
        raise ValueError(f"{cls.__name__} must set a non-empty predictor_name")
    _REGISTRY[cls.predictor_name] = cls
    _REGISTRY.setdefault(cls.predictor_type, cls)
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
