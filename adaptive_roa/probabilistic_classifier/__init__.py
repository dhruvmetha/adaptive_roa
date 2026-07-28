from .base import ProbabilisticClassifier
from .registry import (
    register_probabilistic_classifier,
    get_probabilistic_classifier_class,
)

# Import the concrete wrappers for their registration side-effects, so that
# get_probabilistic_classifier_class("classifier"/"generative") works for any
# consumer without having to import the submodules manually first.
from . import classifier as _classifier  # noqa: F401,E402
from . import flow_matching as _flow_matching  # noqa: F401,E402
from . import bayesian as _bayesian  # noqa: F401,E402

__all__ = [
    "ProbabilisticClassifier",
    "register_probabilistic_classifier",
    "get_probabilistic_classifier_class",
]
