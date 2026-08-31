"""The Protocols in interfaces.py must match what implementations actually do."""
import inspect

from adaptive_roa.adaptive_v2 import interfaces
from adaptive_roa.adaptive_v2.trainers.classifier_trainer import ClassifierTrainer
from adaptive_roa.adaptive_v2.trainers.flow_matching_trainer import FlowMatchingTrainer


def test_predictor_trainer_protocol_matches_implementations():
    """Both shipped trainers take dataset_files: dict, not train_file/val_file."""
    proto_params = list(
        inspect.signature(interfaces.PredictorTrainer.fit).parameters
    )
    assert "dataset_files" in proto_params
    assert "train_file" not in proto_params
    for trainer_cls in (ClassifierTrainer, FlowMatchingTrainer):
        impl_params = list(inspect.signature(trainer_cls.fit).parameters)
        assert impl_params[1] == "dataset_files", trainer_cls.__name__


def test_handle_protocols_exist():
    """Both handle contracts are declared, including the determinism asymmetry."""
    assert hasattr(interfaces, "OutcomeModelHandle")
    assert hasattr(interfaces, "FinalStateModelHandle")
    assert "identical" in interfaces.OutcomeModelHandle.__doc__
    assert "fresh" in interfaces.FinalStateModelHandle.__doc__
