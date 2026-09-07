from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.full_roa import FullROAEvaluator


def _cfg(predictor_type="generative"):
    return OmegaConf.create({
        "predictor_type": predictor_type,
        "alpha_eval": 0.1,
        "attractor_radius": 0.2,
        "num_mc_samples_eval": 10,
        "decision_rule": "two_sided",
        "refine_invalids": False,
        "collapse_invalid_to_failure": True,
        "refine_t_min": 0.7,
        "refine_t_max": 0.9,
        "refine_num_steps": 100,
        "refine_max_attempts": 5,
        "max_eval_rows": None,
        "verbose": False,
    })


def test_evaluator_stores_cfg_fields():
    ev = FullROAEvaluator(_cfg(), system=None, device="cpu")
    assert ev.predictor_type == "generative"
    assert ev.attractor_radius == 0.2
    assert ev.num_mc_samples_eval == 10
    assert ev.decision_rule == "two_sided"
    assert ev.collapse_invalid_to_failure is True


def test_evaluator_classifier_cfg():
    ev = FullROAEvaluator(_cfg("classifier"), system=None, device="cpu")
    assert ev.predictor_type == "classifier"
