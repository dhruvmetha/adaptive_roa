"""Direct acquisition strategy (D2 with lambda/delta thresholds)."""

from __future__ import annotations

from typing import Any

from adaptive_roa.adaptive.balanced_sampler import UncertainSampler
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


def resolve_rule(cfg: Any) -> str:
    """acquisition.rule: lambda_delta (default, the current tests) or tau (threshold=fpr_cap)."""
    rule = str(cfg.get("rule", "lambda_delta"))
    if rule not in ("lambda_delta", "tau"):
        raise ValueError(f"acquisition.rule must be 'lambda_delta' or 'tau', got {rule!r}")
    return rule


def taus_for_rule(rule: str, threshold_state: ThresholdState):
    """(tau_s, tau_f, tau_inv) for rule=tau; refuses a rule the threshold state cannot serve."""
    taus = getattr(threshold_state, "taus", None)
    if rule == "tau":
        if taus is None:
            raise ValueError("acquisition.rule=tau needs thresholds from threshold=fpr_cap; "
                             "this threshold state carries lambda/delta only")
        return taus
    if taus is not None:
        raise ValueError("threshold=fpr_cap fits tau_s/tau_f/tau_inv, whose converted delta can be "
                         "negative; set acquisition.rule=tau")
    return None


class DirectAcquisitionStrategy:
    mode = "direct"

    def __init__(self, cfg: Any):
        self.d2_ratio = float(cfg.d2_ratio)
        self.batch_size_sampling = int(cfg.batch_size_sampling)
        self.max_samples_per_epoch = int(cfg.max_samples_per_epoch)
        self.decision_rule = str(cfg.decision_rule)
        self.verbose = bool(cfg.verbose)
        self.rule = resolve_rule(cfg)

    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        taus = taus_for_rule(self.rule, threshold_state)
        sampler = UncertainSampler(
            dataset_builder=pool.dataset_builder,
            target_count=target_count,
            batch_size=self.batch_size_sampling,
            max_candidates=self.max_samples_per_epoch,
        )
        rule_kwargs = {"rule": "tau", "taus": taus} if self.rule == "tau" else {}
        sample_result = sampler.sample_direct(
            prob_estimator=probability_backend.estimator,
            lambda_star=threshold_state.lambda_star,
            delta_star=threshold_state.delta_star,
            decision_rule=self.decision_rule,
            exclude=exclude or set(),
            verbose=self.verbose,
            **rule_kwargs,
        )
        return AcquisitionResult(
            d1_indices=[],
            d2_indices=list(sample_result.uncertain_indices),
            n_candidates_evaluated=int(sample_result.n_candidates_evaluated),
            n_certain_discarded=int(sample_result.n_certain_discarded),
            n_invalid_added=0,
            diagnostics={"n_batches": int(sample_result.n_batches),
                         **({"rule": "tau", "taus": list(taus)} if self.rule == "tau" else {})},
        )
