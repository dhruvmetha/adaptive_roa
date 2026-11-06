"""tau* regime hysteresis (ablation A9; threshold=j_sel_hysteresis).

J_sel's argmin flips between two regimes from round to round: "inv" (tau_s + tau_f
>= 1, where no state can pass both tests, so every abstention is A_inv) and "out"
(tau_s + tau_f < 1). This backend keeps the incumbent regime unless the other one
is clearly better:

- round 0 takes the global argmin, as JSelThresholdBackend does;
- each later round takes the best pair in each regime, and switches only when the
  other regime's best J_sel is below the incumbent's by more than ``se_mult`` times
  the bootstrap SE of the gap J_inc - J_other (``n_boot`` resamples of the fit rows'
  trajectories, not of the rows: a trajectory contributes many correlated rows);
- tau* is refit inside the chosen regime every round.

``margin`` set to a number replaces the bootstrap with that fixed margin (the
offline replay against the stored predictions_thrfit uses it). The state carries
the regime, both regimes' best J and the margin; ``restore`` reads them back from
the last banked epoch so a resumed run keeps its incumbent.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.eval import det_v2_rule
from adaptive_roa.adaptive_v2.threshold.jsel_threshold import JSelThresholdBackend, to_lambda_delta
from adaptive_roa.adaptive_v2.types import ThresholdState

OTHER = {"inv": "out", "out": "inv"}


def regime_minima(j: np.ndarray) -> tuple[float, float]:
    """Best J in "inv" and in "out" on the [99, 99] grid."""
    return float(j[det_v2_rule.region_mask("inv")].min()), float(j[det_v2_rule.region_mask("out")].min())


def bootstrap_gap_se(p_s, p_f, y, groups, w: float, n_boot: int, rng: np.random.Generator) -> float:
    """SE of best-J_sel("inv") - best-J_sel("out") over trajectory resamples of the fit rows."""
    key = det_v2_rule.cell_keys(p_s, p_f)
    succ = np.asarray(y) == 1
    _, traj = np.unique(np.asarray(groups), return_inverse=True)
    traj = traj.ravel()
    n_traj = int(traj.max()) + 1
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, n_traj, n_traj), minlength=n_traj)
        counts = det_v2_rule.counts_from_keys(key, succ, weights=mult[traj])
        j_inv, j_out = regime_minima(det_v2_rule.objectives(counts, w)["j_sel"])
        gaps[b] = j_inv - j_out
    return float(gaps.std(ddof=1))


class HysteresisJSelThresholdBackend(JSelThresholdBackend):
    needs_groups = True

    def __init__(self, cfg: Any, system: Any, device: str):
        super().__init__(cfg, system, device)
        if self.objective != "j_sel":
            raise ValueError("threshold=j_sel_hysteresis compares J_sel across regimes; objective must be j_sel")
        if self.predictor_type == "classifier":
            raise ValueError("classifiers search tau_s + tau_f >= 1 only, so there is no regime to hold (A9 excludes them)")
        self.n_boot = int(cfg.get("n_boot", 200))
        self.se_mult = float(cfg.get("se_mult", 2.0))
        margin = cfg.get("margin", None)
        self.fixed_margin = None if margin is None else float(margin)
        self.seed = int(cfg.get("seed", 0))
        self.region: str | None = None
        self.optimize_mode = "det_v2_j_sel_hysteresis"

    def restore(self, threshold_state: dict | None) -> None:
        """Take the incumbent regime from a banked epoch's threshold_state (resume)."""
        region = (threshold_state or {}).get("threshold_region")
        if region is not None and region not in OTHER:
            raise ValueError(f"threshold_region must be one of {tuple(OTHER)}, got {region!r}")
        self.region = region

    def _margin(self, p_s, p_f, y, groups) -> float:
        if self.fixed_margin is not None:
            return self.fixed_margin
        if groups is None:
            raise ValueError("threshold=j_sel_hysteresis needs each fit row's trajectory (groups) for the bootstrap")
        # Seeded by the run seed and the fit set's size, so a resumed round redraws the same resamples.
        rng = np.random.default_rng([self.seed, len(np.asarray(y)), len(np.unique(groups))])
        return self.se_mult * bootstrap_gap_se(p_s, p_f, y, groups, self.w, self.n_boot, rng)

    def optimize_probs(self, states, y, p_success, p_failure, p_invalid,
                       num_mc_samples: int | None, groups=None) -> ThresholdState:
        j = det_v2_rule.objectives(det_v2_rule.grid_counts(p_success, p_failure, y), self.w)["j_sel"]
        j_inv, j_out = regime_minima(j)
        margin = self._margin(p_success, p_failure, y, groups)
        previous = self.region
        if previous is None:
            glob = det_v2_rule.fit(p_success, p_failure, y, w=self.w, objective="j_sel")
            region = det_v2_rule.region_of(glob["tau_s"], glob["tau_f"])
        else:
            j_inc, j_oth = (j_inv, j_out) if previous == "inv" else (j_out, j_inv)
            region = OTHER[previous] if j_oth < j_inc - margin else previous
        res = det_v2_rule.fit(p_success, p_failure, y, w=self.w, objective="j_sel", region=region)
        self.region = region
        ts, tf = res["tau_s"], res["tau_f"]
        lam, dl = to_lambda_delta(ts, tf)
        self.last_result = {**res, "num_mc_samples": num_mc_samples, "region": region,
                            "previous_region": previous, "j_inv": j_inv, "j_out": j_out, "margin": margin}
        self._last_fit = {
            "states": np.asarray(states), "labels": np.asarray(y),
            "p_success": np.asarray(p_success), "p_failure": np.asarray(p_failure),
            "p_invalid": np.asarray(p_invalid),
            "num_mc_samples": int(num_mc_samples) if num_mc_samples is not None else 1,
        }
        if groups is not None:
            self._last_fit["trajectory"] = np.asarray(groups)
        if self.verbose:
            how = "global argmin" if previous is None else (f"switched from {previous}" if region != previous else "held")
            print(f"[j_sel_hysteresis] {res['n']} threshold-fit rows, K={num_mc_samples}, w={self.w}: "
                  f"best J inv={j_inv:.4f} out={j_out:.4f}, margin={margin:.4f}; regime {region} ({how})")
            print(f"[j_sel_hysteresis] tau_s*={ts}, tau_f*={tf}: J={res['j']:.4f} MR={res['mr']:.4f} "
                  f"AR={res['ar']:.4f} FCR={res['fcr']:.4f} REC={res['rec']:.4f}")
        return ThresholdState(lambda_star=float(lam), delta_star=float(dl),
                              tau_s_star=ts, tau_f_star=tf, threshold_objective="j_sel", threshold_w=self.w,
                              threshold_region=region, threshold_j_inv=j_inv, threshold_j_out=j_out,
                              threshold_margin=float(margin))
