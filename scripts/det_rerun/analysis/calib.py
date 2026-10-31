"""p_success for the log score (KL): invalid-mass handling and one-parameter calibration.

  raw      what the eval scores: count arms (k + 0.5) / (K + 1) on n_success, else clip
  renorm   count arms (n_s + 0.5) / (n_s + n_f + 1): invalid samples carry no vote
  temp     sigmoid(logit(p) / T), T fit on val by NLL
  platt    sigmoid(a * logit(p) + b), fit on val by NLL
  iso      isotonic regression fit on val (an upper bound, not a proposal)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression

CLIP = 1e-15


def counts(p, k):
    return np.rint(np.asarray(p, float) * k)


def raw(ps, k):
    if k:
        return (counts(ps, k) + 0.5) / (k + 1.0)
    return np.clip(np.asarray(ps, float), CLIP, 1 - CLIP)


def renorm(ps, pf, k):
    if not k:
        return raw(ps, k)
    ns, nf = counts(ps, k), counts(pf, k)
    return (ns + 0.5) / (ns + nf + 1.0)


def nll(p, y):
    p = np.clip(p, CLIP, 1 - CLIP)
    y1 = (np.asarray(y) == 1).astype(float)
    return float(-np.mean(y1 * np.log(p) + (1 - y1) * np.log(1 - p)))


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temp(p_val, y_val) -> float:
    z = _logit(p_val)
    r = minimize_scalar(lambda lt: nll(_sig(z / np.exp(lt)), y_val), bounds=(-4, 4), method="bounded")
    return float(np.exp(r.x))


def fit_platt(p_val, y_val) -> tuple[float, float]:
    z = _logit(p_val)
    r = minimize(lambda ab: nll(_sig(ab[0] * z + ab[1]), y_val), x0=[1.0, 0.0], method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1])


def apply_temp(p, t):
    return _sig(_logit(p) / t)


def apply_platt(p, ab):
    return _sig(ab[0] * _logit(p) + ab[1])


def fit_iso(p_val, y_val):
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(p_val, (np.asarray(y_val) == 1).astype(float))
    return iso
