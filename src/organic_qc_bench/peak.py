"""CQEC fidelity-gain peak vs state dimension."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .core import (
    fidelity,
    organic_noise,
    recursive_covariant,
    cqec_recovery,
    adaptive_rounds,
)


def random_pure_state(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def cqec_recover(rho_target: np.ndarray, rho_noisy: np.ndarray, d: int):
    rho_cat, _, _ = recursive_covariant(rho_noisy.copy(), d, adaptive_rounds(d))
    return cqec_recovery(rho_target, rho_noisy, rho_cat)


def sweep_gamma(
    d: int,
    gammas: np.ndarray,
    delta_factor: float = 0.5,
    n_trials: int = 10,
) -> Dict[str, List[float]]:
    """Sweep ``gamma`` and return mean fidelity & CQEC gain.

    Random pure states are used as the reference.  Returns dicts with keys
    ``gamma``, ``fid_noisy``, ``fid_cqec``, ``delta_f``.
    """
    out: Dict[str, List[float]] = {
        "gamma": list(map(float, gammas)),
        "fid_noisy": [], "fid_cqec": [], "delta_f": [],
    }
    for g in gammas:
        delta = min(1.0, delta_factor * g)
        fn_list, fc_list = [], []
        for t in range(n_trials):
            rho_t = random_pure_state(d, seed=42 + d + 1000 * t)
            rho_n = organic_noise(rho_t, g, delta)
            rho_c = cqec_recover(rho_t, rho_n, d)
            fn_list.append(fidelity(rho_t, rho_n))
            fc_list.append(fidelity(rho_t, rho_c))
        fn = float(np.mean(fn_list))
        fc = float(np.mean(fc_list))
        out["fid_noisy"].append(fn)
        out["fid_cqec"].append(fc)
        out["delta_f"].append(fc - fn)
    return out


def find_peak(sweep: Dict[str, List[float]]):
    """Return ``(gamma_peak, delta_f_max)`` from a sweep result."""
    df = np.asarray(sweep["delta_f"])
    g  = np.asarray(sweep["gamma"])
    idx = int(np.argmax(df))
    return float(g[idx]), float(df[idx])


def scaling(
    dims: List[int] = (2, 4, 8, 16, 32, 64),
    gammas: np.ndarray | None = None,
    n_trials: int | None = None,
):
    """Run ``sweep_gamma`` over multiple dimensions and locate the peak.

    Returns a list of ``{d, gamma_peak, delta_f_max, n_trials}`` dicts.
    """
    if gammas is None:
        gammas = np.logspace(-2, 0.7, 24)
    summary = []
    for d in dims:
        n_tr = n_trials if n_trials is not None else (10 if d <= 16 else 4)
        sweep = sweep_gamma(d, gammas, n_trials=n_tr)
        gpk, dfmax = find_peak(sweep)
        summary.append(dict(d=d, gamma_peak=gpk, delta_f_max=dfmax,
                            n_trials=n_tr))
    return summary
