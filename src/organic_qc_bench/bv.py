"""Bernstein--Vazirani benchmark under organic noise channels."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .core import (
    organic_noise,
    recursive_covariant,
    cqec_recovery,
    adaptive_rounds,
)
from .profiles import OrganicProfile


def hadamard_n(n: int) -> np.ndarray:
    """n-qubit Hadamard."""
    H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    H = H1
    for _ in range(n - 1):
        H = np.kron(H, H1)
    return H


def bv_oracle_unitary(s: int, n: int) -> np.ndarray:
    r"""Diagonal phase oracle :math:`O_s = \\mathrm{diag}((-1)^{s\\cdot x})`."""
    d = 1 << n
    diag = np.empty(d, dtype=complex)
    for x in range(d):
        parity = bin(s & x).count("1") & 1
        diag[x] = -1.0 if parity else 1.0
    return np.diag(diag)


def bv_run(
    s: int,
    n: int,
    profile: OrganicProfile,
    apply_cqec: bool = False,
):
    """Run BV in density-matrix form for one hidden string.

    Returns ``(rho_final, rho_target, p_s)`` where ``p_s`` is the
    probability of measuring the correct string.
    """
    d = 1 << n
    H = hadamard_n(n)
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0
    # H^n
    rho = H @ rho @ H.conj().T
    rho = organic_noise(rho, profile.gamma, profile.delta)
    # Oracle
    O = bv_oracle_unitary(s, n)
    rho = O @ rho @ O.conj().T
    rho = organic_noise(rho, profile.gamma, profile.delta)
    # H^n
    rho = H @ rho @ H.conj().T
    rho_ideal = np.zeros((d, d), dtype=complex)
    rho_ideal[s, s] = 1.0
    rho = organic_noise(rho, profile.gamma, profile.delta)
    if apply_cqec:
        rho_cat, _, _ = recursive_covariant(rho.copy(), d, adaptive_rounds(d))
        rho = cqec_recovery(rho_ideal, rho, rho_cat)
    p_s = float(np.real(rho[s, s]))
    return rho, rho_ideal, p_s


def _wilson_ci(k: int, n: int, conf: float = 0.95):
    from scipy.stats import norm
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - (1 - conf) / 2)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(centre), float(half)


def benchmark(
    n_qubits: int,
    profiles: Mapping[str, OrganicProfile],
    n_trials: int = 100,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Monte-Carlo BV benchmark over ``n_trials`` random hidden strings.

    Returns a dict keyed by ``"<profile_name>"`` and
    ``"<profile_name>_cqec"`` containing per-profile success rates,
    Wilson 95% CIs, and the mean ideal-state population.
    """
    rng = np.random.default_rng(seed)
    d = 1 << n_qubits
    out: Dict[str, Dict[str, float]] = {}
    for prof_name, prof in profiles.items():
        for cqec in (False, True):
            key = prof_name + ("_cqec" if cqec else "")
            successes = 0
            p_s_list: list[float] = []
            for _ in range(n_trials):
                s = int(rng.integers(0, d))
                rho, _, p_s = bv_run(s, n_qubits, prof, apply_cqec=cqec)
                p_s_list.append(p_s)
                probs = np.real(np.diag(rho))
                probs = np.maximum(probs, 0)
                probs /= probs.sum()
                guess = int(rng.choice(d, p=probs))
                if guess == s:
                    successes += 1
            centre, half = _wilson_ci(successes, n_trials)
            out[key] = dict(
                profile=prof_name,
                cqec=cqec,
                n_qubits=n_qubits,
                n_trials=n_trials,
                successes=successes,
                success_rate=successes / n_trials,
                ci_half_width_95=half,
                mean_p_s=float(np.mean(p_s_list)),
            )
    return out


def classical_one_query_rate(n_qubits: int) -> float:
    """Best classical 1-query strategy: random guess with success ``2^{-n}``."""
    return 2.0 ** (-n_qubits)
