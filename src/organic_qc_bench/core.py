"""Quantum-state primitives: fidelity, channels, and CQEC.

Vendors the minimal subset of the covariant-purification CQEC package
needed to reproduce the benchmarks of Wakaura & Tanimae (2026).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# fidelity / purity / coherence
# ---------------------------------------------------------------------------

def _msqrt(A: np.ndarray) -> np.ndarray:
    """Hermitian-positive matrix square root via eigendecomposition."""
    A = (A + A.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""Uhlmann fidelity :math:`F(\rho,\sigma)=(\mathrm{Tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho})^2`."""
    sr = _msqrt(rho)
    M = sr @ sigma @ sr
    eigvals = np.linalg.eigvalsh((M + M.conj().T) / 2)
    return float(np.sum(np.sqrt(np.maximum(eigvals, 0.0))) ** 2)


def purity(rho: np.ndarray) -> float:
    r"""Purity :math:`P=\mathrm{Tr}(\rho^2)`."""
    return float(np.real(np.trace(rho @ rho)))


def l1_coherence(rho: np.ndarray) -> float:
    r""":math:`\ell_1` coherence :math:`\sum_{i\ne j}|\rho_{ij}|`."""
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))


def concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence for two-qubit states (d=4)."""
    if rho.shape != (4, 4):
        raise ValueError("concurrence is defined for two-qubit (d=4) states")
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Y = np.kron(sy, sy)
    rho_tilde = Y @ rho.conj() @ Y
    R = _msqrt(rho) @ rho_tilde @ _msqrt(rho)
    eigvals = np.sqrt(np.maximum(np.linalg.eigvalsh(R), 0.0))
    eigvals = np.sort(eigvals)[::-1]
    return float(max(0.0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]))


# ---------------------------------------------------------------------------
# noise channels
# ---------------------------------------------------------------------------

def dephasing_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    r"""Pure dephasing :math:`\rho_{ij}\to\rho_{ij}e^{-\gamma}` for :math:`i\ne j`."""
    d = rho.shape[0]
    out = rho.copy()
    if gamma <= 0:
        return out
    factor = np.exp(-gamma)
    mask = np.full((d, d), factor, dtype=complex)
    np.fill_diagonal(mask, 1.0)
    return out * mask


def depolarizing_channel(rho: np.ndarray, delta: float) -> np.ndarray:
    r"""Depolarising channel :math:`\rho\to(1-\delta)\rho+\delta\,\mathbb I/d`."""
    d = rho.shape[0]
    delta = max(0.0, min(1.0, float(delta)))
    return (1 - delta) * rho + delta * np.eye(d, dtype=complex) / d


def organic_noise(rho: np.ndarray, gamma: float, delta: float) -> np.ndarray:
    r"""The organic noise channel :math:`\mathcal{E}_\delta\circ\mathcal{D}_\gamma`.

    This is the channel used throughout the paper; Hermitises and
    re-normalises numerically.
    """
    out = dephasing_channel(rho, gamma)
    out = depolarizing_channel(out, delta)
    out = (out + out.conj().T) / 2
    tr = float(np.real(np.trace(out)))
    return out / tr if tr > 1e-15 else out


# ---------------------------------------------------------------------------
# CQEC primitives (recursive covariant purification + recovery)
# ---------------------------------------------------------------------------

def _build_swap(d: int) -> np.ndarray:
    d2 = d * d
    S = np.zeros((d2, d2), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[j * d + i, i * d + j] = 1.0
    return S


def swap_test_purify(
    rho: np.ndarray, sigma: np.ndarray, d: int
) -> Tuple[np.ndarray, float]:
    r"""Symmetric (SWAP-test) purification step.

    Project :math:`\rho\otimes\sigma` onto the symmetric subspace, trace
    out the second copy, and return the resulting purified state and
    the success probability.
    """
    SWAP = _build_swap(d)
    Pi = (np.eye(d * d, dtype=complex) + SWAP) / 2.0
    rho_sigma = np.kron(rho, sigma)
    proj = Pi @ rho_sigma @ Pi
    p = float(np.real(np.trace(proj)))
    if p < 1e-15:
        return rho.copy(), 0.0
    proj /= p
    out = np.zeros((d, d), dtype=complex)
    # partial trace over the second copy
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[i, j] += proj[i * d + k, j * d + k]
    return out, p


def recursive_covariant(
    rho_noisy: np.ndarray, d: int, n_rounds: int
) -> Tuple[np.ndarray, int, float]:
    """Apply ``n_rounds`` of recursive symmetric purification.

    Returns ``(rho_cat, n_copies_used, p_total)``.
    """
    rho = rho_noisy.copy()
    p_total = 1.0
    for _ in range(n_rounds):
        rho, p = swap_test_purify(rho, rho, d)
        p_total *= p
    return rho, 2 ** n_rounds, p_total


def cqec_recovery(
    rho_target: np.ndarray, rho_noisy: np.ndarray, rho_cat: np.ndarray
) -> np.ndarray:
    """Simplified CQEC recovery towards the target state.

    Implements the catalyst-dependent recovery map of the companion
    3-LQBH preprint:

    .. math::
       \\mathcal{R}(\\rho_{\\rm alg})_{ij} = \\rho_{\\rm alg,ij}
        + \\eta_{ij}\\,|\\rho_{\\rm target,ij}|\\,e^{i\\arg(\\rho_{\\rm target,ij})}

    with ``η_ij = 1 − exp(−|ρ_cat,ij| d P)``.
    """
    d = rho_target.shape[0]
    P = purity(rho_cat)
    rho_rec = rho_noisy.copy()
    tol = 1e-10
    for i in range(d):
        for j in range(i + 1, d):
            if abs(rho_target[i, j]) > tol and abs(rho_cat[i, j]) > tol:
                cat_coh = abs(rho_cat[i, j])
                noisy_coh = abs(rho_noisy[i, j])
                if noisy_coh > 1e-15:
                    phase = np.angle(rho_target[i, j])
                    mag_t = abs(rho_target[i, j])
                    eta = 1.0 - np.exp(-cat_coh * d * P)
                    mag_r = noisy_coh + eta * max(0.0, mag_t - noisy_coh)
                    rho_rec[i, j] = mag_r * np.exp(1j * phase)
                    rho_rec[j, i] = rho_rec[i, j].conj()
    # project back to a valid density matrix
    eigvals, eigvecs = np.linalg.eigh((rho_rec + rho_rec.conj().T) / 2)
    eigvals = np.maximum(eigvals, 0.0)
    rho_rec = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    rho_rec /= np.trace(rho_rec)
    return rho_rec


def adaptive_rounds(d: int) -> int:
    """Recommended ``n_rounds`` for ``swap_test_purify``.

    Cost scales as :math:`\\mathcal O(d^6)` per round, so we use 2 rounds
    for :math:`d\\le 16` and 1 round otherwise.
    """
    return 2 if d <= 16 else 1
