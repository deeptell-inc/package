"""Algorithm state-preparation factories.

Each ``make_*`` returns ``(rho, d)`` — an ideal target density matrix and
its dimension. The same conventions as the manuscript benchmarks.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def make_qkan(seed: int = 42):
    """QKAN: d=4 Chebyshev-amplitude pure state."""
    amps = np.array([1.0, 0.5, -0.5, 0.0], dtype=complex)
    amps /= np.linalg.norm(amps)
    return np.outer(amps, amps.conj()), 4


def make_qdrift(seed: int = 42):
    """qDRIFT (random product formula on a 3-qubit Heisenberg Hamiltonian)."""
    d = 8
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    J, h = 1.0, 0.5
    terms = []
    coeffs = []
    for p0, p1 in [(0, 1), (1, 2)]:
        for pauli in [sx, sy, sz]:
            ops = [I2, I2, I2]
            ops[p0] = pauli; ops[p1] = pauli
            terms.append(J * np.kron(np.kron(ops[0], ops[1]), ops[2]))
            coeffs.append(J)
    for q in range(3):
        ops = [I2, I2, I2]; ops[q] = sz
        terms.append(h * np.kron(np.kron(ops[0], ops[1]), ops[2]))
        coeffs.append(h)

    total = sum(coeffs)
    probs = [c / total for c in coeffs]
    rng = np.random.default_rng(seed)
    n_gates = 80
    tau = 1.0 * total / n_gates
    U = np.eye(d, dtype=complex)
    for _ in range(n_gates):
        idx = rng.choice(len(terms), p=probs)
        U = expm(-1j * tau * terms[idx] / coeffs[idx] * coeffs[idx]) @ U
    psi = U[:, 0]
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj()), d


def make_cfqpe(seed: int = 42):
    """Control-free QPE: d=16 random-phase amplitude state."""
    d = 16
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, d)
    weights = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25,
                        0.22, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05])
    amps = np.exp(1j * phases) * weights
    amps /= np.linalg.norm(amps)
    return np.outer(amps, amps.conj()), d


def make_regev(seed: int = 42):
    """Regev factoring (discrete Gaussian-like) target: d=64."""
    d = 64
    grid = np.arange(d)
    sigma = d / 4.0
    amps = np.exp(-grid ** 2 / (2 * sigma ** 2))
    amps = amps * np.exp(2j * np.pi * grid * 9 / d)
    amps /= np.linalg.norm(amps)
    return np.outer(amps, amps.conj()), d


def make_bell():
    """Bell state |Phi+⟩."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = psi[3] = 1.0 / np.sqrt(2)
    return np.outer(psi, psi.conj()), 4


def make_ghz(n_qubits: int = 3):
    """n-qubit GHZ state."""
    d = 1 << n_qubits
    psi = np.zeros(d, dtype=complex)
    psi[0] = psi[-1] = 1.0 / np.sqrt(2)
    return np.outer(psi, psi.conj()), d
