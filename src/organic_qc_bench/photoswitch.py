"""Diarylethene-photoswitch CZ-gate model (Path 2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.linalg import expm

try:                                # numpy >= 2.0
    from numpy import trapezoid as _trapezoid
except ImportError:                 # numpy < 2.0
    _trapezoid = np.trapz  # type: ignore[attr-defined]

from .core import fidelity, organic_noise
from .profiles import OrganicProfile

# Pauli & 2-qubit operators
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_SI = np.eye(2, dtype=complex)
_ZZ = np.kron(_SZ, _SZ)


@dataclass(frozen=True)
class PhotoSwitchParams:
    """Diarylethene photoswitch timing parameters."""
    tau_close_ps: float = 10.0
    tau_open_ns: float = 1.0
    J_closed_GHz: float = 0.05
    J_open_GHz: float = 5e-5
    switching_eff: float = 0.95


def _coupling_profile(t_ns: np.ndarray, params: PhotoSwitchParams,
                      switch_on_ns: float = 0.0,
                      switch_off_ns: float | None = None) -> np.ndarray:
    tau_on = params.tau_close_ps / 1000.0
    tau_off = params.tau_open_ns
    J_hi = params.J_closed_GHz * params.switching_eff
    J_lo = params.J_open_GHz
    J = np.full_like(t_ns, J_lo, dtype=float)
    for i, t in enumerate(t_ns):
        if t < switch_on_ns:
            J[i] = J_lo
        elif switch_off_ns is not None and t >= switch_off_ns:
            dt = t - switch_off_ns
            J[i] = J_lo + (J_hi - J_lo) * np.exp(-dt / tau_off)
        else:
            dt = t - switch_on_ns
            J[i] = J_lo + (J_hi - J_lo) * (1 - np.exp(-dt / tau_on))
    return J


def simulate_cz(
    profile: OrganicProfile,
    t_gate_ns: float,
    n_steps: int = 200,
    params: PhotoSwitchParams = PhotoSwitchParams(),
):
    """Simulate a diarylethene-photoswitched CZ on a ``|++⟩`` initial state.

    Returns a dict with ``fid``, ``purity_final``, and integrated phase.
    """
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi0 = np.kron(plus, plus)
    rho = np.outer(psi0, psi0.conj())
    # ideal target: U = exp(-i π/4 ZZ)
    U_id = expm(-1j * (np.pi / 4) * _ZZ)
    rho_target = U_id @ rho @ U_id.conj().T

    t = np.linspace(0, t_gate_ns, n_steps)
    dt = t[1] - t[0]
    J_traj = _coupling_profile(t, params, 0.0, t_gate_ns)
    gamma_step = profile.gamma * dt / t_gate_ns
    delta_step = profile.delta * dt / t_gate_ns

    for k in range(n_steps):
        phase_k = 2 * np.pi * J_traj[k] * dt
        U_k = expm(-1j * 0.5 * phase_k * _ZZ)
        rho = U_k @ rho @ U_k.conj().T
        rho = organic_noise(rho, gamma_step, delta_step)

    return dict(
        fid=fidelity(rho_target, rho),
        purity_final=float(np.real(np.trace(rho @ rho))),
        target_phase=np.pi / 2,
        accumulated_phase=float(_trapezoid(J_traj * 2 * np.pi, t)),
    )


def sweep_gate_time(
    profile: OrganicProfile,
    t_grid_ns: List[float] = None,
    params: PhotoSwitchParams = PhotoSwitchParams(),
):
    """Sweep ``t_gate`` and return best gate-time + corresponding fidelity."""
    if t_grid_ns is None:
        t_grid_ns = np.linspace(0.5, 20.0, 12).tolist()
    fids = []
    for t in t_grid_ns:
        res = simulate_cz(profile, t, params=params)
        fids.append(res["fid"])
    i = int(np.argmax(fids))
    return dict(
        gate_time_ns=list(t_grid_ns),
        fid=fids,
        best_gate_time_ns=t_grid_ns[i],
        best_fid=fids[i],
    )
