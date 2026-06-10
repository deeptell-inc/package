"""SVILC physics on the κ-(BEDT-TTF)₂X anisotropic triangular lattice."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class TriangularLattice:
    """Anisotropic triangular lattice with periodic boundary conditions."""
    Lx: int = 8
    Ly: int = 8
    t:  float = 1.0
    tp: float = 0.8   # diagonal hopping (kappa-(BEDT-TTF)2X t'/t ≈ 0.8)
    pbc: bool = True

    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly

    def idx(self, x: int, y: int) -> int:
        return (x % self.Lx) * self.Ly + (y % self.Ly)

    def coords(self, i: int) -> Tuple[int, int]:
        return i // self.Ly, i % self.Ly

    def bonds(self):
        """Yield (i, j, t_ij) with j > i."""
        for x in range(self.Lx):
            for y in range(self.Ly):
                i = self.idx(x, y)
                for dx, dy, w in [(1, 0, self.t), (0, 1, self.t),
                                  (1, 1, self.tp), (1, -1, self.tp)]:
                    j = self.idx(x + dx, y + dy)
                    if j > i:
                        yield i, j, w


def spin_vortex_field(lat: TriangularLattice,
                      centres: Sequence[Tuple[int, int]],
                      windings: Sequence[int]) -> np.ndarray:
    r"""χ(r) = Σ_v w_v arctan2(y-y_v, x-x_v)."""
    n = lat.n_sites
    chi = np.zeros(n)
    for i in range(n):
        x, y = lat.coords(i)
        for (cx, cy), w in zip(centres, windings):
            chi[i] += w * np.arctan2(y - cy, x - cx)
    return chi


def external_current_phase(lat: TriangularLattice,
                           src: int, snk: int,
                           amplitude: float,
                           r_cut: float | None = None) -> np.ndarray:
    """Approximate Peierls phase induced by a feed current src→snk."""
    if r_cut is None:
        r_cut = 0.5 * lat.Lx
    n = lat.n_sites
    A = np.zeros((n, n))
    sx, sy = lat.coords(src)
    dx, dy = lat.coords(snk)
    vx, vy = dx - sx, dy - sy
    norm = np.hypot(vx, vy) + 1e-6
    vx /= norm; vy /= norm
    mx, my = (sx + dx) / 2.0, (sy + dy) / 2.0
    for i, j, _ in lat.bonds():
        xi, yi = lat.coords(i)
        xj, yj = lat.coords(j)
        bx, by = (xi + xj) / 2.0, (yi + yj) / 2.0
        r = np.hypot(bx - mx, by - my)
        if r > r_cut:
            continue
        ux = xj - xi
        uy = yj - yi
        A[i, j] = amplitude * (ux * vx + uy * vy) * np.exp(-r / 2.0)
        A[j, i] = -A[i, j]
    return A


def phase_frustration_coupling(lat: TriangularLattice,
                               chi_A: np.ndarray,
                               chi_B: np.ndarray,
                               bond_cutoff: float = 3.0,
                               A_extra: np.ndarray | None = None,
                               ) -> float:
    """Gauge-invariant two-SVQ coupling proxy (Eq. 14 of the paper)."""
    V = 0.0
    for i, j, _ in lat.bonds():
        xi, yi = lat.coords(i)
        xj, yj = lat.coords(j)
        mx, my = 0.5 * (xi + xj), 0.5 * (yi + yj)
        dist = np.hypot(mx - lat.Lx / 2.0, my - lat.Ly / 2.0)
        if dist > bond_cutoff:
            continue
        dA = chi_A[j] - chi_A[i]
        dB = chi_B[j] - chi_B[i]
        if A_extra is not None:
            dA += A_extra[i, j]
            dB += A_extra[i, j]
        same = abs(np.sin(0.5 * (dA + dB)))
        opp = abs(np.sin(0.5 * (dA - dB)))
        V += (same - opp)
    return float(V)
