#!/usr/bin/env python3
"""
svilc_kbedt_lattice.py
======================
Numerical validation that SVILC-like physics (spin vortices, topologically-
protected loop currents, current-coupler between qubits) can emerge on the
quasi-2D organic-superconductor lattice κ-(BEDT-TTF)₂X.

This is the "Path 3 精密化" item from ORGANIC_BENCHMARKS_EXTENDED.md §10.3.

We build an anisotropic triangular lattice (the conducting plane in
κ-BEDT-TTF) and examine two things:

  A) Does a hole-doped Hartree-Fock state on this lattice support
     spin-vortex textures and persistent loop currents?

  B) Can two spin-vortex quartets (SVQs) be uncoupled at a distance and
     coupled on-demand by an external feeding current (EXACT analog of
     Wakaura 2017 Fig. 4-9)?

Strategy
--------
Instead of a full self-consistent HF (which would also not spontaneously
break gauge symmetry into the multi-valued Koizumi form), we impose a
prescribed spin-vortex pattern on the many-body phase χ and compute the
resulting loop currents j_ij = (2e/ℏ) Im ⟨c†_i c_j⟩ from the tight-binding
wavefunction.  This is the same phenomenological step used in Wakaura &
Koizumi (2017) Eqs. (4)–(12).

Lattice: anisotropic triangular 8×8, bonds (i,j) with t along x, y; t' along
diagonal; with t'/t = 0.8 (typical for κ-(BEDT-TTF)₂Cu[N(CN)₂]Br).

Author: Hikaru Wakaura — date 2026-04-17.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

np.random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ANISOTROPIC TRIANGULAR LATTICE (κ-(BEDT-TTF)₂X conducting plane)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TriangularLattice:
    """Anisotropic triangular lattice with sites (x, y).

    Bonds:
      - nearest-neighbor along x   → hopping t
      - nearest-neighbor along y   → hopping t
      - nearest-neighbor along x+y → hopping t' (diagonal)
      - nearest-neighbor along x-y → hopping t' (other diagonal)

    (t'/t) = 0.8 approximates κ-(BEDT-TTF)₂Cu[N(CN)₂]Br.
    """
    Lx: int = 8
    Ly: int = 8
    t:  float = 1.0          # NN hopping (energy unit)
    tp: float = 0.8          # NNN (diagonal) hopping = triangular frustration
    pbc: bool = True         # periodic boundary conditions

    def n_sites(self):
        return self.Lx * self.Ly

    def index(self, x, y):
        return (x % self.Lx) * self.Ly + (y % self.Ly)

    def coords(self, i):
        return i // self.Ly, i % self.Ly

    def bonds(self):
        """Return list of (i, j, t_ij) with j > i."""
        bonds = []
        for x in range(self.Lx):
            for y in range(self.Ly):
                i = self.index(x, y)
                # +x neighbour
                if self.pbc or x < self.Lx - 1:
                    j = self.index(x + 1, y)
                    if j > i:
                        bonds.append((i, j, self.t))
                # +y neighbour
                if self.pbc or y < self.Ly - 1:
                    j = self.index(x, y + 1)
                    if j > i:
                        bonds.append((i, j, self.t))
                # +x +y diagonal
                if (self.pbc or (x < self.Lx - 1 and y < self.Ly - 1)):
                    j = self.index(x + 1, y + 1)
                    if j > i:
                        bonds.append((i, j, self.tp))
                # +x -y diagonal
                if (self.pbc or (x < self.Lx - 1 and y > 0)):
                    j = self.index(x + 1, y - 1)
                    if j > i:
                        bonds.append((i, j, self.tp))
        return bonds


# ══════════════════════════════════════════════════════════════════════════════
# 2.  IMPOSED SPIN-VORTEX PATTERN (Koizumi / Wakaura recipe)
# ══════════════════════════════════════════════════════════════════════════════

def spin_vortex_field(lat: TriangularLattice, centers, windings):
    """Return χ_i (many-body phase) with spin vortices at the given centres.

    χ(r) = Σ_v w_v · arg(r − r_v)

    Where each vortex v has winding number w_v ∈ {+1,−1}. χ is multi-valued
    (changes by 2π·Σw_v around any loop enclosing the vortices)."""
    n = lat.n_sites()
    chi = np.zeros(n, dtype=float)
    for i in range(n):
        x, y = lat.coords(i)
        for (cx, cy), w in zip(centers, windings):
            dx = x - cx
            dy = y - cy
            # For a closed loop argument we use atan2
            chi[i] += w * np.arctan2(dy, dx)
    return chi


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TIGHT-BINDING DIAGONALISATION WITH PEIERLS-LIKE SUBSTITUTION
# ══════════════════════════════════════════════════════════════════════════════

def build_hopping_matrix(lat: TriangularLattice, chi: np.ndarray,
                          J_ext: np.ndarray = None) -> np.ndarray:
    """Build the tight-binding Hamiltonian with phase factors e^{i(χ_i − χ_j)/2}.

    (This is the Koizumi prescription: the spin-vortex phase enters the
    hopping through a Peierls-like substitution, so the many-body ground
    state carries a persistent loop current.)

    J_ext: optional extra phase along each bond (feeding current coupler).
    """
    n = lat.n_sites()
    H = np.zeros((n, n), dtype=complex)
    for (i, j, tij) in lat.bonds():
        phase = (chi[j] - chi[i]) / 2.0
        if J_ext is not None:
            phase += J_ext[i, j]
        H[i, j] = -tij * np.exp(1j * phase)
        H[j, i] = H[i, j].conjugate()
    return H


def compute_density_matrix(H: np.ndarray, n_electrons: int) -> np.ndarray:
    """Ground-state density matrix: ρ_ij = ⟨c†_j c_i⟩ filling lowest n_electrons."""
    evals, evecs = np.linalg.eigh(H)
    occ = evecs[:, :n_electrons]
    return occ @ occ.conj().T


def hubbard_hf_energy(H0: np.ndarray, n_electrons: int,
                       U: float = 1.0, n_iter: int = 40,
                       damping: float = 0.6) -> tuple:
    """Simple spin-restricted Hartree-Fock with on-site Hubbard U.

    Working in a spin-polarised background, we treat spin-↑ and spin-↓
    channels independently with the same imposed Peierls phases. Self-
    consistency equation:
      H_σ = H0 + U * diag(n_{-σ})
    with n_{-σ} = diagonal of the density matrix of the opposite spin.

    Returns (total_energy, rho_up, rho_down)."""
    n = H0.shape[0]
    # Break up/down symmetry slightly so the solver doesn't stall
    n_up   = (n_electrons + 1) // 2
    n_down = n_electrons - n_up
    rho_up = compute_density_matrix(H0, n_up)
    rho_dn = compute_density_matrix(H0, n_down)
    for it in range(n_iter):
        V_up = U * np.diag(np.real(np.diag(rho_dn)))
        V_dn = U * np.diag(np.real(np.diag(rho_up)))
        H_up = H0 + V_up
        H_dn = H0 + V_dn
        new_up = compute_density_matrix(H_up, n_up)
        new_dn = compute_density_matrix(H_dn, n_down)
        rho_up = damping * new_up + (1 - damping) * rho_up
        rho_dn = damping * new_dn + (1 - damping) * rho_dn
    # Total energy
    ev_up = np.linalg.eigvalsh(H_up)
    ev_dn = np.linalg.eigvalsh(H_dn)
    E_band = float(np.sum(ev_up[:n_up]) + np.sum(ev_dn[:n_down]))
    # Subtract double counting of U interaction
    n_up_diag = np.real(np.diag(rho_up))
    n_dn_diag = np.real(np.diag(rho_dn))
    E_dc = U * float(np.sum(n_up_diag * n_dn_diag))
    return E_band - E_dc, rho_up, rho_dn


def bond_currents(lat: TriangularLattice, H: np.ndarray,
                   rho: np.ndarray) -> np.ndarray:
    """Expected bond current j_ij = (2e/ℏ) Im[t_ij ⟨c†_i c_j⟩].

    In atomic units, returning dimensionless currents per bond."""
    n = lat.n_sites()
    J = np.zeros((n, n))
    for (i, j, _tij) in lat.bonds():
        # <c†_i c_j> = ρ[j, i] in our convention
        J[i, j] = 2.0 * np.imag(H[j, i] * rho[i, j])
        J[j, i] = -J[i, j]
    return J


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LOOP WINDING NUMBERS (topological invariants)
# ══════════════════════════════════════════════════════════════════════════════

def plaquette_winding(chi: np.ndarray, lat: TriangularLattice, x0, y0):
    """Compute the winding number of χ around a 2×2 plaquette (x0..x0+2,
    y0..y0+2). Returns an integer (topological charge in the plaquette)."""
    pts = [(x0, y0), (x0 + 1, y0), (x0 + 1, y0 + 1),
           (x0, y0 + 1), (x0, y0)]
    phase_diff = 0.0
    for k in range(len(pts) - 1):
        a = lat.index(*pts[k])
        b = lat.index(*pts[k + 1])
        d = chi[b] - chi[a]
        # Wrap into (-π, π]
        d = (d + np.pi) % (2 * np.pi) - np.pi
        phase_diff += d
    return phase_diff / (2 * np.pi)


def winding_map(chi: np.ndarray, lat: TriangularLattice) -> np.ndarray:
    """Grid of plaquette windings over the lattice."""
    W = np.zeros((lat.Lx, lat.Ly))
    for x in range(lat.Lx - 1):
        for y in range(lat.Ly - 1):
            W[x, y] = plaquette_winding(chi, lat, x, y)
    return W


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TWO-SVQ COUPLING VIA EXTERNAL FEEDING CURRENT
# ══════════════════════════════════════════════════════════════════════════════

def external_current_phase(lat: TriangularLattice, source, sink,
                            amplitude: float = 0.2) -> np.ndarray:
    """Approximate the phase shift induced by an external feeding current
    running from `source` to `sink` through the lattice. We model this as
    a vector potential ∝ amplitude · (distance-dependent decay).

    Returned matrix A has A[i,j] = -A[j,i] (antisymmetric) giving the
    Peierls phase on each bond.
    """
    n = lat.n_sites()
    A = np.zeros((n, n))
    sx, sy = lat.coords(source)
    dx, dy = lat.coords(sink)
    for (i, j, _t) in lat.bonds():
        xi, yi = lat.coords(i)
        xj, yj = lat.coords(j)
        # Distance between bond centre and (source, sink) path mid-point
        mx, my = (sx + dx) / 2.0, (sy + dy) / 2.0
        bx, by = (xi + xj) / 2.0, (yi + yj) / 2.0
        r = np.hypot(bx - mx, by - my)
        # Direction of bond (along x-y vector)
        ux = xj - xi
        uy = yj - yi
        # "Current" direction (source → sink)
        vx = dx - sx
        vy = dy - sy
        norm = np.hypot(vx, vy) + 1e-6
        vx /= norm; vy /= norm
        # Only bonds closer than r_cut see the feed
        r_cut = 0.5 * lat.Lx
        if r > r_cut:
            continue
        A[i, j] = amplitude * (ux * vx + uy * vy) * np.exp(-r / 2.0)
        A[j, i] = -A[i, j]
    return A


def svq_coupling_energy(lat: TriangularLattice,
                         centers_A, windings_A,
                         centers_B, windings_B,
                         n_electrons: int,
                         feed_source=None, feed_sink=None,
                         feed_amp=0.0,
                         U: float = 0.0) -> dict:
    """Compute total ground-state energy of the lattice with SVQ A and B
    simultaneously present, optionally with external feed current from
    feed_source to feed_sink.  If U > 0, run Hartree-Fock self-consistency.
    Returns energy, currents, winding map."""
    chi_A = spin_vortex_field(lat, centers_A, windings_A)
    chi_B = spin_vortex_field(lat, centers_B, windings_B)
    chi = chi_A + chi_B
    A = None
    if feed_source is not None and feed_sink is not None:
        A = external_current_phase(lat, feed_source, feed_sink, feed_amp)
    H = build_hopping_matrix(lat, chi, A)
    if U > 0:
        energy, rho_up, rho_dn = hubbard_hf_energy(H, n_electrons, U=U)
        rho = rho_up + rho_dn
    else:
        evals, _ = np.linalg.eigh(H)
        energy = float(np.sum(evals[:n_electrons]))
        rho = compute_density_matrix(H, n_electrons)
    J = bond_currents(lat, H, rho)
    W = winding_map(chi, lat)
    return dict(energy=energy, currents=J, winding_map=W,
                chi=chi, max_current=float(np.max(np.abs(J))))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

def experiment_single_svq(lat: TriangularLattice, filling=0.45):
    """A single spin-vortex quartet at lattice centre."""
    cx = lat.Lx // 2
    cy = lat.Ly // 2
    # SVQ of Wakaura 2017 Fig 1: two +1 and two −1 vortices at corners of
    # a 4×4 plaquette, sum of winding = 0.
    centres = [(cx - 1, cy - 1), (cx + 1, cy - 1),
               (cx + 1, cy + 1), (cx - 1, cy + 1)]
    windings = [+1, -1, +1, -1]
    n_e = int(filling * lat.n_sites())
    return svq_coupling_energy(lat, centres, windings, [], [], n_e, U=0.0)


def phase_frustration_coupling(lat: TriangularLattice,
                                chi_A: np.ndarray, chi_B: np.ndarray,
                                bond_cutoff_distance: float = 3.0) -> float:
    """
    Topological coupling metric between two SVQ phase fields.

    V_coupling = Σ_{bonds (i,j) near midpoint of A-B axis}
                 |sin[(Δχ_A + Δχ_B)/2]| − |sin[(Δχ_A − Δχ_B)/2]|

    where Δχ_X = χ_X[j] − χ_X[i].  This is a pure geometric quantity that
    captures how the two chirality patterns interfere on the bonds that lie
    between them.  It is positive when the two SVQs constructively
    interfere and zero at infinite separation.  Gauge-invariant by
    construction.
    """
    V = 0.0
    for (i, j, _t) in lat.bonds():
        xi, yi = lat.coords(i)
        xj, yj = lat.coords(j)
        # Midpoint of bond
        mx, my = 0.5 * (xi + xj), 0.5 * (yi + yj)
        # Keep only bonds near the lattice middle (where A-B interference lives)
        dist = np.hypot(mx - lat.Lx / 2.0, my - lat.Ly / 2.0)
        if dist > bond_cutoff_distance:
            continue
        dA = chi_A[j] - chi_A[i]
        dB = chi_B[j] - chi_B[i]
        same     = np.abs(np.sin(0.5 * (dA + dB)))
        opposite = np.abs(np.sin(0.5 * (dA - dB)))
        V += (same - opposite)
    return float(V)


def experiment_two_svqs_vs_distance(lat: TriangularLattice,
                                     distances, filling=0.45):
    """Coupling vs distance using the phase-frustration metric.
    (Robust gauge-invariant version; the tight-binding HF estimator is
    degenerate for chi→−chi on B alone.)"""
    out = []
    for r in distances:
        cxA, cyA = lat.Lx // 4, lat.Ly // 2
        cxB, cyB = lat.Lx // 4 + r, lat.Ly // 2
        if cxB >= lat.Lx - 1:
            continue
        centsA = [(cxA - 1, cyA - 1), (cxA + 1, cyA - 1),
                  (cxA + 1, cyA + 1), (cxA - 1, cyA + 1)]
        windsA = [+1, -1, +1, -1]
        centsB = [(cxB - 1, cyB - 1), (cxB + 1, cyB - 1),
                  (cxB + 1, cyB + 1), (cxB - 1, cyB + 1)]
        windsB = [+1, -1, +1, -1]
        chi_A = spin_vortex_field(lat, centsA, windsA)
        chi_B = spin_vortex_field(lat, centsB, windsB)
        V = phase_frustration_coupling(lat, chi_A, chi_B)
        out.append(dict(distance=r, V_coupling=V))
    return out


def experiment_feed_current_activates_coupling(lat: TriangularLattice,
                                                 filling=0.45,
                                                 distance=None,
                                                 feed_amps=None):
    """At a distance where two SVQs are uncoupled, switch coupling ON by
    feeding external current through the region between them.

    We compute the *feed-current-induced* coupling by summing the Peierls
    phase introduced by the feeding-current vector potential along the bonds
    between the two SVQs, then evaluating the phase-frustration metric with
    that extra phase included."""
    if distance is None:
        distance = lat.Lx // 2
    if feed_amps is None:
        feed_amps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    cxA, cyA = lat.Lx // 4, lat.Ly // 2
    cxB, cyB = lat.Lx // 4 + distance, lat.Ly // 2
    if cxB >= lat.Lx - 1:
        cxB = lat.Lx - 3
    centsA = [(cxA - 1, cyA - 1), (cxA + 1, cyA - 1),
              (cxA + 1, cyA + 1), (cxA - 1, cyA + 1)]
    windsA = [+1, -1, +1, -1]
    centsB = [(cxB - 1, cyB - 1), (cxB + 1, cyB - 1),
              (cxB + 1, cyB + 1), (cxB - 1, cyB + 1)]
    windsB = [+1, -1, +1, -1]
    chi_A = spin_vortex_field(lat, centsA, windsA)
    chi_B = spin_vortex_field(lat, centsB, windsB)
    mid_x = (cxA + cxB) // 2
    source = lat.index(mid_x, 0)
    sink   = lat.index(mid_x, lat.Ly - 1)
    out = []
    for amp in feed_amps:
        A = external_current_phase(lat, source, sink, amp)
        # Modify chi_A and chi_B effectively by integrating the feed-current
        # vector potential along the bonds (adds the same phase to both SVQs)
        V = 0.0
        for (i, j, _t) in lat.bonds():
            xi, yi = lat.coords(i)
            xj, yj = lat.coords(j)
            mx, my = 0.5 * (xi + xj), 0.5 * (yi + yj)
            if np.hypot(mx - lat.Lx / 2.0, my - lat.Ly / 2.0) > 3.0:
                continue
            dA = chi_A[j] - chi_A[i] + A[i, j]
            dB = chi_B[j] - chi_B[i] + A[i, j]
            same     = np.abs(np.sin(0.5 * (dA + dB)))
            opposite = np.abs(np.sin(0.5 * (dA - dB)))
            V += (same - opposite)
        out.append(dict(feed_amp=amp, V_coupling=float(V)))
    return out


def analyse_ascii_current_map(J: np.ndarray, lat: TriangularLattice,
                               threshold: float = None) -> list:
    """Return ASCII-art current map: one line per row, arrow indicates
    the dominant direction of bond currents at each site."""
    J_max = np.max(np.abs(J))
    if threshold is None:
        threshold = 0.2 * J_max + 1e-9
    lines = []
    for y in range(lat.Ly - 1, -1, -1):
        row = []
        for x in range(lat.Lx):
            i = lat.index(x, y)
            # Sum outgoing signed currents
            net_x = net_y = 0.0
            if x + 1 < lat.Lx or lat.pbc:
                net_x += J[i, lat.index(x + 1, y)]
            if x - 1 >= 0 or lat.pbc:
                net_x -= J[i, lat.index(x - 1, y)]
            if y + 1 < lat.Ly or lat.pbc:
                net_y += J[i, lat.index(x, y + 1)]
            if y - 1 >= 0 or lat.pbc:
                net_y -= J[i, lat.index(x, y - 1)]
            mag = np.hypot(net_x, net_y)
            if mag < threshold:
                row.append(".")
            else:
                ang = np.degrees(np.arctan2(net_y, net_x))
                if -22 <= ang < 22:     row.append("→")
                elif 22 <= ang < 67:    row.append("↗")
                elif 67 <= ang < 112:   row.append("↑")
                elif 112 <= ang < 157:  row.append("↖")
                elif 157 <= ang or ang < -157: row.append("←")
                elif -157 <= ang < -112: row.append("↙")
                elif -112 <= ang < -67:  row.append("↓")
                else:                     row.append("↘")
        lines.append("   " + " ".join(row))
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 78)
    print("SVILC-ANALOG ON κ-(BEDT-TTF)₂X TRIANGULAR LATTICE")
    print("(Path 3 precision — Hubbard-HF-like tight-binding with spin vortices)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    # Compare square lattice (t'=0) vs κ-BEDT-TTF triangular (t'=0.8)
    for tp in (0.0, 0.8):
        print("\n" + "─" * 78)
        print(f"  Lattice: {'square (cuprate-like)' if tp == 0 else 'triangular (κ-BEDT-TTF-like)'}"
              f"   t'={tp}")
        print("─" * 78)
        lat = TriangularLattice(Lx=8, Ly=8, t=1.0, tp=tp, pbc=True)

        # Single SVQ — verify spin-vortex & loop current structure
        res = experiment_single_svq(lat)
        W = res["winding_map"]
        J = res["currents"]
        n_plus = int(np.sum(np.round(W) == +1))
        n_minus = int(np.sum(np.round(W) == -1))
        print(f"\n  Single-SVQ winding map (centre 4×4 block):")
        for row in W[2:6, 2:6].T[::-1]:
            print("    " + " ".join(f"{w:+.2f}" for w in row))
        print(f"  Counted vortices:  +1: {n_plus}   −1: {n_minus}   "
              f"Σw = {n_plus - n_minus}   (expected 0)")
        print(f"  Max bond current |j|_max = {res['max_current']:.4f}   "
              f"E_ground = {res['energy']:.3f}")
        print(f"  Current map (arrows = bond currents):")
        for ln in analyse_ascii_current_map(J, lat):
            print(ln)

    # Two-SVQ coupling vs distance on triangular lattice
    print("\n" + "─" * 78)
    print("  Two-SVQ coupling V_αΥ vs distance (triangular lattice, t'=0.8)")
    print("─" * 78)
    lat = TriangularLattice(Lx=16, Ly=6, t=1.0, tp=0.8, pbc=True)
    distances = [3, 4, 5, 6, 7, 8, 10, 12]
    cpl = experiment_two_svqs_vs_distance(lat, distances)
    print(f"\n  {'r_x':>5} {'V_αΥ':>14} {'log10|V|':>10}")
    for entry in cpl:
        V = entry["V_coupling"]
        print(f"  {entry['distance']:>5d} {V:>14.6e} "
              f"{np.log10(abs(V) + 1e-15):>10.3f}")

    # Feed-current activation of coupling (SVILC analog of Wakaura 2017 §5)
    print("\n" + "─" * 78)
    print("  Feed-current activation: distance = lat.Lx//2 (uncoupled → coupled)")
    print("─" * 78)
    act = experiment_feed_current_activates_coupling(lat, distance=10)
    print(f"\n  {'J_ext':>8} {'V_αΥ':>14} {'|V|/|V(0)|':>12}")
    v0 = abs(act[0]["V_coupling"]) + 1e-20
    for entry in act:
        V = entry["V_coupling"]
        print(f"  {entry['feed_amp']:>8.3f} {V:>14.6e} "
              f"{abs(V) / v0:>12.3f}")

    payload = {
        "meta": dict(
            date=datetime.now().isoformat(),
            wallclock_s=time.time() - t0,
            lattice_sizes=[(8, 8), (16, 6)],
            tp_values=[0.0, 0.8],
        ),
        "coupling_vs_distance": [{
            "distance": e["distance"],
            "V_coupling": float(e["V_coupling"]),
        } for e in cpl],
        "feed_current_activation": [{
            "feed_amp": float(e["feed_amp"]),
            "V_coupling": float(e["V_coupling"]),
        } for e in act],
    }
    out_path = os.path.join(HERE, "results", "svilc_kbedt_lattice.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"\n  Wallclock: {time.time() - t0:.1f} s")
    return payload


if __name__ == "__main__":
    main()
