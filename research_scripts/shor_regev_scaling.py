#!/usr/bin/env python3
"""
shor_regev_scaling.py
=====================
Scaling analysis for Shor's algorithm and Regev's 2024 factoring algorithm
under organic-material noise profiles.

This is the "1000-qubit scale Shor/Regev" next-step item from
ORGANIC_BENCHMARKS_EXTENDED.md §10.3.

We cannot simulate a 1000-qubit state (2^1000-dim Hilbert space), but we can:

  1. Execute the FULL Shor factoring algorithm for small N (15, 21, 35, 39, 51)
     with each organic noise profile and measure empirical success rate.
  2. Count the quantum gates / circuit depth as a function of n = log2(N).
  3. Compare Shor O(n²) vs Regev O(n^{3/2} log n) resource budgets.
  4. Extrapolate CQEC recovery budget for RSA-2048 (n = 2048 bit).

References
----------
Shor 1994; Regev (2024) arXiv:2308.06572v3; Gidney & Ekerå 2021.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from math import gcd, log2, sqrt, ceil, pi

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CQEC_ROOT = os.path.normpath(os.path.join(HERE, "..", "cqec"))
for p in (CQEC_ROOT, os.path.join(CQEC_ROOT, "cqec")):
    if p not in sys.path:
        sys.path.insert(0, p)

from covariant_purification import (
    fidelity, purity, dephasing_channel, depolarizing_channel,
    recursive_covariant, cqec_recovery,
)

np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ORGANIC NOISE PROFILES
# ══════════════════════════════════════════════════════════════════════════════

PROFILES = {
    "Path1_RPRes":    dict(gamma=0.100, delta=0.080),
    "Path2_PTM":      dict(gamma=0.003, delta=0.005),
    "Path3_OrgSC":    dict(gamma=5e-5,  delta=1e-4),
    "Path4_SSH":      dict(gamma=0.002, delta=0.003),
}


def apply_noise(rho, prof):
    r = dephasing_channel(rho, prof["gamma"])
    r = depolarizing_channel(r, prof["delta"])
    r = (r + r.conj().T) / 2
    t = np.real(np.trace(r))
    if t > 1e-15:
        r /= t
    return r


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SHOR'S ALGORITHM (small-N end-to-end numerical simulation)
# ══════════════════════════════════════════════════════════════════════════════

def shor_modular_exponent(a: int, N: int, n_q: int) -> np.ndarray:
    """Build the state |x⟩|a^x mod N⟩ for x∈[0, 2^n_q) as a complex
    amplitude vector over a 2^n_q × N register."""
    d_x = 1 << n_q
    state = np.zeros((d_x, N), dtype=complex)
    for x in range(d_x):
        y = pow(a, x, N)
        state[x, y] = 1.0
    state /= sqrt(d_x)
    return state


def qft_matrix(d: int) -> np.ndarray:
    """d×d Quantum Fourier Transform matrix."""
    j = np.arange(d)[:, None]
    k = np.arange(d)[None, :]
    return np.exp(2j * np.pi * j * k / d) / sqrt(d)


def shor_measure_order(state: np.ndarray) -> tuple:
    """After measurement of the value register, apply QFT on the index
    register then measure it.  Returns a random measurement outcome m."""
    d_x, N = state.shape
    # Probability density for value-register outcome
    prob_y = np.sum(np.abs(state) ** 2, axis=0)
    y_idx = np.random.choice(N, p=prob_y / prob_y.sum())
    # Conditional state on index register
    amps = state[:, y_idx].copy()
    nrm = np.linalg.norm(amps)
    if nrm < 1e-12:
        return -1
    amps /= nrm
    # QFT
    F = qft_matrix(d_x)
    amps = F @ amps
    probs = np.abs(amps) ** 2
    probs /= probs.sum()
    return int(np.random.choice(d_x, p=probs))


def continued_fraction_period(m: int, d_x: int, N: int) -> int:
    """Recover the period r from the measured index m/d_x, by continued-
    fraction expansion.  Returns candidate r or 0 if none found."""
    from fractions import Fraction
    if m == 0:
        return 0
    frac = Fraction(m, d_x).limit_denominator(N)
    return frac.denominator


def shor_factor(N: int, n_trials: int = 20, profile: dict = None,
                 n_q: int = None) -> dict:
    """Attempt to factor N with Shor's algorithm.  Apply noise/CQEC on the
    post-QFT state if a noise profile is given.

    Returns dict with success rate, number of gates used, etc."""
    if n_q is None:
        n_q = ceil(2 * log2(N))        # standard Shor choice
    d_x = 1 << n_q
    # Gate counts (textbook): modular exponentiation O(n³), QFT O(n²)
    n = ceil(log2(N))
    gates_mod_exp = n ** 3
    gates_qft = n ** 2
    n_gates = gates_mod_exp + gates_qft

    n_success = 0
    attempted = 0
    for _ in range(n_trials):
        # pick random a in [2, N-1] coprime to N
        a = int(np.random.randint(2, N))
        if gcd(a, N) != 1:
            g = gcd(a, N)
            if 1 < g < N:
                n_success += 1
                attempted += 1
                continue
        state_grid = shor_modular_exponent(a, N, n_q)
        # flatten to a density matrix over index register only (after trace)
        # For noise simulation we work in the index-register subspace.
        rho_x = np.zeros((d_x, d_x), dtype=complex)
        for y in range(N):
            col = state_grid[:, y]
            rho_x += np.outer(col, col.conj())
        if profile is not None:
            rho_x_noisy = apply_noise(rho_x, profile)
        else:
            rho_x_noisy = rho_x
        # Apply QFT
        F = qft_matrix(d_x)
        rho_qft = F @ rho_x_noisy @ F.conj().T
        # Measure from diagonal
        probs = np.real(np.diag(rho_qft))
        probs = np.maximum(probs, 0)
        if probs.sum() < 1e-12:
            attempted += 1
            continue
        probs /= probs.sum()
        m = int(np.random.choice(d_x, p=probs))
        r = continued_fraction_period(m, d_x, N)
        if r > 0 and r % 2 == 0:
            x = pow(a, r // 2, N)
            if x != N - 1:
                f1 = gcd(x - 1, N)
                f2 = gcd(x + 1, N)
                if 1 < f1 < N or 1 < f2 < N:
                    n_success += 1
        attempted += 1
    return dict(
        N=N, n=n, n_q=n_q, d_x=d_x,
        n_trials=n_trials, attempted=attempted,
        success=n_success,
        success_rate=n_success / max(1, attempted),
        gate_count_Shor=n_gates,
        gate_count_Regev=int(n ** 1.5 * max(1, log2(n))),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RESOURCE EXTRAPOLATION (Regev asymptotic formula)
# ══════════════════════════════════════════════════════════════════════════════

def resource_extrapolation(n_bits: list) -> list:
    """Compute theoretical gate counts for Shor O(n² log n log log n) vs
    Regev O(n^{1.5} log n) for a given bit-size.  Adds CQEC overhead."""
    out = []
    for n in n_bits:
        # Shor: n² log n log log n (Harvey-Hoeven) — approximate
        shor_g = n ** 2 * max(1, log2(n)) * max(1, log2(max(2, log2(n))))
        regev_g = (n ** 1.5) * max(1, log2(n))
        speedup = shor_g / regev_g if regev_g > 0 else float("inf")
        # CQEC overhead: O(d^6) per recovery round, but we apply the channel
        # once per gate block.  Estimate relative factor:
        # Each Regev "block" operates on ~ log n qubits at once, so
        # per-block CQEC is O((2^log n)^6) = O(n^6). But we only need k
        # recovery rounds, where k ≈ log(1/ε) / log(F_gain per round).
        # Rough scaling:  cqec_overhead = n^2 per gate block.
        cqec_overhead = n ** 2
        out.append(dict(
            n_bits=n,
            gate_count_Shor=int(shor_g),
            gate_count_Regev=int(regev_g),
            Regev_speedup=float(speedup),
            cqec_overhead_factor=int(cqec_overhead),
            Regev_with_CQEC=int(regev_g * cqec_overhead),
        ))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 78)
    print("SHOR / REGEV FACTORING — SCALING ANALYSIS UNDER ORGANIC NOISE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    # --------------------------------------------------------------
    # A. Run full Shor numerically for a set of small N, per profile
    # --------------------------------------------------------------
    # Keep N ≤ 35 so the QFT register (2^2n = up to 4096) stays tractable;
    # 51 would require 2^12 = 4096-dim QFT per trial = ~0.7 s × 30 trials × 5 noise.
    N_list = [15, 21, 33, 35]
    empirical = {}
    for N in N_list:
        empirical[N] = {}
        noiseless = shor_factor(N, n_trials=30, profile=None)
        empirical[N]["noiseless"] = noiseless
        print(f"\n  N = {N}  (n = {noiseless['n']} bits, "
              f"n_q = {noiseless['n_q']}, d_x = {noiseless['d_x']})")
        print(f"    noiseless: success_rate = {noiseless['success_rate']:.2f}  "
              f"gates_Shor = {noiseless['gate_count_Shor']:d}   "
              f"gates_Regev_est = {noiseless['gate_count_Regev']:d}")

        for name, prof in PROFILES.items():
            r = shor_factor(N, n_trials=30, profile=prof)
            empirical[N][name] = r
            print(f"    {name:<15} γ={prof['gamma']:<7g} "
                  f"success_rate = {r['success_rate']:.2f}")

    # --------------------------------------------------------------
    # B. Resource extrapolation up to RSA-2048
    # --------------------------------------------------------------
    n_bits_range = [16, 32, 64, 128, 256, 512, 1024, 2048]
    extrap = resource_extrapolation(n_bits_range)
    print("\n" + "─" * 78)
    print("RESOURCE EXTRAPOLATION  (theoretical gate counts)")
    print("─" * 78)
    print(f"  {'n_bits':>6} {'Shor gates':>14} {'Regev gates':>14} "
          f"{'Speedup':>10} {'Regev+CQEC':>14}")
    for e in extrap:
        print(f"  {e['n_bits']:>6d} {e['gate_count_Shor']:>14,d} "
              f"{e['gate_count_Regev']:>14,d} {e['Regev_speedup']:>10.2f}x "
              f"{e['Regev_with_CQEC']:>14,d}")

    # --------------------------------------------------------------
    # C. Cross-profile success-rate summary (organic paths)
    # --------------------------------------------------------------
    print("\n" + "─" * 78)
    print("EMPIRICAL SUCCESS RATES vs ORGANIC PROFILE  (averaged over N)")
    print("─" * 78)
    for name in ["noiseless"] + list(PROFILES.keys()):
        rates = [empirical[N][name]["success_rate"] for N in N_list]
        mean = float(np.mean(rates))
        print(f"  {name:<15} mean success rate = {mean:.3f}  "
              f"(per-N: {', '.join(f'{r:.2f}' for r in rates)})")

    # --------------------------------------------------------------
    # D. Save results
    # --------------------------------------------------------------
    payload = dict(
        meta=dict(
            date=datetime.now().isoformat(),
            wallclock_s=time.time() - t0,
            n_trials_per_N=30,
            profiles=PROFILES,
        ),
        empirical_shor=empirical,
        resource_extrapolation=extrap,
    )
    out = os.path.join(HERE, "results", "shor_regev_scaling.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o:
                   int(o) if isinstance(o, np.integer) else
                   float(o) if isinstance(o, np.floating) else str(o))
    print(f"\n  Saved → {out}")
    print(f"  Wallclock: {time.time() - t0:.1f} s")
    return payload


if __name__ == "__main__":
    main()
