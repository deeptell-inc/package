#!/usr/bin/env python3
"""
hybrid_qrc_qc.py
================
Hybrid Quantum Reservoir Computer (Path 1) → Quantum Computer (Path 2)
pipeline simulation.

This is the "ハイブリッド QRC-QC アーキテクチャ" next-step item from
ORGANIC_BENCHMARKS_EXTENDED.md §10.3.

Pipeline
--------
    noisy input signal
      │
      ▼
    PATH 1 (radical-pair reservoir, γ=0.10, RT, tune for richness)
      │    noise-driven reservoir dynamics extracts denoised features
      ▼
    classical bridge (dim reduction + re-encoding)
      │
      ▼
    PATH 2 (PTM coherent QC, γ=0.003, RT, high-fidelity gates)
      │    precision quantum algorithm (QPE-like phase estimation)
      ▼
    output estimate

Task
----
Given a noisy time-series signal x(t) = cos(2π f_true t + φ_true) + noise,
recover the phase φ_true ∈ [0, 2π).

We compare four pipelines:
  A) Classical FFT-peak phase recovery
  B) Path 2 alone (direct QPE on raw input)
  C) Path 1 alone (reservoir features → Ridge regression)
  D) Hybrid Path 1 → Path 2 (reservoir preprocess + QPE)
"""

import os
import sys
import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CQEC_ROOT = os.path.normpath(os.path.join(HERE, "..", "cqec"))
for p in (CQEC_ROOT, os.path.join(CQEC_ROOT, "cqec")):
    if p not in sys.path:
        sys.path.insert(0, p)

from covariant_purification import (
    fidelity, purity, l1_coherence,
    dephasing_channel, depolarizing_channel,
    recursive_covariant, cqec_recovery,
)

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NOISE PROFILES
# ══════════════════════════════════════════════════════════════════════════════

PATH1_RES = dict(gamma=0.10,  delta=0.08)   # reservoir
PATH2_QC  = dict(gamma=0.003, delta=0.005)   # coherent QC


def apply_noise(rho, prof):
    r = dephasing_channel(rho, prof["gamma"])
    r = depolarizing_channel(r, prof["delta"])
    r = (r + r.conj().T) / 2
    tr = np.real(np.trace(r))
    if tr > 1e-15:
        r /= tr
    return r


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TASK: noisy-signal phase recovery
# ══════════════════════════════════════════════════════════════════════════════

def gen_task(n_samples=200, n_points=16, f_true=1.0, noise_std=0.5,
              seed=None):
    """Generate (signals, phases).  signals.shape = (n_samples, n_points);
    phases ∈ [0, 2π) — the label to predict.
    Time grid t = k/n_points (no endpoint) so a frequency f_true=1 lands
    exactly in FFT bin 1 with no spectral leakage."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n_samples)
    t = np.arange(n_points) / n_points
    signals = np.cos(2 * np.pi * f_true * t[None, :] + phases[:, None])
    signals += noise_std * rng.standard_normal((n_samples, n_points))
    return signals, phases


def angular_mse(pred, target):
    """Phase error modulo 2π."""
    diff = (pred - target + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(diff ** 2))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PATH 1: Radical-pair quantum reservoir preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def reservoir_encode(signal: np.ndarray, d: int = 16) -> np.ndarray:
    """Amplitude-encode a length-n_points signal into a d-dim pure state."""
    amp = signal.copy().astype(float)
    amp = amp - amp.min() + 1e-6
    if len(amp) < d:
        amp = np.pad(amp, (0, d - len(amp)))
    elif len(amp) > d:
        amp = amp[:d]
    nrm = np.linalg.norm(amp)
    amp = amp / nrm if nrm > 1e-12 else np.ones(d) / np.sqrt(d)
    return amp


def reservoir_dynamics(rho0: np.ndarray, n_steps: int = 6,
                        prof: dict = PATH1_RES) -> list:
    """Driven reservoir: at each step apply a fixed unitary kick then noise."""
    d = rho0.shape[0]
    # Fixed reservoir Hamiltonian: XY-like band with random couplings (seeded)
    rng = np.random.default_rng(123)
    H = rng.standard_normal((d, d)) * 0.3
    H = (H + H.T) / 2
    from scipy.linalg import expm
    U = expm(-1j * 0.3 * H)
    rho = rho0.copy()
    trace = []
    for _ in range(n_steps):
        rho = U @ rho @ U.conj().T
        rho = apply_noise(rho, prof)
        trace.append(rho.copy())
    return trace


def extract_reservoir_features(rho_trace: list) -> np.ndarray:
    """Flatten per-step density-matrix features from the reservoir trace."""
    feats = []
    for rho in rho_trace:
        d = rho.shape[0]
        diag = np.real(np.diag(rho))
        upper = [np.abs(rho[i, j]) for i in range(d) for j in range(i + 1, d)]
        feats.extend(diag.tolist())
        feats.extend(upper[:32])   # cap to keep dim manageable
        feats.append(purity(rho))
    return np.asarray(feats)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PATH 2: Quantum Phase Estimation on a denoised amplitude state
# ══════════════════════════════════════════════════════════════════════════════

def qft(d: int) -> np.ndarray:
    j = np.arange(d)[:, None]
    k = np.arange(d)[None, :]
    return np.exp(2j * np.pi * j * k / d) / np.sqrt(d)


def qpe_encode(signal: np.ndarray, d: int = 16,
                dc_bias: float = 0.8) -> np.ndarray:
    """Complex-amplitude encoding with a DC reference for unambiguous
    phase estimation.

    We (a) add a DC bias so the density matrix has a stable reference bin,
    (b) convert to analytic signal via Hilbert transform so the fundamental
    has a single spectral peak. The resulting ρ has non-zero coherence
    between bin 0 (DC) and bin f, carrying the phase φ without the
    π-ambiguity of bare-cosine amplitude encoding.
    """
    from scipy.signal import hilbert
    if len(signal) < d:
        s = np.pad(signal, (0, d - len(signal)))
    else:
        s = signal[:d]
    # Add DC reference (gives bin 0 a known amplitude)
    s = s + dc_bias
    z = hilbert(s)
    amp = z.astype(complex)
    n = np.linalg.norm(amp)
    return amp / n if n > 1e-12 else np.ones(d, dtype=complex) / np.sqrt(d)


def qpe_phase_estimate(signal: np.ndarray, d: int = 16,
                        prof: dict = PATH2_QC) -> float:
    """
    QPE on (signal + DC bias) converted to analytic form via Hilbert.
    With the +i-convention QFT, the fundamental-frequency peak of an
    analytic e^{i 2π k/d + iφ} lands at bin d-1, while the DC bias puts
    a reference peak at bin 0. The coherence ρ_out[d-1, 0] therefore
    carries +φ unambiguously.
    """
    amp = qpe_encode(signal, d)
    rho = np.outer(amp, amp.conj())
    rho = apply_noise(rho, prof)
    F = qft(d)
    out = F @ rho @ F.conj().T
    coh = out[d - 1, 0]
    if np.abs(coh) < 1e-10:
        return 0.0
    return float(np.angle(coh) % (2 * np.pi))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CLASSICAL BASELINE: FFT peak phase
# ══════════════════════════════════════════════════════════════════════════════

def classical_phase_estimate(signal: np.ndarray, f_true: float = 1.0) -> float:
    n = len(signal)
    F = np.fft.fft(signal)
    k_target = int(round(f_true * n / n))   # index of f_true
    k_target = max(1, k_target)              # avoid DC
    phase = np.angle(F[k_target])
    return float(phase % (2 * np.pi))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_classical(signals, f_true=1.0):
    return np.array([classical_phase_estimate(s, f_true) for s in signals])


def pipeline_path2_only(signals, d=16):
    return np.array([qpe_phase_estimate(s, d, PATH2_QC) for s in signals])


def pipeline_path1_ridge(signals_train, phases_train, signals_test,
                          d=16, n_steps=6):
    """Path 1 (reservoir) → Ridge regression for phase (cos, sin)."""
    if not HAS_SKLEARN:
        return None

    def feat(sig):
        amp = reservoir_encode(sig, d)
        rho = np.outer(amp, amp.conj())
        trace = reservoir_dynamics(rho, n_steps=n_steps, prof=PATH1_RES)
        return extract_reservoir_features(trace)

    X_tr = np.stack([feat(s) for s in signals_train])
    X_te = np.stack([feat(s) for s in signals_test])
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    # Regress cos and sin of the phase
    y_tr = np.column_stack([np.cos(phases_train), np.sin(phases_train)])
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    phase_hat = np.arctan2(y_hat[:, 1], y_hat[:, 0]) % (2 * np.pi)
    return phase_hat


def pipeline_hybrid(signals_train, phases_train, signals_test,
                     d=16, n_steps=6):
    """Hybrid Path 1 → Path 2:
       * Path 1 (reservoir + ridge) gives a coarse phase estimate θ₁̂.
       * Path 2 (QPE) gives a fine phase estimate θ₂̂.
       * Fuse via circular inverse-variance weighting learned from the
         training set (empirical-Bayes phase averaging)."""
    if not HAS_SKLEARN:
        return None

    # Path 1 predictions (on train and test)
    def reservoir_feat(sig):
        amp = qpe_encode(sig, d)
        rho = np.outer(amp, amp.conj())
        trace = reservoir_dynamics(rho, n_steps=n_steps, prof=PATH1_RES)
        return extract_reservoir_features(trace)

    X_tr = np.stack([reservoir_feat(s) for s in signals_train])
    X_te = np.stack([reservoir_feat(s) for s in signals_test])
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    y_tr = np.column_stack([np.cos(phases_train), np.sin(phases_train)])
    m1 = Ridge(alpha=1.0).fit(X_tr, y_tr)
    pred1_tr = np.arctan2(*m1.predict(X_tr)[:, [1, 0]].T) % (2 * np.pi)
    pred1_te = np.arctan2(*m1.predict(X_te)[:, [1, 0]].T) % (2 * np.pi)

    # Path 2 predictions (direct QPE)
    pred2_tr = np.array([qpe_phase_estimate(s, d, PATH2_QC)
                          for s in signals_train])
    pred2_te = np.array([qpe_phase_estimate(s, d, PATH2_QC)
                          for s in signals_test])

    # Estimate per-pipeline phase variance on training set
    def circ_var(pred, true):
        diff = (pred - true + np.pi) % (2 * np.pi) - np.pi
        return float(np.var(diff))
    v1 = circ_var(pred1_tr, phases_train) + 1e-8
    v2 = circ_var(pred2_tr, phases_train) + 1e-8
    w1 = 1.0 / v1
    w2 = 1.0 / v2

    # Circular weighted mean
    fused = np.angle(w1 * np.exp(1j * pred1_te) + w2 * np.exp(1j * pred2_te))
    return fused % (2 * np.pi)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 78)
    print("HYBRID QRC (Path 1) → QC (Path 2) PIPELINE")
    print("Task: phase recovery from noisy cosine signal")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    results_by_noise = {}
    noise_levels = [0.1, 0.3, 0.5, 0.8, 1.2]
    n_train = 120
    n_test = 120

    for noise_std in noise_levels:
        print("\n" + "─" * 78)
        print(f"Input noise σ = {noise_std}")
        print("─" * 78)

        sigs_tr, ph_tr = gen_task(n_samples=n_train, noise_std=noise_std,
                                   seed=100 + int(100 * noise_std))
        sigs_te, ph_te = gen_task(n_samples=n_test, noise_std=noise_std,
                                   seed=200 + int(100 * noise_std))

        r = {}

        p_cls = pipeline_classical(sigs_te)
        r["classical_FFT"] = dict(mse=angular_mse(p_cls, ph_te))
        print(f"  A) Classical FFT           MSE = {r['classical_FFT']['mse']:.4f}")

        p_p2 = pipeline_path2_only(sigs_te)
        r["path2_only"] = dict(mse=angular_mse(p_p2, ph_te))
        print(f"  B) Path 2 only (QPE, γ={PATH2_QC['gamma']})    "
              f"MSE = {r['path2_only']['mse']:.4f}")

        p_p1 = pipeline_path1_ridge(sigs_tr, ph_tr, sigs_te)
        if p_p1 is not None:
            r["path1_reservoir_ridge"] = dict(mse=angular_mse(p_p1, ph_te))
            print(f"  C) Path 1 reservoir+ridge (γ={PATH1_RES['gamma']}) "
                  f"MSE = {r['path1_reservoir_ridge']['mse']:.4f}")

        p_hy = pipeline_hybrid(sigs_tr, ph_tr, sigs_te)
        r["hybrid_path1_path2"] = dict(mse=angular_mse(p_hy, ph_te))
        print(f"  D) Hybrid Path 1 → Path 2  "
              f"MSE = {r['hybrid_path1_path2']['mse']:.4f}")

        # Best pipeline
        best = min(r.items(), key=lambda kv: kv[1]["mse"])
        print(f"  → best: {best[0]}  MSE = {best[1]['mse']:.4f}")

        results_by_noise[noise_std] = r

    # Rank summary
    print("\n" + "═" * 78)
    print("SUMMARY (phase MSE, lower = better)")
    print("═" * 78)
    print(f"\n  {'σ_noise':>8} {'Classical':>11} {'Path2':>10} {'Path1+Ridge':>13}"
          f" {'Hybrid':>10} {'Best':>18}")
    for noise, r in results_by_noise.items():
        classical = r.get("classical_FFT", {}).get("mse", float("nan"))
        p2 = r.get("path2_only", {}).get("mse", float("nan"))
        p1 = r.get("path1_reservoir_ridge", {}).get("mse", float("nan"))
        hy = r.get("hybrid_path1_path2", {}).get("mse", float("nan"))
        best_name = min(r.items(), key=lambda kv: kv[1]["mse"])[0]
        print(f"  {noise:>8.2f} {classical:>11.4f} {p2:>10.4f} {p1:>13.4f}"
              f" {hy:>10.4f} {best_name:>18}")

    payload = {
        "meta": dict(
            date=datetime.now().isoformat(),
            wallclock_s=time.time() - t0,
            n_train=n_train, n_test=n_test,
            path1=PATH1_RES, path2=PATH2_QC,
        ),
        "results_by_noise": {
            str(k): v for k, v in results_by_noise.items()
        },
    }
    out = os.path.join(HERE, "results", "hybrid_qrc_qc.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved → {out}")
    print(f"  Wallclock: {time.time() - t0:.1f} s")
    return payload


if __name__ == "__main__":
    main()
