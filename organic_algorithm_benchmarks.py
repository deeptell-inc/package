#!/usr/bin/env python3
"""
organic_algorithm_benchmarks.py
================================
Run five canonical quantum-algorithm benchmarks under **organic material
noise profiles** corresponding to Paths 1, 2, 4 of the feasibility study
(see ORGANIC_QC_FEASIBILITY.md):

    Path 1  — Organic quantum reservoir (engineered radical-pair ensemble, RT)
    Path 2  — Organic radical spin qubit quantum computer (PTM/Trityl, RT)
    Path 4  — Topological soliton qubit (SSH polyacetylene, RT)

Benchmarks (inspired by papers attached by the user):
    1. Shor / Regev factoring  — state prep from cqec.algorithms.make_regev
       (cf. Regev 2024, arXiv:2308.06572)
    2. Quantum Phase Estimation — make_cfqpe
       (cf. Clinton et al., PRX Quantum 7, 010345 (2026))
    3. Quantum KAN              — make_qkan
       (cf. Ivashkov et al., arXiv:2410.04435)
    4. Time-series prediction   — spike series → density-matrix features
       (brainQ.spike_timeseries_quantum style)
    5. MNIST classification     — sklearn digits → density-matrix features
       (brainQ.mnist_quantum_brain style)

For each (material × algorithm) we compare:
    ideal  — noise-free
    noisy  — decohered state (organic noise profile)
    cqec   — after recursive-covariant purification + CQEC recovery
             (reuses cqec.covariant_purification)
    classical baseline (ML only)

This script is self-contained: it pulls state-prep from cqec.algorithms
and noise/recovery from cqec.covariant_purification by adding them to
the import path.
"""

import os
import sys
import json
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

# ── Locate sibling packages ────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
CQEC_ROOT = os.path.normpath(os.path.join(HERE, "..", "cqec"))
BRAINQ_ROOT = os.path.normpath(os.path.join(HERE, "..", "brainQ"))
for p in (CQEC_ROOT, BRAINQ_ROOT, os.path.join(CQEC_ROOT, "cqec")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cqec.algorithms import make_qkan, make_cfqpe, make_regev        # noqa
from covariant_purification import (                                   # noqa
    fidelity,
    purity,
    l1_coherence,
    dephasing_channel,
    depolarizing_channel,
    recursive_covariant,
    cqec_recovery,
)

try:
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SEED = 42
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ORGANIC NOISE PROFILES (Paths 1, 2, 4)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrganicProfile:
    """Noise profile for an organic material implementation."""

    name: str
    path_id: int
    material: str
    gamma: float           # effective dephasing rate (dimensionless)
    delta: float           # depolarizing probability per channel application
    T2_us: float           # coherence time in µs
    gate_ns: float         # gate time in ns
    T_op: float            # operation temperature in K
    notes: str = ""

    @property
    def below_EB(self) -> bool:
        return self.gamma < 0.3


ORGANIC_PROFILES = [
    # Path 1 — reservoir: we pick a moderate γ, because reservoirs benefit
    # from some dissipation; the "rich" setting is the engineered RP.
    OrganicProfile(
        name="Path1_RadicalPairRes",
        path_id=1,
        material="Engineered flavin-nitroxide RP (viscous organic host)",
        gamma=0.10,          # tuneable: chosen near γ_c for reservoir richness
        delta=0.08,
        T2_us=0.10,
        gate_ns=10.0,
        T_op=298.0,
        notes="Deliberately near γ_c so reservoir retains driven dissipation.",
    ),
    # Path 2 — coherent quantum computer: PTM radical spin qubit
    OrganicProfile(
        name="Path2_PTMRadical",
        path_id=2,
        material="PTM radical in COF lattice (EDSR, RT)",
        gamma=0.003,
        delta=0.005,
        T2_us=3.0,
        gate_ns=8.0,
        T_op=298.0,
        notes="γ≪γ_c → standard QEC converges; used for Shor/QPE/QKAN.",
    ),
    # Path 4 — topological soliton qubit
    OrganicProfile(
        name="Path4_SSHSoliton",
        path_id=4,
        material="trans-polyacetylene SSH soliton",
        gamma=0.002,
        delta=0.003,
        T2_us=0.5,
        gate_ns=1.0,
        T_op=298.0,
        notes="Z₂ winding-number protection (topological, RT).",
    ),
]


def apply_organic_noise(rho: np.ndarray, profile: OrganicProfile) -> np.ndarray:
    """Apply the organic noise channel (dephasing + depolarizing)."""
    out = dephasing_channel(rho, profile.gamma)
    out = depolarizing_channel(out, profile.delta)
    # Hermitise & renormalise to keep matrix numerically a valid DM.
    out = (out + out.conj().T) / 2
    tr = np.real(np.trace(out))
    if tr > 1e-15:
        out = out / tr
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. CATALYST / CQEC PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def cqec_pipeline(rho_target: np.ndarray,
                  rho_noisy: np.ndarray,
                  d: int,
                  n_rounds: int = 2) -> tuple:
    """
    Apply recursive covariant purification to a noisy catalyst copy,
    then CQEC-recover the noisy algorithmic state.

    Returns (rho_corrected, n_copies_used, success_prob).
    """
    rho_cat_noisy = rho_noisy.copy()
    rho_cat, n_copies, p_succ = recursive_covariant(rho_cat_noisy, d, n_rounds)
    rho_cor = cqec_recovery(rho_target, rho_noisy, rho_cat)
    return rho_cor, n_copies, p_succ


# ══════════════════════════════════════════════════════════════════════════════
# 3. ALGORITHM BENCHMARKS (state-preparation benchmarks)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlgoResult:
    algorithm: str
    profile: str
    path_id: int
    d: int
    fid_noisy: float
    fid_cqec: float
    pur_ideal: float
    pur_noisy: float
    pur_cqec: float
    coh_ideal: float
    coh_noisy: float
    coh_cqec: float
    n_copies: int
    p_success: float


ALG_FACTORIES = {
    "QKAN":         (make_qkan,  4,  "Quantum KAN (Chebyshev-amplitude)"),
    "QPE":          (make_cfqpe, 16, "Control-free QPE (Fermi-Hubbard)"),
    "Shor_Regev":   (make_regev, 64, "Regev factoring (discrete Gaussian)"),
}


def run_algorithm_benchmarks(profiles: list) -> list:
    """Run the three state-prep benchmarks for every organic profile."""
    results = []
    for alg_key, (factory, d, long_name) in ALG_FACTORIES.items():
        rho_target, d_actual = factory(seed=SEED)
        assert d_actual == d, f"dim mismatch for {alg_key}: {d_actual} vs {d}"

        for prof in profiles:
            rho_noisy = apply_organic_noise(rho_target, prof)
            rho_cor, n_copies, p_succ = cqec_pipeline(rho_target, rho_noisy, d,
                                                      n_rounds=2)

            res = AlgoResult(
                algorithm=alg_key,
                profile=prof.name,
                path_id=prof.path_id,
                d=d,
                fid_noisy=fidelity(rho_target, rho_noisy),
                fid_cqec=fidelity(rho_target, rho_cor),
                pur_ideal=purity(rho_target),
                pur_noisy=purity(rho_noisy),
                pur_cqec=purity(rho_cor),
                coh_ideal=l1_coherence(rho_target),
                coh_noisy=l1_coherence(rho_noisy),
                coh_cqec=l1_coherence(rho_cor),
                n_copies=n_copies,
                p_success=p_succ,
            )
            results.append(res)
            print(f"  [{alg_key:<11}] {prof.name:<22} "
                  f"F_noisy={res.fid_noisy:.4f}  F_cqec={res.fid_cqec:.4f}  "
                  f"P={res.pur_noisy:.3f}→{res.pur_cqec:.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. ML BENCHMARKS: time-series + MNIST
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(rho: np.ndarray) -> np.ndarray:
    """Diagonal, upper-triangle magnitudes, plus purity & l1-coherence."""
    d = rho.shape[0]
    diag = np.real(np.diag(rho))
    upper = [np.abs(rho[i, j]) for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([diag, upper, [purity(rho), l1_coherence(rho)]])


# ─── 4.1 Spike time-series prediction ─────────────────────────────────────

def generate_spike_series(n_series=60, n_total=60,
                           spike_rate=0.12, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_total)
    data = np.zeros((n_series, n_total))
    for s in range(n_series):
        freq = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        baseline = 0.5 * np.sin(freq * t + phase)
        spikes = np.zeros(n_total)
        for pos in np.where(rng.random(n_total) < spike_rate)[0]:
            amp = rng.uniform(3.0, 8.0)
            for dt in range(min(5, n_total - pos)):
                spikes[pos + dt] += amp * np.exp(-dt / 1.2)
        noise = 0.3 * rng.standard_normal(n_total)
        data[s] = baseline + spikes + noise
    return data


def encode_window(window: np.ndarray, d: int = 8) -> np.ndarray:
    amp = window.astype(float)
    amp = amp - amp.min() + 1e-6
    n = np.linalg.norm(amp)
    amp = amp / n if n > 1e-12 else np.ones(d) / np.sqrt(d)
    return np.outer(amp, amp.conj())


def timeseries_features(window: np.ndarray,
                         profile: OrganicProfile = None,
                         apply_cqec: bool = False,
                         d: int = 8) -> np.ndarray:
    rho_ideal = encode_window(window, d)
    if profile is None:
        return extract_features(rho_ideal)
    rho_n = apply_organic_noise(rho_ideal, profile)
    if apply_cqec:
        rho_n, _, _ = cqec_pipeline(rho_ideal, rho_n, d, n_rounds=1)
    return extract_features(rho_n)


def run_timeseries_benchmark(profiles: list, d: int = 8) -> dict:
    if not HAS_SKLEARN:
        return {"error": "sklearn unavailable"}

    # 60 series, 50 input + 10 forecast
    series = generate_spike_series(n_series=60, n_total=60, seed=SEED)
    train = series[:50]   # 50 series for training windows
    test = series[50:]    # 10 series held out

    window_len = d
    horizon = 1

    def build_xy(series_block, feat_fn):
        X, y = [], []
        for row in series_block:
            for i in range(len(row) - window_len - horizon):
                w = row[i:i + window_len]
                nxt = row[i + window_len]
                X.append(feat_fn(w))
                y.append(nxt)
        return np.array(X), np.array(y)

    def eval_setting(feat_fn, label):
        X_tr, y_tr = build_xy(train, feat_fn)
        X_te, y_te = build_xy(test,  feat_fn)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        return dict(
            label=label,
            mse=float(mean_squared_error(y_te, y_hat)),
            mae=float(mean_absolute_error(y_te, y_hat)),
            n_train=int(len(y_tr)),
            n_test=int(len(y_te)),
        )

    results = {}
    # Classical: raw window values as features (no quantum)
    results["classical"] = eval_setting(lambda w: w.copy(), "Classical raw")
    # Ideal quantum (noise-free)
    results["ideal"] = eval_setting(
        lambda w: timeseries_features(w, None, False, d), "Ideal quantum")

    for prof in profiles:
        noisy = eval_setting(
            lambda w, p=prof: timeseries_features(w, p, False, d),
            f"Noisy {prof.name}")
        cqec = eval_setting(
            lambda w, p=prof: timeseries_features(w, p, True, d),
            f"CQEC  {prof.name}")
        results[f"noisy_{prof.name}"] = noisy
        results[f"cqec_{prof.name}"] = cqec
        print(f"  [timeseries] {prof.name:<22} "
              f"MSE noisy={noisy['mse']:.3f}  cqec={cqec['mse']:.3f}  "
              f"(classical={results['classical']['mse']:.3f})")
    return results


# ─── 4.2 MNIST classification ──────────────────────────────────────────────

def mnist_image_features(img: np.ndarray,
                          profile: OrganicProfile = None,
                          apply_cqec: bool = False,
                          d: int = 8) -> np.ndarray:
    pixels = img.flatten().astype(float)
    pmax = pixels.max()
    if pmax > 0:
        pixels /= pmax
    groups = len(pixels) // d
    feats = []
    for g in range(groups):
        amp = pixels[g * d:(g + 1) * d]
        n = np.linalg.norm(amp)
        amp = amp / n if n > 1e-12 else np.ones(d) / np.sqrt(d)
        rho_ideal = np.outer(amp, amp.conj())
        if profile is None:
            feats.append(extract_features(rho_ideal))
            continue
        rho_n = apply_organic_noise(rho_ideal, profile)
        if apply_cqec:
            rho_n, _, _ = cqec_pipeline(rho_ideal, rho_n, d, n_rounds=1)
        feats.append(extract_features(rho_n))
    return np.concatenate(feats)


def run_mnist_benchmark(profiles: list,
                         n_train: int = 200,
                         n_test: int = 100,
                         d: int = 8) -> dict:
    if not HAS_SKLEARN:
        return {"error": "sklearn unavailable"}

    digits = load_digits()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(digits.target))
    tr = idx[:n_train]
    te = idx[n_train:n_train + n_test]
    X_tr_img = digits.images[tr]
    X_te_img = digits.images[te]
    y_tr = digits.target[tr]
    y_te = digits.target[te]

    def fit_eval(feat_fn, label):
        X_tr = np.stack([feat_fn(img) for img in X_tr_img])
        X_te = np.stack([feat_fn(img) for img in X_te_img])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=SEED)
        clf.fit(X_tr, y_tr)
        y_hat = clf.predict(X_te)
        return dict(
            label=label,
            acc=float(accuracy_score(y_te, y_hat)),
            n_train=int(n_train),
            n_test=int(n_test),
            n_features=int(X_tr.shape[1]),
        )

    results = {}
    results["classical"] = fit_eval(lambda im: im.flatten().astype(float), "Classical raw")
    results["ideal"] = fit_eval(lambda im: mnist_image_features(im, None, False, d),
                                "Ideal quantum")
    for prof in profiles:
        noisy = fit_eval(lambda im, p=prof: mnist_image_features(im, p, False, d),
                         f"Noisy {prof.name}")
        cqec  = fit_eval(lambda im, p=prof: mnist_image_features(im, p, True, d),
                         f"CQEC  {prof.name}")
        results[f"noisy_{prof.name}"] = noisy
        results[f"cqec_{prof.name}"] = cqec
        print(f"  [MNIST]      {prof.name:<22} "
              f"acc noisy={noisy['acc']:.3f}  cqec={cqec['acc']:.3f}  "
              f"(classical={results['classical']['acc']:.3f})")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"unserialisable {type(obj)}")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print("ORGANIC QC ALGORITHM BENCHMARKS  —  Paths 1, 2, 4")
    print("Based on cqec/brainQ frameworks; state-preps from cqec.algorithms")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    print("\nOrganic noise profiles:")
    for prof in ORGANIC_PROFILES:
        print(f"  Path {prof.path_id}: {prof.name:<22} γ={prof.gamma:<6} "
              f"δ={prof.delta:<6} T2={prof.T2_us}µs  @{prof.T_op:.0f} K  "
              f"below γ_c? {'YES' if prof.below_EB else 'no'}")

    # ---- Algorithm benchmarks (QKAN, QPE, Regev) -----------------------
    print("\n" + "─" * 78)
    print("1. STATE-PREPARATION ALGORITHM BENCHMARKS")
    print("─" * 78)
    alg_results = run_algorithm_benchmarks(ORGANIC_PROFILES)

    # ---- γ-sweep for Path 1 reservoir ---------------------------------
    print("\n" + "─" * 78)
    print("2. NOISE SWEEP — γ-dependence of fidelity & CQEC gain")
    print("─" * 78)
    gamma_sweep = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    sweep_results = {}
    for alg_key, (factory, d, _) in ALG_FACTORIES.items():
        rho_target, _ = factory(seed=SEED)
        rows = []
        for g in gamma_sweep:
            prof = OrganicProfile(name=f"sweep_g{g}", path_id=0,
                                   material="γ-sweep", gamma=g,
                                   delta=min(0.5, g * 0.8),
                                   T2_us=1.0, gate_ns=10.0, T_op=298.0)
            rho_n = apply_organic_noise(rho_target, prof)
            rho_c, _, _ = cqec_pipeline(rho_target, rho_n, d, n_rounds=2)
            rows.append(dict(
                gamma=g,
                fid_noisy=fidelity(rho_target, rho_n),
                fid_cqec=fidelity(rho_target, rho_c),
                pur_noisy=purity(rho_n),
                pur_cqec=purity(rho_c),
            ))
        sweep_results[alg_key] = rows
        print(f"\n  {alg_key}:")
        print(f"    {'γ':>8} {'F_noisy':>8} {'F_cqec':>8} {'ΔF':>8}")
        for r in rows:
            print(f"    {r['gamma']:>8.3f} {r['fid_noisy']:>8.4f} "
                  f"{r['fid_cqec']:>8.4f} {r['fid_cqec']-r['fid_noisy']:>+8.4f}")

    # ---- ML benchmarks ------------------------------------------------
    print("\n" + "─" * 78)
    print("3. TIME-SERIES PREDICTION (spike series, Ridge on DM features)")
    print("─" * 78)
    ts_results = run_timeseries_benchmark(ORGANIC_PROFILES, d=8)

    print("\n" + "─" * 78)
    print("4. MNIST CLASSIFICATION (sklearn digits, RBF-SVM on DM features)")
    print("─" * 78)
    mnist_results = run_mnist_benchmark(ORGANIC_PROFILES, d=8)

    total_s = time.time() - t0
    print(f"\nTotal wallclock: {total_s:.1f} s")

    # ---- Aggregate & save ----------------------------------------------
    payload = {
        "meta": {
            "date": datetime.now().isoformat(),
            "seed": SEED,
            "wallclock_s": total_s,
            "has_sklearn": HAS_SKLEARN,
        },
        "profiles": [asdict(p) for p in ORGANIC_PROFILES],
        "algorithm_benchmarks": [asdict(r) for r in alg_results],
        "gamma_sweep": sweep_results,
        "timeseries_benchmark": ts_results,
        "mnist_benchmark": mnist_results,
    }
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "organic_algorithm_benchmarks.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\nSaved results → {out_path}")

    # ---- Pretty comparison table --------------------------------------
    print("\n" + "═" * 78)
    print("SUMMARY TABLE: algorithm fidelity per organic path")
    print("═" * 78)
    print(f"\n  {'Algorithm':<14} {'Profile':<22} {'F_noisy':>8} {'F_cqec':>8} "
          f"{'ΔF':>8} {'Pur_cqec':>10}")
    print("  " + "─" * 72)
    for r in alg_results:
        dF = r.fid_cqec - r.fid_noisy
        print(f"  {r.algorithm:<14} {r.profile:<22} {r.fid_noisy:>8.4f} "
              f"{r.fid_cqec:>8.4f} {dF:>+8.4f} {r.pur_cqec:>10.4f}")

    if "classical" in ts_results:
        print("\n  Time-series MSE (lower = better):")
        for k, v in ts_results.items():
            if isinstance(v, dict) and "mse" in v:
                print(f"    {k:<35} MSE={v['mse']:.4f}  MAE={v['mae']:.4f}")

    if "classical" in mnist_results:
        print("\n  MNIST classification accuracy (higher = better):")
        for k, v in mnist_results.items():
            if isinstance(v, dict) and "acc" in v:
                print(f"    {k:<35} acc={v['acc']:.4f}")

    return payload


if __name__ == "__main__":
    main()
