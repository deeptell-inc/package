#!/usr/bin/env python3
"""
organic_benchmarks_extended.py
==============================
Extensions to `organic_algorithm_benchmarks.py` implementing the five
"next steps" listed in ORGANIC_ALGORITHM_BENCHMARKS.md §8:

  1. Larger γ-sweep grid + 10-trial confidence intervals
  2. qDRIFT (d=8) added to the algorithm benchmark suite
  3. MNIST full 10-class, 1797-sample, 5-fold evaluation
  4. Path 3 (organic superconductor SVILC, κ-BEDT-TTF, 4 K) as a comparison
  5. Diarylethene photoswitch gate model for 2-qubit entangling gates

All benchmarks reuse `cqec.algorithms`, `cqec.covariant_purification`, and
the brainQ-style feature extraction, in line with the previous script.
"""

import os
import sys
import json
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from scipy.linalg import expm

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CQEC_ROOT = os.path.normpath(os.path.join(HERE, "..", "cqec"))
BRAINQ_ROOT = os.path.normpath(os.path.join(HERE, "..", "brainQ"))
for p in (CQEC_ROOT, BRAINQ_ROOT, os.path.join(CQEC_ROOT, "cqec")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cqec.algorithms import (                                           # noqa
    make_qkan, make_cfqpe, make_regev, make_qdrift, make_bell,
)
from covariant_purification import (                                     # noqa
    fidelity, purity, l1_coherence,
    dephasing_channel, depolarizing_channel,
    recursive_covariant, cqec_recovery,
)

try:
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score, classification_report)
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SEED = 42
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ORGANIC NOISE PROFILES (now including Path 3)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrganicProfile:
    name: str
    path_id: int
    material: str
    gamma: float
    delta: float
    T2_us: float
    gate_ns: float
    T_op: float
    notes: str = ""

    @property
    def below_EB(self) -> bool:
        return self.gamma < 0.3


PROFILES = [
    OrganicProfile(
        name="Path1_RadicalPairRes",
        path_id=1,
        material="Engineered flavin-nitroxide RP (RT)",
        gamma=0.10, delta=0.08,
        T2_us=0.10, gate_ns=10.0, T_op=298.0,
        notes="Reservoir regime, γ near γ_c."),
    OrganicProfile(
        name="Path2_PTMRadical",
        path_id=2,
        material="PTM radical in COF (EDSR, RT)",
        gamma=0.003, delta=0.005,
        T2_us=3.0, gate_ns=8.0, T_op=298.0,
        notes="Room-T coherent QC, γ ≪ γ_c."),
    # NEW — Path 3: organic superconductor SVILC analog
    OrganicProfile(
        name="Path3_OrganicSC_SVILC",
        path_id=3,
        material="κ-(BEDT-TTF)₂Cu[N(CN)₂]Br (4 K)",
        gamma=5e-5, delta=1e-4,
        T2_us=100.0, gate_ns=5.0, T_op=4.0,
        notes="Topological winding-number protection, 4 K."),
    OrganicProfile(
        name="Path4_SSHSoliton",
        path_id=4,
        material="trans-polyacetylene SSH soliton (RT)",
        gamma=0.002, delta=0.003,
        T2_us=0.5, gate_ns=1.0, T_op=298.0,
        notes="Z₂ topological soliton, RT."),
]


def apply_organic_noise(rho, profile):
    out = dephasing_channel(rho, profile.gamma)
    out = depolarizing_channel(out, profile.delta)
    out = (out + out.conj().T) / 2
    tr = np.real(np.trace(out))
    if tr > 1e-15:
        out = out / tr
    return out


def adaptive_rounds(d):
    """Keep CQEC cost bounded: 2 rounds for small d, 1 for large d.
    Complexity of recursive_covariant ≈ n_rounds × O(d⁶)."""
    if d <= 16:
        return 2
    return 1


def cqec_pipeline(rho_target, rho_noisy, d, n_rounds=None):
    if n_rounds is None:
        n_rounds = adaptive_rounds(d)
    rho_cat, n_cop, p = recursive_covariant(rho_noisy.copy(), d, n_rounds)
    return cqec_recovery(rho_target, rho_noisy, rho_cat), n_cop, p


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXTENSION #1 & #2: γ-sweep with 10 trials + qDRIFT
# ══════════════════════════════════════════════════════════════════════════════

ALG_FACTORIES = {
    "QKAN":       (make_qkan,   4),
    "qDRIFT":     (make_qdrift, 8),      # NEW
    "QPE":        (make_cfqpe, 16),
    "Shor_Regev": (make_regev, 64),
}


def ci95(values):
    """Return (mean, half-width-95%-CI) using t-approx for small n."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if n < 2:
        return mean, 0.0
    std = float(arr.std(ddof=1))
    se = std / np.sqrt(n)
    return mean, float(1.96 * se)


def run_gamma_sweep_with_ci(gamma_grid, n_trials=10, n_rounds=None):
    """γ-sweep for every algorithm, averaged over n_trials seeds.
    n_rounds=None → adaptive (2 for d≤16, 1 for d>16) to keep CQEC tractable."""
    results = {}
    for alg_key, (factory, d) in ALG_FACTORIES.items():
        # Large-dimension Regev gets fewer trials (CQEC cost scales as d⁶ per round)
        n_tr_this = max(3, n_trials // 2) if d >= 32 else n_trials
        per_gamma = []
        for g in gamma_grid:
            prof = OrganicProfile(
                name=f"sweep_g{g}", path_id=0, material="γ-sweep",
                gamma=g, delta=min(0.5, g * 0.8),
                T2_us=1.0, gate_ns=10.0, T_op=298.0)
            fid_n, fid_c, pur_n, pur_c = [], [], [], []
            for trial in range(n_tr_this):
                rho_target, _ = factory(seed=SEED + trial)
                rho_n = apply_organic_noise(rho_target, prof)
                rho_c, _, _ = cqec_pipeline(rho_target, rho_n, d, n_rounds)
                fid_n.append(fidelity(rho_target, rho_n))
                fid_c.append(fidelity(rho_target, rho_c))
                pur_n.append(purity(rho_n))
                pur_c.append(purity(rho_c))
            fn_m, fn_ci = ci95(fid_n)
            fc_m, fc_ci = ci95(fid_c)
            pn_m, _ = ci95(pur_n)
            pc_m, _ = ci95(pur_c)
            per_gamma.append(dict(
                gamma=g, n_trials=n_tr_this,
                fid_noisy_mean=fn_m, fid_noisy_ci=fn_ci,
                fid_cqec_mean=fc_m, fid_cqec_ci=fc_ci,
                purity_noisy=pn_m, purity_cqec=pc_m,
            ))
        results[alg_key] = per_gamma
        print(f"\n  {alg_key}  (n_trials={n_tr_this})")
        print(f"    {'γ':>8} {'F_noisy':>14} {'F_cqec':>14} {'ΔF':>8}")
        for r in per_gamma:
            print(f"    {r['gamma']:>8.4f} "
                  f"{r['fid_noisy_mean']:.4f}±{r['fid_noisy_ci']:.4f}   "
                  f"{r['fid_cqec_mean']:.4f}±{r['fid_cqec_ci']:.4f}   "
                  f"{r['fid_cqec_mean']-r['fid_noisy_mean']:+.4f}")
    return results


def run_all_paths_algorithm_benchmarks(profiles, n_trials=10, n_rounds=None):
    """Run every alg × every path with CI across n_trials seeds."""
    out = []
    for alg_key, (factory, d) in ALG_FACTORIES.items():
        # Large-d fewer trials (CQEC cost scales as d⁶)
        n_tr_this = max(3, n_trials // 2) if d >= 32 else n_trials
        for prof in profiles:
            fid_n, fid_c, pur_c = [], [], []
            for trial in range(n_tr_this):
                rho_t, _ = factory(seed=SEED + trial)
                rho_n = apply_organic_noise(rho_t, prof)
                rho_c, _, _ = cqec_pipeline(rho_t, rho_n, d, n_rounds)
                fid_n.append(fidelity(rho_t, rho_n))
                fid_c.append(fidelity(rho_t, rho_c))
                pur_c.append(purity(rho_c))
            fn_m, fn_ci = ci95(fid_n)
            fc_m, fc_ci = ci95(fid_c)
            pc_m, _ = ci95(pur_c)
            out.append(dict(
                algorithm=alg_key, profile=prof.name, path_id=prof.path_id,
                d=d, n_trials=n_tr_this,
                fid_noisy_mean=fn_m, fid_noisy_ci=fn_ci,
                fid_cqec_mean=fc_m, fid_cqec_ci=fc_ci,
                pur_cqec_mean=pc_m,
            ))
            print(f"  [{alg_key:<11}] {prof.name:<23} "
                  f"F_n={fn_m:.4f}±{fn_ci:.4f}   F_c={fc_m:.4f}±{fc_ci:.4f}   "
                  f"ΔF={fc_m-fn_m:+.4f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXTENSION #3: Full MNIST (1797 samples, 5-fold)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(rho):
    d = rho.shape[0]
    diag = np.real(np.diag(rho))
    upper = [np.abs(rho[i, j]) for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([diag, upper, [purity(rho), l1_coherence(rho)]])


def mnist_image_features(img, profile=None, apply_cqec=False, d=8):
    pixels = img.flatten().astype(float)
    pmax = pixels.max()
    if pmax > 0:
        pixels /= pmax
    feats = []
    for g in range(len(pixels) // d):
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


def run_mnist_full(profiles, n_splits=5, d=8):
    """Full MNIST with 5-fold stratified CV, all 1797 digits."""
    if not HAS_SKLEARN:
        return {"error": "sklearn unavailable"}

    digits = load_digits()
    X_img = digits.images           # (1797, 8, 8)
    y = digits.target
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    def build_feats(images, feat_fn):
        return np.stack([feat_fn(im) for im in images])

    def fold_eval_precomputed(X_all, label):
        """5-fold CV on precomputed feature matrix X_all."""
        accs, f1s = [], []
        for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_all, y)):
            X_tr = X_all[tr_idx]
            X_te = X_all[te_idx]
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
            clf = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=SEED)
            clf.fit(X_tr, y[tr_idx])
            y_hat = clf.predict(X_te)
            accs.append(accuracy_score(y[te_idx], y_hat))
            f1s.append(f1_score(y[te_idx], y_hat, average="macro"))
        acc_m, acc_ci = ci95(accs)
        f1_m, f1_ci = ci95(f1s)
        return dict(label=label, acc_mean=acc_m, acc_ci=acc_ci,
                    f1_mean=f1_m, f1_ci=f1_ci,
                    n_samples=int(len(y)), n_splits=int(n_splits),
                    n_features=int(X_all.shape[1]))

    def fold_eval(feat_fn, label):
        # Precompute features ONCE across all 1797 images, then CV
        X_all = build_feats(X_img, feat_fn)
        return fold_eval_precomputed(X_all, label)

    results = {}
    results["classical"] = fold_eval(lambda im: im.flatten().astype(float),
                                     "Classical raw pixels")
    print(f"  [MNIST]  Classical raw      "
          f"acc={results['classical']['acc_mean']:.4f}±"
          f"{results['classical']['acc_ci']:.4f}")

    results["ideal_quantum"] = fold_eval(
        lambda im: mnist_image_features(im, None, False, d), "Ideal quantum")
    print(f"  [MNIST]  Ideal quantum       "
          f"acc={results['ideal_quantum']['acc_mean']:.4f}±"
          f"{results['ideal_quantum']['acc_ci']:.4f}")

    for prof in profiles:
        n_res = fold_eval(lambda im, p=prof: mnist_image_features(im, p, False, d),
                           f"Noisy {prof.name}")
        c_res = fold_eval(lambda im, p=prof: mnist_image_features(im, p, True, d),
                           f"CQEC  {prof.name}")
        results[f"noisy_{prof.name}"] = n_res
        results[f"cqec_{prof.name}"] = c_res
        print(f"  [MNIST]  {prof.name:<23} "
              f"noisy={n_res['acc_mean']:.4f}±{n_res['acc_ci']:.4f}   "
              f"cqec={c_res['acc_mean']:.4f}±{c_res['acc_ci']:.4f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXTENSION #5: Diarylethene photoswitch 2-qubit entangling gate
# ══════════════════════════════════════════════════════════════════════════════

# Pauli operators
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SI = np.eye(2, dtype=complex)
ZZ = np.kron(SZ, SZ)
XI = np.kron(SX, SI)
IX = np.kron(SI, SX)


@dataclass
class PhotoSwitchParams:
    """Diarylethene photoswitch timing parameters.

    J_closed is set to 0.05 GHz so that the ideal gate time (integral J dt = 0.25
    gives a CZ up to local rotations) lands at ~5 ns — physically reasonable
    for molecular exchange couplings (10-100 MHz) and inside our scan grid.
    """
    tau_close_ps: float = 10.0      # ring-closure (UV) time constant
    tau_open_ns: float = 1.0        # ring-opening (vis) time constant
    J_closed_GHz: float = 0.05      # molecular-scale exchange (~50 MHz)
    J_open_GHz: float = 0.00005     # 1000× suppression when open
    switching_eff: float = 0.95     # photo-conversion efficiency


def photoswitch_coupling_trajectory(t_ns, params: PhotoSwitchParams,
                                     switch_on_at_ns=0.0,
                                     switch_off_at_ns=None):
    """Piecewise coupling J(t) in GHz across the gate window.
    Exponential rise at UV trigger, exponential fall at visible trigger."""
    tau_on_ns = params.tau_close_ps / 1000.0   # convert ps → ns
    tau_off_ns = params.tau_open_ns
    J_hi, J_lo = params.J_closed_GHz, params.J_open_GHz
    eff = params.switching_eff

    J = np.zeros_like(t_ns, dtype=float)
    for i, t in enumerate(t_ns):
        if t < switch_on_at_ns:
            J[i] = J_lo
        elif switch_off_at_ns is not None and t >= switch_off_at_ns:
            dt = t - switch_off_at_ns
            # Exponential decay from (J_hi*eff) toward J_lo
            J[i] = J_lo + (J_hi * eff - J_lo) * np.exp(-dt / tau_off_ns)
        else:
            dt = t - switch_on_at_ns
            J[i] = J_lo + (J_hi * eff - J_lo) * (1 - np.exp(-dt / tau_on_ns))
    return J


def simulate_cz_gate(profile: OrganicProfile,
                      params: PhotoSwitchParams,
                      t_gate_ns=5.0,
                      n_steps=200):
    """
    Simulate a controlled-Z gate implemented via photoswitched ZZ coupling.
    The photoswitch is turned ON at t=0 and OFF at t=t_gate.
    Noise (organic dephasing+depolarizing) accumulates throughout.

    Target: apply U_target = exp(-i π/4 · ZZ) up to local phases.
    Initial state: |++⟩.
    """
    # Initial state: |++⟩ (superposition for both qubits)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi0 = np.kron(plus, plus)
    rho = np.outer(psi0, psi0.conj())
    d = 4

    # Ideal target: apply U_id = exp(-i (π/4) ZZ) on |++⟩
    U_id = expm(-1j * (np.pi / 4) * ZZ)
    rho_target = U_id @ rho @ U_id.conj().T

    # Simulate: at each time-step, Hamiltonian H = J(t) * π/2 * ZZ (GHz units)
    # We also apply incremental dephasing per step proportional to γ*dt/t_gate.
    t = np.linspace(0, t_gate_ns, n_steps)
    dt = t[1] - t[0]
    J_traj = photoswitch_coupling_trajectory(
        t, params, switch_on_at_ns=0.0, switch_off_at_ns=t_gate_ns)

    # Integrate ∫ J dt to check total phase accumulation
    accumulated_phase = np.trapz(J_traj * 2 * np.pi, t)  # rad

    # Per-step γ: fraction of total γ in this dt
    gamma_per_step = profile.gamma * dt / t_gate_ns
    delta_per_step = profile.delta * dt / t_gate_ns

    for k in range(n_steps):
        # Local unitary: H_k = π · J_k · ZZ  (J in GHz, t in ns → dimensionless phase)
        phase_k = 2 * np.pi * J_traj[k] * dt
        U_k = expm(-1j * 0.5 * phase_k * ZZ)
        rho = U_k @ rho @ U_k.conj().T
        # Incremental noise
        step_prof = OrganicProfile(
            name=profile.name + "_step", path_id=profile.path_id,
            material="", gamma=gamma_per_step, delta=delta_per_step,
            T2_us=profile.T2_us, gate_ns=profile.gate_ns, T_op=profile.T_op)
        rho = apply_organic_noise(rho, step_prof)

    F = fidelity(rho_target, rho)
    return dict(
        fid=F,
        purity_final=purity(rho),
        accumulated_phase_rad=float(accumulated_phase),
        target_phase_rad=float(np.pi / 2),   # π/2 is what π/4·ZZ operator gives
        phase_error=float(accumulated_phase - np.pi / 2),
    )


def run_photoswitch_benchmark(profiles, n_gate_times=12):
    """Sweep gate time and measure CZ-gate fidelity for each path.

    With J_closed=0.05 GHz the ideal unitary U=exp(-iπ/4·ZZ) is obtained when
    ∫J dt = 0.25 ⇒ t_opt ≈ 5 ns. The grid covers 0.5-20 ns so the optimum is
    interior.
    """
    if n_gate_times < 2:
        raise ValueError("n_gate_times must be ≥ 2")
    t_grid = np.linspace(0.5, 20.0, n_gate_times)
    results = {"gate_time_ns": t_grid.tolist()}
    params = PhotoSwitchParams()

    for prof in profiles:
        fids, phase_errs = [], []
        for t_g in t_grid:
            res = simulate_cz_gate(prof, params, t_gate_ns=float(t_g),
                                    n_steps=200)
            fids.append(res["fid"])
            phase_errs.append(res["phase_error"])
        results[prof.name] = dict(
            fid=fids,
            phase_error_rad=phase_errs,
            path_id=prof.path_id,
        )
        best_t = t_grid[int(np.argmax(fids))]
        best_f = max(fids)
        print(f"  [photoswitch]  {prof.name:<23} "
              f"best gate time = {best_t:.1f} ns  →  F = {best_f:.4f}")
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
    raise TypeError(f"non-serialisable: {type(obj)}")


def main():
    t0 = time.time()
    print("=" * 78)
    print("ORGANIC QC EXTENDED BENCHMARKS  —  5 next-step items")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    print("\nOrganic profiles (now including Path 3):")
    for p in PROFILES:
        print(f"  Path {p.path_id}: {p.name:<23} γ={p.gamma:<9g} "
              f"T₂={p.T2_us}µs  @{p.T_op:.0f} K  below γ_c? "
              f"{'YES' if p.below_EB else 'no'}")

    # ── 1 & 4 & 2: all-paths algorithm benchmarks (with qDRIFT, Path 3, CIs)
    print("\n" + "─" * 78)
    print("1. ALL-PATHS ALGORITHM BENCHMARKS  (qDRIFT added, Path 3 added, CIs)")
    print("─" * 78)
    all_paths_results = run_all_paths_algorithm_benchmarks(
        PROFILES, n_trials=10, n_rounds=2)

    # ── 1: larger γ-sweep with CI
    print("\n" + "─" * 78)
    print("2. LARGE γ-SWEEP WITH 95% CIs  (15 grid points × 10 trials)")
    print("─" * 78)
    gamma_grid = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05,
                  0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]
    gamma_sweep = run_gamma_sweep_with_ci(gamma_grid, n_trials=10, n_rounds=2)

    # ── 3: full MNIST
    print("\n" + "─" * 78)
    print("3. FULL MNIST  (sklearn digits, 1797 samples, 5-fold CV)")
    print("─" * 78)
    mnist = run_mnist_full(PROFILES, n_splits=5, d=8)

    # ── 5: photoswitch gate
    print("\n" + "─" * 78)
    print("4. DIARYLETHENE PHOTOSWITCH GATE MODEL  (CZ via ZZ photoswitched)")
    print("─" * 78)
    photoswitch = run_photoswitch_benchmark(PROFILES, n_gate_times=10)

    total_s = time.time() - t0
    print(f"\n  Total wallclock: {total_s:.1f} s")

    payload = {
        "meta": {
            "date": datetime.now().isoformat(),
            "seed": SEED,
            "wallclock_s": total_s,
            "has_sklearn": HAS_SKLEARN,
            "n_trials_alg": 10,
            "n_trials_gamma_sweep": 10,
            "mnist_n_splits": 5,
        },
        "profiles": [asdict(p) for p in PROFILES],
        "algorithm_benchmarks_all_paths": all_paths_results,
        "gamma_sweep_with_ci": gamma_sweep,
        "mnist_full": mnist,
        "photoswitch_gate": photoswitch,
    }
    out_path = os.path.join(HERE, "results",
                             "organic_benchmarks_extended.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\n  Saved → {out_path}")

    # ── Summary tables ──────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("SUMMARY: Algorithm × Path fidelities (mean ± 95% CI)")
    print("═" * 78)
    print(f"\n  {'Alg':<11} {'Profile':<24} {'F_noisy':>16} {'F_cqec':>16} {'ΔF':>8}")
    print("  " + "─" * 75)
    for r in all_paths_results:
        print(f"  {r['algorithm']:<11} {r['profile']:<24} "
              f"{r['fid_noisy_mean']:.4f}±{r['fid_noisy_ci']:.4f}  "
              f"{r['fid_cqec_mean']:.4f}±{r['fid_cqec_ci']:.4f}  "
              f"{r['fid_cqec_mean']-r['fid_noisy_mean']:+.4f}")

    print("\n  MNIST full (5-fold CV, 1797 samples):")
    for key, v in mnist.items():
        if isinstance(v, dict) and "acc_mean" in v:
            print(f"    {key:<35} acc={v['acc_mean']:.4f}±{v['acc_ci']:.4f}   "
                  f"F1={v['f1_mean']:.4f}±{v['f1_ci']:.4f}")

    print("\n  Photoswitch gate: best fidelity per path")
    for k, v in photoswitch.items():
        if isinstance(v, dict) and "fid" in v:
            best_i = int(np.argmax(v["fid"]))
            print(f"    {k:<24} best F={max(v['fid']):.4f}  "
                  f"at t_gate={photoswitch['gate_time_ns'][best_i]:.1f} ns")

    return payload


if __name__ == "__main__":
    main()
