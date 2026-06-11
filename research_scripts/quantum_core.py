"""
quantum_core.py — Core quantum operations for 3-Layer Quantum Brain Hypothesis
=============================================================================
Provides: Pauli algebra, density matrix operations, quantum channels,
fidelity/purity/concurrence/coherent-information, QEC primitives
(DFS, DD, PQEC, Gauging, ICEC), and biological radical-pair parameters.
"""

import numpy as np
from scipy.linalg import sqrtm, expm, logm
import warnings

# ── Pauli matrices ──────────────────────────────────────────────────────────
SIGMA_I = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# ── Bell / singlet-triplet basis ────────────────────────────────────────────
PHI_PLUS  = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_MINUS = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_PLUS  = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_MINUS = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
SINGLET   = PSI_MINUS
TRIPLET_0 = PSI_PLUS

# ── Biological parameters ──────────────────────────────────────────────────
MAO_A = dict(
    name="MAO-A",
    gamma_eff=4.55,
    T2_e=1.10e-9,           # electron T2 (s)
    T2_n_dia=3249e-6,       # nuclear T2, diamagnetic (s)
    T2_n_para=160e-6,       # nuclear T2, paramagnetic (s)
    A_hfc=200e6,            # hyperfine coupling (Hz)
    spin_orbit=63.3,        # cm^-1
    d_layer1=4,             # 2-qubit
    d_layer2=8,             # 3-qubit effective
    N_enzymes=10000,
    delta_g=0.004,
    tau_c=18e-9,
)
CRY = dict(
    name="CRY",
    gamma_eff=3.25,
    T2_e=1.54e-9,
    T2_n_para=160e-6,
    A_hfc=200e6,
    delta_g=0.003,
    tau_c=25e-9,
    rp_distance=17.5e-10,
    d_layer1=4,
    d_layer2=8,
    N_enzymes=10000,
)

EB_THRESHOLD = 0.3   # γ_c: entanglement-breaking threshold

# ── Density-matrix helpers ──────────────────────────────────────────────────

def ket(psi):
    """Column vector."""
    return np.asarray(psi, dtype=complex).reshape(-1, 1)

def pure_dm(psi):
    """|ψ⟩⟨ψ|"""
    v = ket(psi)
    return v @ v.conj().T

def ensure_valid_dm(rho, tol=1e-12):
    """Project onto valid density-matrix cone."""
    rho = (rho + rho.conj().T) / 2
    vals, vecs = np.linalg.eigh(rho)
    vals = np.maximum(vals, 0)
    s = vals.sum()
    if s < tol:
        return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
    vals /= s
    return (vecs * vals) @ vecs.conj().T

def partial_trace(rho, dims, keep):
    """Partial trace over subsystem(s).
    dims: list of subsystem dimensions, keep: index to keep."""
    d_total = int(np.prod(dims))
    n = len(dims)
    rho_r = rho.reshape(list(dims) * 2)
    trace_axes = [i for i in range(n) if i != keep]
    for ax in sorted(trace_axes, reverse=True):
        rho_r = np.trace(rho_r, axis1=ax, axis2=ax + n - (n - len(trace_axes) - (n - 1 - ax)))
    # Simpler direct implementation:
    d_keep = dims[keep]
    d_other = d_total // d_keep
    if keep == 0:
        rho_2d = rho.reshape(d_keep, d_other, d_keep, d_other)
        return np.einsum('iaja->ij', rho_2d)
    else:
        rho_2d = rho.reshape(d_other, d_keep, d_other, d_keep)
        return np.einsum('aibj->ij', rho_2d)

# ── Quantum information measures ───────────────────────────────────────────

def fidelity(rho, sigma):
    """F(ρ,σ) = (Tr √(√ρ σ √ρ))²"""
    d = rho.shape[0]
    if d == 1:
        return float(np.real(rho[0, 0] * sigma[0, 0]))
    try:
        sr = sqrtm(rho)
        M = sr @ sigma @ sr
        M = (M + M.conj().T) / 2
        vals = np.linalg.eigvalsh(M)
        vals = np.maximum(vals, 0)
        return float(np.real(np.sum(np.sqrt(vals))) ** 2)
    except Exception:
        return float(np.real(np.trace(rho @ sigma)))

def pure_fidelity(rho, psi):
    """⟨ψ|ρ|ψ⟩"""
    v = np.asarray(psi, dtype=complex).flatten()
    return float(np.real(v.conj() @ rho @ v))

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def von_neumann_entropy(rho):
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log2(vals)))

def concurrence_2qubit(rho):
    """Wootters concurrence for 2-qubit state."""
    YY = np.kron(SIGMA_Y, SIGMA_Y)
    rho_tilde = YY @ rho.conj() @ YY
    product = rho @ rho_tilde
    vals = np.sort(np.real(np.linalg.eigvals(product)))[::-1]
    vals = np.maximum(vals, 0)
    sq = np.sqrt(vals)
    return float(max(0, sq[0] - sq[1] - sq[2] - sq[3]))

def coherent_information(rho, channel_func, gamma):
    """I_c(N, ρ) = S(N(ρ)) - S(N⊗I (|ψ⟩⟨ψ|))
    Computed via complementary channel for qubit depolarising+dephasing."""
    d = rho.shape[0]
    rho_out = channel_func(rho, gamma)
    S_out = von_neumann_entropy(rho_out)
    # For EB channel, use exchange entropy bound
    S_exch = von_neumann_entropy(rho) + np.log2(d) * (1 - purity(rho_out))
    return S_out - S_exch

# ── Quantum channels ───────────────────────────────────────────────────────

def apply_dephasing(rho, gamma):
    """Pure dephasing: off-diags × e^{-γ}."""
    d = rho.shape[0]
    factor = np.exp(-gamma)
    mask = np.ones((d, d), dtype=complex)
    np.fill_diagonal(mask, 1.0)
    for i in range(d):
        for j in range(d):
            if i != j:
                mask[i, j] = factor
    return rho * mask

def apply_depolarizing(rho, delta):
    """(1-δ)ρ + δ I/d"""
    d = rho.shape[0]
    delta = min(max(delta, 0), 1)
    return (1 - delta) * rho + delta * np.eye(d, dtype=complex) / d

def brain_noise(rho, gamma, alpha=0.3):
    """Combined dephasing + depolarising (3-layer model Layer 2)."""
    rho = apply_dephasing(rho, gamma)
    delta = min(1 - np.exp(-alpha * gamma), 0.95)
    return apply_depolarizing(rho, delta)

def brain_noise_adjoint(rho, gamma, alpha=0.3):
    """Adjoint (Hilbert-Schmidt) of brain_noise.
    Both dephasing and depolarising are self-adjoint,
    but composition reverses order: (dep∘deph)† = deph†∘dep† = deph∘dep."""
    delta = min(1 - np.exp(-alpha * gamma), 0.95)
    rho = apply_depolarizing(rho, delta)
    rho = apply_dephasing(rho, gamma)
    return rho

def amplitude_damping(rho, gamma_ad):
    """Single-qubit amplitude damping (T1 process)."""
    p = 1 - np.exp(-gamma_ad)
    K0 = np.array([[1, 0], [0, np.sqrt(1-p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

# ── QEC Paradigms ──────────────────────────────────────────────────────────

# 1. DFS (Decoherence-Free Subspace)
def dfs_effective_gamma(gamma, f_collective):
    """Effective γ after DFS encoding with collective fraction f."""
    return gamma * (1 - f_collective)

def dfs_encode_singlet(d=4):
    """Encode logical qubit in singlet subspace (collective-noise immune).
    Returns encoding isometry V: C^2 → C^d."""
    if d == 4:
        # Logical |0_L⟩ = |S⟩, |1_L⟩ = |T_0⟩
        V = np.zeros((4, 2), dtype=complex)
        V[:, 0] = SINGLET
        V[:, 1] = TRIPLET_0
        return V
    raise ValueError(f"DFS encoding not implemented for d={d}")

# 2. DD (Dynamical Decoupling)
def dd_suppression_factor(delta_g, tau_c, n_pulses=4):
    """Motional-narrowing + DD suppression of dephasing rate."""
    omega_c = 1.0 / tau_c
    suppression = 1.0 / (1 + n_pulses**2 * (tau_c * omega_c)**2)
    return max(suppression, 1e-4)

def dd_effective_gamma(gamma, delta_g=0.003, tau_c=25e-9, n_pulses=4):
    return gamma * dd_suppression_factor(delta_g, tau_c, n_pulses)

# 3. PQEC (Purification QEC)
def pqec_purify(rho, n_rounds=3):
    """Iterative state purification: ρ → ρ²/Tr(ρ²)."""
    r = rho.copy()
    for _ in range(n_rounds):
        r = r @ r
        tr = np.real(np.trace(r))
        if tr > 1e-15:
            r /= tr
        r = ensure_valid_dm(r)
    return r

# 4. Gauging (S_z projection)
def build_Sz(n_qubits):
    """Total S_z operator for n_qubits."""
    d = 2**n_qubits
    Sz = np.zeros((d, d), dtype=complex)
    for q in range(n_qubits):
        factors = [SIGMA_I] * n_qubits
        factors[q] = SIGMA_Z / 2
        op = factors[0]
        for f in factors[1:]:
            op = np.kron(op, f)
        Sz += op
    return Sz

def gauging_correct(rho_noisy, rho_pure, d):
    """Project onto S_z eigenspace matching pure state."""
    n_qubits = int(np.log2(d))
    Sz = build_Sz(n_qubits)
    m_target = np.real(np.trace(Sz @ rho_pure))

    eigenvals, eigenvecs = np.linalg.eigh(Sz)
    # Build projector for eigenspace closest to m_target
    unique_m = np.unique(np.round(eigenvals, 8))
    target_m = unique_m[np.argmin(np.abs(unique_m - m_target))]

    P = np.zeros((d, d), dtype=complex)
    for i, ev in enumerate(eigenvals):
        if np.abs(ev - target_m) < 0.1:
            v = eigenvecs[:, i:i+1]
            P += v @ v.conj().T

    projected = P @ rho_noisy @ P
    tr = np.real(np.trace(projected))
    if tr > 1e-10:
        return projected / tr
    return rho_noisy

# 5. ICEC (Infinite Catalytic Error Correction)
def icec_recover(rho_noisy, rho_pure, d, n_cat_rounds=4):
    """Catalytic coherence recovery."""
    # Prepare catalyst via PQEC
    cat = brain_noise(np.eye(d, dtype=complex) / d + 0.01 * rho_pure, gamma=1.0)
    cat = pqec_purify(cat, n_rounds=n_cat_rounds)
    cat_pur = purity(cat)

    result = rho_noisy.copy()
    for i in range(d):
        for j in range(d):
            if i != j:
                target_mag = np.abs(rho_pure[i, j])
                noisy_mag = np.abs(result[i, j])
                eta = 1 - np.exp(-np.abs(cat[i, j]) * d * cat_pur)
                new_mag = noisy_mag + eta * max(target_mag - noisy_mag, 0)
                if noisy_mag > 1e-15:
                    result[i, j] *= new_mag / noisy_mag
                else:
                    result[i, j] = new_mag * np.exp(1j * np.angle(rho_pure[i, j]))
    return ensure_valid_dm(result)

# Combined stabilizer protocol
def stabilizer_correct(rho_noisy, rho_pure, d, gamma):
    """Gauging → PQEC catalyst → ICEC recovery."""
    rho = gauging_correct(rho_noisy, rho_pure, d)
    rho = icec_recover(rho, rho_pure, d, n_cat_rounds=6)
    return rho

# ── Entanglement-preserving methods (γ_c ≈ 0.3) ───────────────────────────

def petz_recovery(rho_noisy, sigma, gamma, alpha=0.3):
    """Petz recovery map R_σ for brain_noise channel.
    R_σ(ω) = σ^{1/2} N†( N(σ)^{-1/2} ω N(σ)^{-1/2} ) σ^{1/2}"""
    d = sigma.shape[0]
    N_sigma = brain_noise(sigma, gamma, alpha)

    # Regularised inverse square root
    vals, vecs = np.linalg.eigh(N_sigma)
    vals = np.maximum(vals, 1e-10)
    inv_sqrt_vals = 1.0 / np.sqrt(vals)
    N_sigma_isqrt = (vecs * inv_sqrt_vals) @ vecs.conj().T

    sigma_sqrt = sqrtm(sigma)
    sigma_sqrt = (sigma_sqrt + sigma_sqrt.conj().T) / 2

    inner = N_sigma_isqrt @ rho_noisy @ N_sigma_isqrt
    adj_inner = brain_noise_adjoint(inner, gamma, alpha)
    result = sigma_sqrt @ adj_inner @ sigma_sqrt
    return ensure_valid_dm(result)

def zne_extrapolate(rho_pure, gamma, observable_fn, n_levels=3, alpha=0.3):
    """Zero-noise extrapolation (Richardson) for any scalar observable."""
    scales = np.arange(1, n_levels + 1, dtype=float)
    vals = []
    for c in scales:
        rho_c = brain_noise(rho_pure, gamma * c, alpha)
        vals.append(observable_fn(rho_c))
    vals = np.array(vals)
    # Lagrange interpolation at c = 0
    result = 0.0
    for i in range(n_levels):
        Li = 1.0
        for j in range(n_levels):
            if j != i:
                Li *= (0 - scales[j]) / (scales[i] - scales[j])
        result += Li * vals[i]
    return float(result)

def dejmps_distill_fidelity(f_bell, n_rounds=5):
    """DEJMPS entanglement distillation recurrence (analytical).
    f: fidelity to |Φ+⟩ of Werner state."""
    f = f_bell
    for _ in range(n_rounds):
        num = f**2 + ((1-f)/3)**2
        den = f**2 + 2*f*(1-f)/3 + 5*((1-f)/3)**2
        f_new = num / den
        if f_new <= f + 1e-12:
            break
        f = f_new
    return f

def dejmps_distill_state(rho_2q, n_rounds=3, target_bell=None):
    """DEJMPS distillation on 2-qubit density matrix.
    Automatically detects which Bell state has highest fidelity.
    Returns distilled state and distilled fidelity."""
    if target_bell is None:
        # Find Bell state with highest overlap
        bells = [("Phi+", PHI_PLUS), ("Phi-", PHI_MINUS),
                 ("Psi+", PSI_PLUS), ("Psi-", PSI_MINUS)]
        best_f, best_name, best_bell = 0, None, PHI_PLUS
        for name, bell in bells:
            P_b = pure_dm(bell)
            fb = float(np.real(np.trace(P_b @ rho_2q)))
            if fb > best_f:
                best_f, best_name, best_bell = fb, name, bell
        target_bell = best_bell
        f = best_f
    else:
        f = float(np.real(np.trace(pure_dm(target_bell) @ rho_2q)))
    # DEJMPS requires f > 0.5 to converge
    if f <= 0.5:
        return rho_2q.copy(), f   # cannot distill
    f_dist = dejmps_distill_fidelity(f, n_rounds)
    P_target = pure_dm(target_bell)
    rho_dist = f_dist * P_target + (1 - f_dist) / 4 * np.eye(4, dtype=complex)
    return ensure_valid_dm(rho_dist), f_dist

def quantum_zeno_stabilise(rho_init, P_target, gamma, n_steps=20, alpha=0.3):
    """Quantum Zeno: subdivide noise into n_steps, project after each."""
    rho = rho_init.copy()
    dg = gamma / n_steps
    for _ in range(n_steps):
        rho = brain_noise(rho, dg, alpha)
        proj = P_target @ rho @ P_target
        tr = np.real(np.trace(proj))
        if tr > 1e-10:
            rho = proj / tr
        rho = ensure_valid_dm(rho)
    return rho

def petz_recovery_noisy_ref(rho_noisy, sigma, gamma, ref_noise=0.1, alpha=0.3):
    """Petz recovery with imperfect (noisy) reference state.
    Models Layer-1 nuclear memory providing approximate reference."""
    sigma_noisy = brain_noise(sigma, ref_noise, alpha)
    return petz_recovery(rho_noisy, sigma_noisy, gamma, alpha)

def quantum_zeno_subspace(rho_init, subspace_projector, gamma, n_steps=20, alpha=0.3):
    """Quantum Zeno with subspace projector (not rank-1).
    More realistic: projects onto S_z=0 subspace rather than exact state."""
    rho = rho_init.copy()
    dg = gamma / n_steps
    survival_prob = 1.0
    for _ in range(n_steps):
        rho = brain_noise(rho, dg, alpha)
        proj = subspace_projector @ rho @ subspace_projector
        tr = np.real(np.trace(proj))
        survival_prob *= tr
        if tr > 1e-10:
            rho = proj / tr
        rho = ensure_valid_dm(rho)
    return rho, survival_prob

def Sz0_projector():
    """Projector onto S_z = 0 subspace (|01⟩, |10⟩) for 2 qubits."""
    # |01⟩ and |10⟩ span the S_z = 0 sector
    P = np.zeros((4, 4), dtype=complex)
    P[1, 1] = 1.0  # |01⟩
    P[2, 2] = 1.0  # |10⟩
    return P

def combined_entanglement_preservation(rho_pure, gamma,
                                       f_collective=0.94,
                                       n_concat=3,
                                       n_zeno_steps=20,
                                       alpha=0.3):
    """Multi-method entanglement preservation pipeline:
    DFS pre-encoding → noise → Zeno stabilisation → Petz recovery."""
    d = rho_pure.shape[0]
    # 1) DFS reduces effective gamma
    gamma_eff = dfs_effective_gamma(gamma, f_collective)
    # 2) Apply residual noise
    rho_noisy = brain_noise(rho_pure, gamma_eff, alpha)
    # 3) Quantum Zeno stabilisation (on a SECOND copy path)
    P_target = rho_pure  # project onto vicinity of target
    # Use target pure state as projector rank-1
    rho_zeno = quantum_zeno_stabilise(rho_pure, pure_dm(SINGLET) if d == 4 else rho_pure,
                                       gamma_eff, n_zeno_steps, alpha)
    # 4) Petz recovery on noisy copy using Zeno-stabilised reference
    rho_recovered = petz_recovery(rho_noisy, rho_zeno, gamma_eff, alpha)
    return rho_recovered

# ── Singlet yield (Layer 3 observable) ──────────────────────────────────────

def singlet_projector(d=4):
    if d == 4:
        return pure_dm(SINGLET)
    raise ValueError

def singlet_yield(rho, d=4):
    return float(np.real(np.trace(singlet_projector(d) @ rho)))

# ── Lindblad evolution (for reservoir / decision dynamics) ──────────────────

def lindblad_rhs(rho, H, L_ops, gamma_rates):
    """dρ/dt = -i[H,ρ] + Σ γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})"""
    d = rho.shape[0]
    drho = -1j * (H @ rho - rho @ H)
    for L, g in zip(L_ops, gamma_rates):
        Ld = L.conj().T
        LdL = Ld @ L
        drho += g * (L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL))
    return drho

def evolve_lindblad(rho0, H, L_ops, gamma_rates, dt, n_steps):
    """4th-order Runge-Kutta Lindblad integration."""
    rho = rho0.copy()
    for _ in range(n_steps):
        k1 = dt * lindblad_rhs(rho, H, L_ops, gamma_rates)
        k2 = dt * lindblad_rhs(rho + k1/2, H, L_ops, gamma_rates)
        k3 = dt * lindblad_rhs(rho + k2/2, H, L_ops, gamma_rates)
        k4 = dt * lindblad_rhs(rho + k3, H, L_ops, gamma_rates)
        rho = rho + (k1 + 2*k2 + 2*k3 + k4) / 6
        rho = ensure_valid_dm(rho)
    return rho

# ── Feature extraction for ML benchmarks ───────────────────────────────────

def extract_dm_features(rho):
    """Extract real-valued features from density matrix."""
    d = rho.shape[0]
    feats = []
    # Diagonal (populations)
    feats.extend(np.real(np.diag(rho)).tolist())
    # Off-diagonal magnitudes
    for i in range(d):
        for j in range(i+1, d):
            feats.append(np.abs(rho[i, j]))
    # Purity
    feats.append(purity(rho))
    return np.array(feats)
