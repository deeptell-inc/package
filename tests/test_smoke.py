"""Smoke tests — verify that core primitives, profiles, and one
small benchmark all work end-to-end."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

# numpy emits divide-by-zero RuntimeWarnings on extreme low-rank
# projector multiplications; they are harmless for the smoke tests.
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
)


def test_package_imports():
    import organic_qc_bench as oqb
    assert hasattr(oqb, "__version__")
    assert hasattr(oqb, "PROFILES")


def test_profiles_have_4_paths():
    from organic_qc_bench import PROFILES
    assert set(PROFILES) == {"Path1_RPRes", "Path2_PTM",
                             "Path3_OrgSC", "Path4_SSH"}
    for p in PROFILES.values():
        assert p.gamma >= 0
        assert 0 <= p.delta <= 1


def test_noise_channel_is_trace_preserving():
    from organic_qc_bench import organic_noise
    d = 4
    psi = np.array([1, 0.5, -0.5j, 0.3], dtype=complex)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    out = organic_noise(rho, gamma=0.3, delta=0.1)
    assert math.isclose(float(np.real(np.trace(out))), 1.0, abs_tol=1e-8)


def test_cqec_increases_fidelity_for_bell_state():
    from organic_qc_bench import (
        organic_noise, fidelity, recursive_covariant, cqec_recovery,
    )
    # Bell state |Phi+⟩
    rho_t = np.zeros((4, 4), dtype=complex)
    rho_t[0, 0] = rho_t[0, 3] = rho_t[3, 0] = rho_t[3, 3] = 0.5
    rho_n = organic_noise(rho_t, gamma=0.3, delta=0.1)
    rho_cat, _, _ = recursive_covariant(rho_n.copy(), 4, 2)
    rho_c = cqec_recovery(rho_t, rho_n, rho_cat)
    f_n = fidelity(rho_t, rho_n)
    f_c = fidelity(rho_t, rho_c)
    assert f_c >= f_n  # CQEC should not make things worse


def test_state_factories_return_pure_states():
    from organic_qc_bench import make_qkan, make_bell, make_ghz
    for maker in (make_qkan, make_bell, lambda: make_ghz(3)):
        rho, d = maker()
        assert rho.shape == (d, d)
        # purity ~ 1 for pure states
        assert math.isclose(float(np.real(np.trace(rho @ rho))), 1.0,
                            abs_tol=1e-8)


def test_bv_one_query():
    from organic_qc_bench import PROFILES
    from organic_qc_bench.bv import bv_run
    rho, rho_t, p_s = bv_run(s=3, n=3, profile=PROFILES["Path2_PTM"],
                              apply_cqec=False)
    assert rho.shape == (8, 8)
    assert 0.0 <= p_s <= 1.0


def test_bv_classical_baseline():
    from organic_qc_bench.bv import classical_one_query_rate
    assert math.isclose(classical_one_query_rate(5), 1/32)


def test_peak_scaling_small():
    from organic_qc_bench.peak import sweep_gamma, find_peak
    sweep = sweep_gamma(d=2, gammas=np.logspace(-2, 0.5, 6), n_trials=2)
    gp, df = find_peak(sweep)
    assert isinstance(gp, float)
    assert isinstance(df, float)


def test_photoswitch_cz_runs():
    from organic_qc_bench import PROFILES
    from organic_qc_bench.photoswitch import simulate_cz
    res = simulate_cz(PROFILES["Path2_PTM"], t_gate_ns=5.0, n_steps=40)
    assert 0.0 <= res["fid"] <= 1.0


def test_svilc_metric_runs():
    from organic_qc_bench.svilc import (
        TriangularLattice, spin_vortex_field, phase_frustration_coupling,
    )
    lat = TriangularLattice(Lx=8, Ly=6, tp=0.8)
    chi_a = spin_vortex_field(lat, [(2, 2), (4, 2)], [+1, -1])
    chi_b = spin_vortex_field(lat, [(5, 2), (7, 2)], [+1, -1])
    V = phase_frustration_coupling(lat, chi_a, chi_b)
    assert isinstance(V, float)


def test_cli_info():
    from organic_qc_bench.cli import main
    rc = main(["info"])
    assert rc == 0
