"""organic_qc_bench — benchmark suite for engineered organic quantum platforms.

A pip-installable package supporting the manuscript
"Covariant Error Correction Peaks at the Entanglement-Breaking Threshold:
A Unified Benchmark of Four Magnetic-Field-Free Organic Quantum Platforms"
by H. Wakaura and T. Tanimae (QIRI, 2026).
"""
from importlib.metadata import PackageNotFoundError, version

from .core import (
    fidelity,
    purity,
    l1_coherence,
    concurrence,
    dephasing_channel,
    depolarizing_channel,
    organic_noise,
    swap_test_purify,
    recursive_covariant,
    cqec_recovery,
)
from .profiles import PROFILES, OrganicProfile
from .states import (
    make_qkan,
    make_qdrift,
    make_cfqpe,
    make_regev,
    make_bell,
    make_ghz,
)

try:
    __version__ = version("organic-qc-bench")
except PackageNotFoundError:  # not installed (running from source tree)
    __version__ = "0.1.0+local"

__all__ = [
    "__version__",
    # core
    "fidelity", "purity", "l1_coherence", "concurrence",
    "dephasing_channel", "depolarizing_channel", "organic_noise",
    "swap_test_purify", "recursive_covariant", "cqec_recovery",
    # profiles
    "PROFILES", "OrganicProfile",
    # state factories
    "make_qkan", "make_qdrift", "make_cfqpe", "make_regev",
    "make_bell", "make_ghz",
]
