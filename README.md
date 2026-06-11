# organic-qc-bench

[![PyPI version](https://img.shields.io/pypi/v/organic-qc-bench.svg)](https://pypi.org/project/organic-qc-bench/)
[![CI](https://github.com/WakauraH/organic-qc-bench/actions/workflows/test.yml/badge.svg)](https://github.com/WakauraH/organic-qc-bench/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Benchmark suite for engineered organic quantum platforms — Python
package supporting the manuscript

> H. Wakaura and T. Tanimae,
> *Covariant Error Correction Peaks at the Entanglement-Breaking
> Threshold: A Unified Benchmark of Four Magnetic-Field-Free
> Organic Quantum Platforms*,
> QIRI (Quantum Integrated Research Institute Inc.), 2026.

Companion preprint (3-Layer Quantum Brain Hypothesis):
[Wakaura, *Research Square* (2026), DOI 10.21203/rs.3.rs-9278975/v1](https://doi.org/10.21203/rs.3.rs-9278975/v1).

## What this package contains

| Module | Purpose |
|---|---|
| `organic_qc_bench.core` | Noise channels, fidelity / concurrence, swap-test purification, CQEC recovery |
| `organic_qc_bench.profiles` | The four `OrganicProfile` instances (P1–P4) |
| `organic_qc_bench.states` | State-prep factories: `make_qkan`, `make_qdrift`, `make_cfqpe`, `make_regev`, `make_bell`, `make_ghz` |
| `organic_qc_bench.bv` | Bernstein-Vazirani end-to-end benchmark with CQEC |
| `organic_qc_bench.peak` | CQEC fidelity-gain peak vs state dimension *d* |
| `organic_qc_bench.photoswitch` | Diarylethene-photoswitched CZ-gate simulator |
| `organic_qc_bench.svilc` | κ-(BEDT-TTF)₂X triangular-lattice SVILC analysis |
| `organic_qc_bench.cli` | `organic-qc-bench` console script |

## Installation

```bash
# PyPI
pip install organic-qc-bench

# with ML extras (sklearn for the MNIST/time-series benchmarks)
pip install "organic-qc-bench[ml]"

# with plotting (matplotlib for figure regeneration)
pip install "organic-qc-bench[figures]"

# from source
git clone https://github.com/WakauraH/organic-qc-bench
cd organic-qc-bench
pip install -e ".[dev]"
```

Requires Python ≥ 3.9, NumPy ≥ 1.22, SciPy ≥ 1.8.

## Quick start

```python
import organic_qc_bench as oqb

# 1. Inspect the four organic-material noise profiles
for name, p in oqb.PROFILES.items():
    print(name, p.gamma, p.delta, p.material)

# 2. Run a one-query Bernstein-Vazirani circuit on the PTM-COF profile
from organic_qc_bench.bv import bv_run, benchmark
rho_final, rho_target, p_s = bv_run(
    s=5, n=4, profile=oqb.PROFILES["Path2_PTM"], apply_cqec=True,
)
print(f"P(s) = {p_s:.4f}")

# 3. Find the CQEC fidelity-gain peak for d=8
import numpy as np
from organic_qc_bench.peak import sweep_gamma, find_peak
sweep = sweep_gamma(d=8, gammas=np.logspace(-2, 0.7, 24), n_trials=10)
gamma_peak, delta_f_max = find_peak(sweep)
print(f"gamma_peak = {gamma_peak:.3f}, Delta F_max = {delta_f_max:.4f}")

# 4. Diarylethene CZ-gate fidelity for the κ-BEDT-TTF SVILC profile
from organic_qc_bench.photoswitch import sweep_gate_time
res = sweep_gate_time(oqb.PROFILES["Path3_OrgSC"])
print(f"best F_CZ = {res['best_fid']:.4f} at "
      f"t_gate = {res['best_gate_time_ns']:.2f} ns")
```

## Command-line interface

```bash
# Show package info and all noise profiles
organic-qc-bench info

# 100-trial Bernstein-Vazirani at n=5 on every profile
organic-qc-bench bv -n 5 -t 100 -o bv_results.json

# CQEC peak scaling sweep over d
organic-qc-bench peak --dims 2 4 8 16 -t 10 -o peak.json

# Photoswitch CZ gate scan on Path 2 (PTM-COF)
organic-qc-bench photoswitch --profile Path2_PTM -o ps.json

# κ-(BEDT-TTF) SVILC two-SVQ coupling vs external feed current
organic-qc-bench svilc --Lx 16 --Ly 6 --distance 10 -o svilc.json
```

## Reproducing manuscript figures

All numerical results in the paper use seed 42. To reproduce the
main numbers in Table I and Figs. 1–11, run:

```bash
organic-qc-bench bv     -n 5 -t 100 -o bv_n5.json
organic-qc-bench peak   --dims 2 4 8 16 32 64 -t 10 -o peak.json
organic-qc-bench photoswitch --profile Path2_PTM -o ps_p2.json
organic-qc-bench photoswitch --profile Path3_OrgSC -o ps_p3.json
organic-qc-bench photoswitch --profile Path4_SSH -o ps_p4.json
organic-qc-bench svilc  --Lx 16 --Ly 6 --distance 10 -o svilc.json
```

## Project layout

```
organic-qc-bench/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/organic_qc_bench/
│   ├── __init__.py
│   ├── core.py           # quantum primitives + CQEC
│   ├── profiles.py       # Path 1-4 noise profiles
│   ├── states.py         # algorithm state preparations
│   ├── bv.py             # Bernstein-Vazirani benchmark
│   ├── peak.py           # CQEC peak vs d
│   ├── photoswitch.py    # diarylethene CZ gate
│   ├── svilc.py          # κ-(BEDT-TTF) lattice
│   └── cli.py            # console script
└── tests/test_smoke.py   # smoke tests (pytest)
```

## Citing

If you use this package, please cite both the main manuscript and the
companion 3-LQBH preprint:

```bibtex
@misc{Wakaura2026LQBH,
  author = {Wakaura, Hikaru},
  title  = {3-Layer Quantum Brain Hypothesis: Covariant Error
            Correction, Dynamical Decoupling and Petz Recovery in
            Biological Radical-Pair Systems},
  year   = {2026},
  doi    = {10.21203/rs.3.rs-9278975/v1},
  url    = {https://doi.org/10.21203/rs.3.rs-9278975/v1},
}
```

## Testing

The package ships with eleven `pytest` smoke tests that run in under
a second:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

A GitHub Actions matrix runs the suite on `ubuntu-latest` and
`macos-latest` across Python 3.9, 3.10, 3.11 and 3.12 on every push;
see [`.github/workflows/test.yml`](.github/workflows/test.yml).

## Contributing

Bug reports, feature requests and pull requests are welcome on the
GitHub issue tracker.

* Please open an issue before sending a large patch so that the
  scope can be agreed up front.
* Code style: PEP 8 with `from __future__ import annotations`. Run
  `python -m pytest tests/` before submitting; new functionality
  must come with at least one smoke test.
* All public functions carry type annotations and a short docstring.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

This work was supported by the Quantum Integrated Research Institute
Inc. (QIRI), Tokyo, Japan. We thank colleagues at QIRI for valuable
discussions.
