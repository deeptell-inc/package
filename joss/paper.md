---
title: "organic-qc-bench: A reproducible benchmark suite for Petz-style covariant recovery on dephasing--depolarizing channels"
tags:
  - Python
  - quantum information
  - quantum error correction
  - Petz recovery map
  - covariant codes
  - density-matrix simulation
authors:
  - name: Hikaru Wakaura
    orcid: 0000-0001-8381-8323
    corresponding: true
    affiliation: 1
  - name: Taiki Tanimae
    affiliation: 1
affiliations:
  - name: QIRI (Quantum Integrated Research Institute Inc.), 1--16--3 Akasaka, Minato-ku, Tokyo 107--0061, Japan
    index: 1
date: 10 June 2026
bibliography: paper.bib
---

# Summary

`organic-qc-bench` is a small, pure-Python package that benchmarks a
CPTP, Petz-style covariant-purification recovery map on the
two-parameter dephasing--depolarizing channel class
$\mathcal{N}_\gamma^\delta=\mathcal{E}_\delta\circ\mathcal{D}_\gamma$.
The package implements:

* a vendored minimum of covariant quantum-error-correction (CQEC)
  primitives (Uhlmann fidelity, purity, $\ell_1$-coherence,
  Wootters concurrence, symmetric SWAP-test purification, and a
  CPTP recovery map obtained by composing a Petz-style off-diagonal
  update with a positive-projection wrapper);
* four reusable noise-parameter profiles corresponding to physically
  motivated organic-qubit hosts;
* algorithmic state factories (QKAN, qDRIFT, control-free QPE,
  Shor--Regev, Bell, GHZ);
* a single-shot Bernstein--Vazirani benchmark with Wilson confidence
  intervals;
* a $\gamma$-sweep / peak-scaling driver that locates the
  fidelity-gain peak $\gamma_{\rm peak}(d)$ and its magnitude
  $\Delta F_{\rm max}(d)$ as a function of Hilbert-space dimension
  $d$;
* a diarylethene-photoswitched CZ-gate model
  [@Irie2000; @Ferrando2016];
* a spin-vortex-induced-loop-current (SVILC) lattice analyzer
  [@Wakaura2017];
* a command-line interface (`organic-qc-bench {info,bv,peak,
  photoswitch,svilc}`).

The package ships with eleven smoke tests, a continuous-integration
matrix across Ubuntu/macOS and Python 3.9--3.12, and a single random
seed (42) wired through every result, so that every figure and table
in the companion preprint can be regenerated with a single invocation.

# Statement of need

Approximate covariant quantum error correction and Petz-style
recovery maps are now a mature area of mathematical
quantum-information theory
[@Petz1986; @SutterBertaTomamichel2016; @JungeSutterWilde2018].
However, *implementable* and *reproducible* reference codes for
benchmarking such recovery maps on physically motivated noise channels
are still scarce. Researchers who wish to study approximate recovery
on near-entanglement-breaking channels must therefore re-derive the
recovery formulae and rewrite the boilerplate every time.

`organic-qc-bench` fills this gap by providing a focused,
dependency-light, test-covered reference implementation. It is
intended for two audiences:

1. Quantum-information theorists who want a quickly modifiable
   numerical sandbox to test recovery-map proposals on realistic
   noise. The CPTP recovery map is exposed as a single function
   (`cqec_recovery`) that can be replaced by a user-supplied
   alternative without changing the benchmark driver.

2. Researchers in molecular and organic-radical quantum information
   who want a ready-to-run noise-channel benchmark with parameter
   profiles tied to real material classes
   [@HoreMouritsen2016; @Schaefter2023PTM; @Boehme2009; @MannBayliss2026],
   without having to assemble the simulation infrastructure from
   scratch.

The package is small (less than 1\,000 lines of pure Python in
`src/`), has only `numpy` and `scipy` as runtime dependencies, and
runs every smoke test in well under a second on a laptop.

# State of the field

Existing general-purpose density-matrix simulators such as `QuTiP`
[@Johansson2013] and `Qiskit Aer` provide the underlying linear
algebra and Lindblad solvers needed to simulate noisy quantum
channels, but they do not ship Petz-style recovery primitives,
organic-host noise-parameter profiles, or a publication-grade
$\gamma$-sweep / peak-scaling benchmark driver.
Symbolic packages such as `SymPy` can in principle derive the
recovery-map formulae case by case but do not provide a numerical
benchmark harness. Research codes accompanying individual
recovery-map papers tend to be one-off scripts without continuous
integration, type hints, or a programmatic API.
`organic-qc-bench` is, to our knowledge, the first standalone Python
package that exposes a CPTP Petz-style recovery map together with a
reproducible $\gamma$-sweep driver, physically motivated noise
profiles, and a continuous-integration matrix, in a form that can be
extended without modifying the upstream package.

# Functionality

The public API is exported from the top-level `organic_qc_bench`
namespace. The minimal usage pattern, which reproduces the
fidelity-gain peak on the Bell state, is

```python
import numpy as np
from organic_qc_bench import (
    organic_noise, fidelity,
    recursive_covariant, cqec_recovery, make_bell,
)

np.random.seed(42)
rho_t, d = make_bell()
for gamma in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]:
    rho_n = organic_noise(rho_t, gamma=gamma, delta=0.1)
    rho_cat, _, _ = recursive_covariant(rho_n.copy(), d, 2)
    rho_c = cqec_recovery(rho_t, rho_n, rho_cat)
    print(gamma, fidelity(rho_t, rho_c) - fidelity(rho_t, rho_n))
```

The same content is accessible from the shell via

```bash
pip install organic-qc-bench
organic-qc-bench peak --dims 2 4 8 16 32 64 -t 10 -o peak.json
```

The `peak` subcommand sweeps $\gamma\in[0,1]$ at the requested
dimensions, locates the per-dimension peak, fits a linear
$\log_2 d$ scaling to the peak magnitude, and writes a JSON file
that downstream plotting code consumes directly.

# CPTP construction of the recovery map

The implemented recovery map $\mathcal R$ is built in two stages.
First, a Petz-style off-diagonal update,
$$
\tilde{\rho}_{ij}=\rho_{\mathrm{alg},ij}
 + \eta_{ij}\,
   |\rho_{\mathrm{target},ij}|\,
   e^{i\arg(\rho_{\mathrm{target},ij})},
\qquad
\eta_{ij}=1-\exp\!\bigl[-|\rho_{\mathrm{cat},ij}|\,d\,
   \mathrm{Tr}(\rho_{\mathrm{cat}}^2)\bigr],
$$
is applied. Second, the Hermitian part of the result is
eigendecomposed; non-positive eigenvalues are clipped to zero and the
trace is renormalized. The composition is trace-preserving and
completely positive by construction. The off-diagonal update can be
derived as a leading-order approximation of the Petz map
[@Petz1986] when the reference state is replaced by a noisy
catalyst, but the package does not claim this expression to be the
exact Petz recovery; the positive-projection wrapper makes the
output CPTP regardless of the closeness of the approximation.

# Quality assurance

The package ships with:

* eleven `pytest` smoke tests covering imports, profile sanity
  (`gamma >= 0`, `0 <= delta <= 1`), trace preservation of the
  organic noise channel, the CPTP wrapper acting non-decreasingly on
  Bell-state fidelity, state factories returning pure states, the
  Bernstein--Vazirani classical baseline equalling $2^{-n}$, the
  peak-scaling driver returning floats, the photoswitch simulator
  returning fidelities in $[0,1]$, the SVILC phase-frustration
  functional returning a float, and the CLI `info` subcommand;
* a GitHub Actions matrix that runs the test suite on
  `ubuntu-latest` and `macos-latest` across Python 3.9, 3.10, 3.11
  and 3.12, then builds an sdist and a wheel via `python -m build`;
* an MIT licence and a deterministic `random_state=42` policy
  throughout, so that two consecutive runs of any driver agree
  bit-for-bit;
* type-hint annotations (`py.typed`) and a `pyproject.toml` that
  exposes the package metadata through PEP 621.

# Acknowledgements

We thank colleagues at QIRI for valuable discussions. This work was
supported entirely by QIRI (Quantum Integrated Research Institute
Inc.).

# AI usage disclosure

Portions of this software's documentation, the structure of this
manuscript, and a subset of the test cases were drafted with the
assistance of a large language model. All code, all numerical
results, all mathematical derivations, and the final wording of this
paper were reviewed, verified, and approved by the human authors,
who take full responsibility for the contents.

# References
