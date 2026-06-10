"""Command-line interface: ``organic-qc-bench {bv,peak,photoswitch,svilc,info}``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

import numpy as np

from . import __version__
from .bv import benchmark as bv_benchmark, classical_one_query_rate
from .peak import scaling as peak_scaling
from .photoswitch import PhotoSwitchParams, sweep_gate_time
from .profiles import PROFILES, get_profile
from .svilc import (
    TriangularLattice,
    spin_vortex_field,
    phase_frustration_coupling,
    external_current_phase,
)


def _emit(payload: dict, out: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, default=lambda o:
                      o.tolist() if hasattr(o, "tolist") else str(o))
    if out is None or out == "-":
        print(text)
    else:
        with open(out, "w") as f:
            f.write(text)
        print(f"Saved -> {out}")


def _cmd_info(_args) -> int:
    print(f"organic-qc-bench {__version__}")
    print("\nProfiles:")
    for name, p in PROFILES.items():
        print(f"  {name:<14}  gamma={p.gamma:<8g} delta={p.delta:<8g} "
              f"T2={p.T2_us}us  T_op={p.T_op_K}K  -- {p.material}")
    return 0


def _cmd_bv(args) -> int:
    profiles = {n: PROFILES[n] for n in args.profile}
    res = bv_benchmark(args.n_qubits, profiles,
                       n_trials=args.trials, seed=args.seed)
    payload = {
        "n_qubits": args.n_qubits,
        "n_trials": args.trials,
        "classical_one_query": classical_one_query_rate(args.n_qubits),
        "quantum": res,
    }
    _emit(payload, args.output)
    return 0


def _cmd_peak(args) -> int:
    gammas = np.logspace(args.gamma_log_min, args.gamma_log_max,
                         args.gamma_points)
    summary = peak_scaling(dims=tuple(args.dims), gammas=gammas,
                           n_trials=args.trials)
    _emit({"dims": list(args.dims), "summary": summary,
           "gamma_grid": gammas.tolist()}, args.output)
    return 0


def _cmd_photoswitch(args) -> int:
    prof = get_profile(args.profile)
    res = sweep_gate_time(prof,
                          t_grid_ns=list(np.linspace(args.t_min,
                                                     args.t_max,
                                                     args.points)))
    _emit({"profile": prof.name, **res}, args.output)
    return 0


def _cmd_svilc(args) -> int:
    lat = TriangularLattice(Lx=args.Lx, Ly=args.Ly,
                            t=1.0, tp=args.tp, pbc=True)
    # default SVQ centres around (Lx/4, Ly/2) and (3Lx/4, Ly/2)
    cx_a, cy_a = lat.Lx // 4, lat.Ly // 2
    cx_b, cy_b = (lat.Lx // 4) + args.distance, lat.Ly // 2
    cents_a = [(cx_a - 1, cy_a - 1), (cx_a + 1, cy_a - 1),
               (cx_a + 1, cy_a + 1), (cx_a - 1, cy_a + 1)]
    cents_b = [(cx_b - 1, cy_b - 1), (cx_b + 1, cy_b - 1),
               (cx_b + 1, cy_b + 1), (cx_b - 1, cy_b + 1)]
    winds_a = [+1, -1, +1, -1]
    winds_b = [+1, -1, +1, -1]
    chi_a = spin_vortex_field(lat, cents_a, winds_a)
    chi_b = spin_vortex_field(lat, cents_b, winds_b)
    V_base = phase_frustration_coupling(lat, chi_a, chi_b)
    rows = [{"J_ext": 0.0, "V": V_base}]
    if args.feed_amps:
        for amp in args.feed_amps:
            A = external_current_phase(
                lat,
                lat.idx((cx_a + cx_b) // 2, 0),
                lat.idx((cx_a + cx_b) // 2, lat.Ly - 1),
                amp,
            )
            V = phase_frustration_coupling(lat, chi_a, chi_b, A_extra=A)
            rows.append({"J_ext": float(amp), "V": V})
    _emit({"Lx": lat.Lx, "Ly": lat.Ly, "tp": lat.tp,
           "distance": args.distance, "results": rows}, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="organic-qc-bench",
        description="Benchmark suite for engineered organic quantum platforms.")
    p.add_argument("--version", action="version",
                   version=f"organic-qc-bench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show package + profile information"
                   ).set_defaults(func=_cmd_info)

    p_bv = sub.add_parser("bv", help="Bernstein-Vazirani benchmark")
    p_bv.add_argument("-n", "--n_qubits", type=int, default=5)
    p_bv.add_argument("-t", "--trials", type=int, default=100)
    p_bv.add_argument("--profile", nargs="+",
                      default=["Path1_RPRes", "Path2_PTM",
                               "Path3_OrgSC", "Path4_SSH"])
    p_bv.add_argument("--seed", type=int, default=42)
    p_bv.add_argument("-o", "--output", default="-")
    p_bv.set_defaults(func=_cmd_bv)

    p_peak = sub.add_parser("peak", help="CQEC fidelity-gain peak vs d")
    p_peak.add_argument("--dims", nargs="+", type=int,
                        default=[2, 4, 8, 16])
    p_peak.add_argument("--gamma-log-min", type=float, default=-2.0)
    p_peak.add_argument("--gamma-log-max", type=float, default=0.7)
    p_peak.add_argument("--gamma-points", type=int, default=24)
    p_peak.add_argument("-t", "--trials", type=int, default=10)
    p_peak.add_argument("-o", "--output", default="-")
    p_peak.set_defaults(func=_cmd_peak)

    p_ps = sub.add_parser("photoswitch", help="diarylethene CZ-gate scan")
    p_ps.add_argument("--profile", default="Path2_PTM")
    p_ps.add_argument("--t-min", type=float, default=0.5)
    p_ps.add_argument("--t-max", type=float, default=20.0)
    p_ps.add_argument("--points", type=int, default=12)
    p_ps.add_argument("-o", "--output", default="-")
    p_ps.set_defaults(func=_cmd_photoswitch)

    p_sv = sub.add_parser("svilc", help="κ-(BEDT-TTF) SVILC lattice analysis")
    p_sv.add_argument("--Lx", type=int, default=16)
    p_sv.add_argument("--Ly", type=int, default=6)
    p_sv.add_argument("--tp", type=float, default=0.8)
    p_sv.add_argument("--distance", type=int, default=10)
    p_sv.add_argument("--feed-amps", nargs="*", type=float,
                      default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
    p_sv.add_argument("-o", "--output", default="-")
    p_sv.set_defaults(func=_cmd_svilc)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
