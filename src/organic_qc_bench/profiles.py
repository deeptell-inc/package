"""Noise profiles for the four organic realisation paths."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrganicProfile:
    """A single organic-material noise profile (Path 1--4)."""

    name: str
    path_id: int
    material: str
    gamma: float
    delta: float
    T2_us: float
    gate_ns: float
    T_op_K: float
    notes: str = ""

    @property
    def below_EB(self) -> bool:
        """Whether ``gamma`` is below the entanglement-breaking threshold."""
        return self.gamma < 0.3


PROFILES = {
    "Path1_RPRes": OrganicProfile(
        name="Path1_RPRes",
        path_id=1,
        material="Engineered flavin--TEMPO radical pair (RT)",
        gamma=0.100, delta=0.080,
        T2_us=0.10, gate_ns=10.0, T_op_K=298.0,
        notes="Reservoir-regime; gamma near EB threshold.",
    ),
    "Path2_PTM": OrganicProfile(
        name="Path2_PTM",
        path_id=2,
        material="PTM radical in COF (EDSR, RT)",
        gamma=0.003, delta=0.005,
        T2_us=3.0, gate_ns=8.0, T_op_K=298.0,
        notes="Coherent QC; gamma << gamma_c.",
    ),
    "Path3_OrgSC": OrganicProfile(
        name="Path3_OrgSC",
        path_id=3,
        material="kappa-(BEDT-TTF)2Cu[N(CN)2]Br SVILC (4 K)",
        gamma=5e-5, delta=1e-4,
        T2_us=100.0, gate_ns=5.0, T_op_K=4.0,
        notes="Conditional on experimental SVILC confirmation.",
    ),
    "Path4_SSH": OrganicProfile(
        name="Path4_SSH",
        path_id=4,
        material="trans-polyacetylene SSH soliton (RT)",
        gamma=0.002, delta=0.003,
        T2_us=0.5, gate_ns=1.0, T_op_K=298.0,
        notes="Z2-protected topological soliton qubit.",
    ),
}


def get_profile(name: str) -> OrganicProfile:
    """Look up a profile by short name (P1, Path1, Path1_RPRes, etc.)."""
    if name in PROFILES:
        return PROFILES[name]
    aliases = {
        "P1": "Path1_RPRes", "Path1": "Path1_RPRes",
        "P2": "Path2_PTM",   "Path2": "Path2_PTM",
        "P3": "Path3_OrgSC", "Path3": "Path3_OrgSC",
        "P4": "Path4_SSH",   "Path4": "Path4_SSH",
    }
    if name in aliases:
        return PROFILES[aliases[name]]
    raise KeyError(f"unknown profile {name!r}; valid: {sorted(PROFILES) + sorted(aliases)}")
