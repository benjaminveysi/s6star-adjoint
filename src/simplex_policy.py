"""Field-independent evolution policies for the S6* Simplex engine.

The Simplex spatial discretization and stage engine are independent of the scalar field.
Only the evolution policy selects the coefficient family and generator appropriate to the
physical problem.  This keeps R/C as a representation choice rather than an alternate solver.
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
from src.coefficients_simplex import (
    S6_SIMPLEX_COMM_A, S6_SIMPLEX_COMM_B, S6_SIMPLEX_COMM_C,
    S6_SIMPLEX_PARAB_A, S6_SIMPLEX_PARAB_B, S6_SIMPLEX_PARAB_C,
    YOSHIDA_6_A, YOSHIDA_6_B, BCCHM_6_A, BCCHM_6_B,
)

EvolutionName = Literal["hamiltonian", "parabolic", "s6star", "legacy", "custom"]

@dataclass(frozen=True)
class SimplexEvolutionPolicy:
    name: str
    generator: complex
    stages_a: np.ndarray
    stages_b: np.ndarray
    stages_c: np.ndarray
    uses_commutator: bool
    requires_complex_field: bool = False
    internal_complex: bool = False
    description: str = ""
    unsafe_real_projection: bool = False

    @property
    def kinetic_stage_count(self) -> int:
        return len(self.stages_a)

    @property
    def potential_stage_count(self) -> int:
        return len(self.stages_b)


def make_policy(evolution: str, field: str, scheme: str | None = None) -> SimplexEvolutionPolicy:
    evolution = evolution.lower()
    field = field.lower()
    scheme = None if scheme is None else scheme.lower()

    if evolution in {"s6star", "simplex", "generic"}:
        # One algebraic S6* commutator formulation over either scalar field.
        # For C it is unitary Hamiltonian propagation; for R it is the real-field
        # exponential specialization. The latter is a formal real specialization
        # and is not marketed as a forward-positive diffusion method because the
        # real S6 coefficients contain negative substeps.
        is_complex = field in {"complex", "c"}
        return SimplexEvolutionPolicy(
            name="s6star-simplex-unified",
            generator=-1j if is_complex else -1.0,
            stages_a=np.asarray(S6_SIMPLEX_COMM_A, dtype=float),
            stages_b=np.asarray(S6_SIMPLEX_COMM_B, dtype=float),
            stages_c=np.asarray(S6_SIMPLEX_COMM_C, dtype=float),
            uses_commutator=True,
            requires_complex_field=False,
            internal_complex=is_complex,
            description=(
                "Single field-agnostic S6* commutator formulation. "
                "The same geometric/operator/stage algebra is specialized to R or C by the evolution generator."
            ),
        )

    if evolution in {"hamiltonian", "quantum", "unitary"}:
        if field not in {"complex", "c"}:
            raise ValueError(
                "Hamiltonian evolution uses exp(-i h H) and therefore requires a complex field. "
                "Use field='complex' or choose a parabolic evolution for a real field."
            )
        return SimplexEvolutionPolicy(
            name="hamiltonian-s6-commutator",
            generator=-1j,
            stages_a=np.asarray(S6_SIMPLEX_COMM_A, dtype=float),
            stages_b=np.asarray(S6_SIMPLEX_COMM_B, dtype=float),
            stages_c=np.asarray(S6_SIMPLEX_COMM_C, dtype=float),
            uses_commutator=True,
            requires_complex_field=True,
            internal_complex=True,
            description="Canonical 13-stage S6* commutator-corrected Hamiltonian evolution.",
        )

    if evolution in {"parabolic", "diffusion", "imaginary_time"}:
        # Forward-positive complex coefficients are the stable high-order parabolic policy.
        # They naturally operate in C even when a physical solution is expected to be real.
        # The common solver therefore keeps a complex internal representation and only
        # projects at the macro-step boundary when the caller explicitly requests field='real'.
        if scheme in {"bcchm", None}:
            return SimplexEvolutionPolicy(
                name="parabolic-bcchm6",
                generator=-1.0,
                stages_a=np.asarray(BCCHM_6_A, dtype=complex),
                stages_b=np.asarray(BCCHM_6_B, dtype=complex),
                stages_c=np.zeros(len(BCCHM_6_B), dtype=complex),
                uses_commutator=False,
                requires_complex_field=False,
                internal_complex=True,
                description="Forward-positive BCChM-6 parabolic composition on the common Simplex engine.",
            )
        if field == "real" and scheme in {"real_s6", "s6_real", "s6_commutator_real"}:
            return SimplexEvolutionPolicy(
                name="parabolic-real-s6-commutator",
                generator=-1.0,
                stages_a=np.asarray(S6_SIMPLEX_COMM_A, dtype=float),
                stages_b=np.asarray(S6_SIMPLEX_COMM_B, dtype=float),
                stages_c=np.asarray(S6_SIMPLEX_COMM_C, dtype=float),
                uses_commutator=True,
                requires_complex_field=False,
                internal_complex=False,
                description=(
                    "Field-real specialization of the canonical S6* commutator engine. "
                    "Formally sixth-order; negative diffusion substeps mean it is not a forward-positive parabolic scheme."
                ),
            )
        if scheme in {"s6_parabolic", "s6_commutator_parabolic"}:
            return SimplexEvolutionPolicy(
                name="parabolic-s6-commutator",
                generator=-1.0,
                stages_a=np.asarray(S6_SIMPLEX_PARAB_A, dtype=complex),
                stages_b=np.asarray(S6_SIMPLEX_PARAB_B, dtype=complex),
                stages_c=np.asarray(S6_SIMPLEX_PARAB_C, dtype=complex),
                uses_commutator=True,
                requires_complex_field=False,
                internal_complex=True,
                description="Experimental forward-positive complex-coefficient S6* commutator policy.",
            )
        raise ValueError(f"Unknown parabolic scheme: {scheme}")

    if evolution == "legacy":
        if scheme in {"s6_commutator", "commutator", "s6"} and field == "real":
            return SimplexEvolutionPolicy(
                name="legacy-real-s6-parabolic",
                generator=-1.0,
                stages_a=np.asarray(S6_SIMPLEX_PARAB_A, dtype=complex),
                stages_b=np.asarray(S6_SIMPLEX_PARAB_B, dtype=complex),
                stages_c=np.asarray(S6_SIMPLEX_PARAB_C, dtype=complex),
                uses_commutator=False,
                requires_complex_field=False,
                internal_complex=True,
                unsafe_real_projection=True,
                description="Backward-compatible legacy real projection path; use explicit evolution policies for new code.",
            )
        if scheme in {"yoshida", "pure"}:
            return SimplexEvolutionPolicy(
                name="yoshida6-baseline",
                generator=-1j if field in {"complex", "c"} else -1.0,
                stages_a=np.asarray(YOSHIDA_6_A, dtype=float),
                stages_b=np.asarray(YOSHIDA_6_B, dtype=float),
                stages_c=np.zeros(len(YOSHIDA_6_B), dtype=float),
                uses_commutator=False,
                requires_complex_field=False,
                internal_complex=field in {"complex", "c"},
                description="Classical Yoshida-6 pure splitting baseline realized on Simplex geometry.",
            )
        if scheme in {"bcchm"}:
            return SimplexEvolutionPolicy(
                name="bcchm6-baseline",
                generator=-1.0,
                stages_a=np.asarray(BCCHM_6_A, dtype=complex),
                stages_b=np.asarray(BCCHM_6_B, dtype=complex),
                stages_c=np.zeros(len(BCCHM_6_B), dtype=complex),
                uses_commutator=False,
                requires_complex_field=False,
                internal_complex=True,
                description="Classical BCChM-6 P6S7 baseline realized on Simplex geometry.",
            )
        raise ValueError(f"Unknown legacy scheme: {scheme}")

    raise ValueError(f"Unknown evolution policy: {evolution}")
