"""
===================================================================================================
S6* FAMILY REGISTRY & UNIFIED METADATA (AUDITED & CERTIFIED)
===================================================================================================
Provides verified architectural metadata and honest cost metrics for all 8 members of the S6*
framework across the three physical branches (pruned 2026-10-09: Real-Pure-14/Yoshida-6
removed from the registry — it is a classical baseline, not a family member; pruned 2026-11-07:
Complex-Bullet-14/BCChM-6 and Complex-Instant-10/Castella-Blanes removed from the registry —
they are classical baselines, not family members; retained as external baselines with explicit
attribution, see coefficients.py):

Branch I: Real Family (Hamiltonian, Unitary Waves)
- S6_REAL_STANDARD_33
- S6_REAL_MANIFOLD_33
- S6_REAL_STENCIL_12

Branch II: Complex Family (Parabolic, Dissipative Systems)
- S6_COMPLEX_STANDARD_32
- S6_COMPLEX_FLASH_22
- S6_COMPLEX_STENCIL_12

Branch III: Simplex Family (2D Simplicial Meshes)
- S6_SIMPLEX
===================================================================================================
"""

from src.coefficients import (
    REAL_STANDARD_33_A, REAL_STANDARD_33_B, REAL_STANDARD_33_C,
    REAL_MANIFOLD_33_A, REAL_MANIFOLD_33_B, REAL_MANIFOLD_33_C,
    REAL_STENCIL_12_A, REAL_STENCIL_12_B, REAL_STENCIL_12_C,
    COMPLEX_STANDARD_32_A, COMPLEX_STANDARD_32_B, COMPLEX_STANDARD_32_C,
    COMPLEX_FLASH_22_A, COMPLEX_FLASH_22_B, COMPLEX_FLASH_22_C,
    COMPLEX_STENCIL_12_A, COMPLEX_STENCIL_12_B, COMPLEX_STENCIL_12_C,
    SIMPLEX_A_C, SIMPLEX_B_C, SIMPLEX_A_R, SIMPLEX_B_R
)

S6_FAMILY_MEMBERS = {
    # Branch I: Real
    "S6* Real-Standard-33": {
        "id": "real_standard_33",
        "branch": "Real",
        "physics": "Hamiltonian / Unitary Waves",
        "order": 6,
        "cost_per_step": 33,
        "cost_unit": "FFT transforms",
        "cost_pairs": 16.5,
        "commutator_type": "Full Spectral (21 transforms)",
        "gradient_free": False,
        "provenance": "Canonical 13-stage free-Lie splitting (a1 = -0.50)",
        "coefficients": (REAL_STANDARD_33_A, REAL_STANDARD_33_B, REAL_STANDARD_33_C)
    },
    "S6* Real-Manifold-33": {
        "id": "real_manifold_33",
        "branch": "Real",
        "physics": "Hamiltonian / Unitary Waves",
        "order": 6,
        "cost_per_step": 33,
        "cost_unit": "FFT transforms",
        "cost_pairs": 16.5,
        "commutator_type": "Full Spectral (21 transforms)",
        "gradient_free": False,
        "provenance": "Manifold re-optimization 2026-10-09: minimizer of J=max|cumsum| on the certified branch (a1=-0.108); worst cumsum -0.3281 vs -0.7378 (2.25x). Validated 2026-11-12 on the gate toy (double well + rotational carrier): observed order 5.96/5.99, reversibility 2.33e-15 at T=6 h=0.05, errors ~2.5x smaller than Standard-33 at the same h",
        "coefficients": (REAL_MANIFOLD_33_A, REAL_MANIFOLD_33_B, REAL_MANIFOLD_33_C)
    },
    "S6* Real-Stencil-12": {
        "id": "real_stencil_12",
        "branch": "Real",
        "physics": "Hamiltonian / Unitary Waves",
        "order": 6,
        "cost_per_step": 12,
        "cost_unit": "FFT transforms",
        "cost_pairs": 6.0,
        "commutator_type": "8th-Order Compact Spatial Stencil (0 FFTs)",
        "gradient_free": False,
        "provenance": "Coordinate-space finite difference stencil for double commutators",
        "coefficients": (REAL_STENCIL_12_A, REAL_STENCIL_12_B, REAL_STENCIL_12_C)
    },

    # Branch II: Complex
    "S6* Complex-Standard-32": {
        "id": "complex_standard_32",
        "branch": "Complex",
        "physics": "Parabolic / Dissipative Diffusion",
        "order": 6,
        "cost_per_step": 32,
        "cost_unit": "FFT transforms",
        "cost_pairs": 16.0,
        "commutator_type": "2-Point Gauss-Legendre Subflow",
        "gradient_free": False,
        "provenance": "Forward-positive Root 11 free-Lie splitting",
        "coefficients": (COMPLEX_STANDARD_32_A, COMPLEX_STANDARD_32_B, COMPLEX_STANDARD_32_C)
    },
    "S6* Complex-Flash-22": {
        "id": "complex_flash_22",
        "branch": "Complex",
        "physics": "Parabolic / Dissipative Diffusion",
        "order": 6,
        "cost_per_step": 22,
        "cost_unit": "FFT transforms",
        "cost_pairs": 11.0,
        "commutator_type": "Midpoint Subflow Spectral (10 transforms)",
        "gradient_free": False,
        "provenance": "Midpoint Bernoulli subflow with c1=c7=0",
        "coefficients": (COMPLEX_FLASH_22_A, COMPLEX_FLASH_22_B, COMPLEX_FLASH_22_C)
    },
    "S6* Complex-Stencil-12": {
        "id": "complex_stencil_12",
        "branch": "Complex",
        "physics": "Parabolic / Dissipative Diffusion",
        "order": 6,
        "cost_per_step": 12,
        "cost_unit": "FFT transforms",
        "cost_pairs": 6.0,
        "commutator_type": "8th-Order Compact Spatial Stencil (0 FFTs)",
        "gradient_free": False,
        "provenance": "Forward-positive Root 11 with 8th-order coordinate stencils",
        "coefficients": (COMPLEX_STENCIL_12_A, COMPLEX_STENCIL_12_B, COMPLEX_STENCIL_12_C)
    },

    # Branch III: Simplex
    "S6* Simplex-Commutator": {
        "id": "simplex_commutator",
        "branch": "Simplex",
        "physics": "2D Hamiltonian waves on triangular simplicial meshes",
        "order": 6,
        "cost_per_step": "96 K-SpMVs + 96 C-SpMVs asymptotically at m=16 with B-boundary fusion",
        "cost_unit": "sparse matvecs per macro-step",
        "commutator_type": "Exact discrete C = [V,[M^-1 K,V]] with Krylov B-stage exponential",
        "gradient_free": True,
        "provenance": "Genuine S6* canonical commutator-corrected splitting on simplicial FE meshes; V16 unified field/evolution engine",
        "architecture": "field-agnostic SimplexSolver with separate evolution policy",
        "coefficients": (SIMPLEX_A_R, SIMPLEX_B_R)
    },
    "Simplex-Yoshida-6": {
        "id": "simplex_yoshida_6",
        "branch": "Simplex",
        "physics": "2D Hamiltonian waves on triangular simplicial meshes",
        "order": 6,
        "cost_per_step": "112 K-SpMVs asymptotically with kinetic FSAL at m=16",
        "cost_unit": "sparse matvecs per macro-step",
        "commutator_type": "None (pure Yoshida T-V-T composition)",
        "gradient_free": True,
        "provenance": "Classical Yoshida-6 composition realized on simplicial FE geometry",
        "coefficients": (SIMPLEX_A_R, SIMPLEX_B_R)
    }
}

def list_family_members():
    return list(S6_FAMILY_MEMBERS.keys())
