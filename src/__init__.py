"""
===================================================================================================
S6* GEOMETRIC SPLITTING INTEGRATION FRAMEWORK (PYTHON API)
===================================================================================================
Comprehensive library supporting the active S6* family members and retained classical baselines of the S6* splitting framework:

Branch I: Real Geometric Family (Hamiltonian, Unitary Waves)
- S6* Real-Standard-33 (33 FFT transforms/step)
- S6* Real-Stencil-12  (12 FFT transforms/step)
(Yoshida-6 baseline: YOSHIDA_6_* coefficients + RealGPESolver.step_yoshida6 — not a family member)

Branch II: Complex Geometric Family (Parabolic, Dissipative Systems)
- S6* Complex-Standard-32 (32 FFT transforms/step)
- S6* Complex-Flash-22    (22 FFT transforms/step)
- S6* Complex-Stencil-12  (12 FFT transforms/step)
(BCChM-6 P6S7 baseline: BCCHM_6_* coefficients + ComplexParabolicSolver.step_bcchm_6 — not a family member;
Castella-Blanes baseline: CASTELLA_BLANES_* coefficients + ComplexParabolicSolver.step_castella_blanes — not a family member)

Branch III: Simplex Geometry-Native Family (2D Simplicial Meshes over K in {R, C})
- S6* Simplex (7 * m_krylov SpMVs/step with FSAL)
===================================================================================================
"""

from src.coefficients import (
    REAL_STANDARD_33_A, REAL_STANDARD_33_B, REAL_STANDARD_33_C, REAL_STANDARD_33_FFTS,
    YOSHIDA_6_A, YOSHIDA_6_B, YOSHIDA_6_FFTS,
    REAL_STENCIL_12_A, REAL_STENCIL_12_B, REAL_STENCIL_12_C, REAL_STENCIL_12_FFTS,
    STENCIL_D, STENCIL_D2, STENCIL_D2_CENTER,
    COMPLEX_STANDARD_32_A, COMPLEX_STANDARD_32_B, COMPLEX_STANDARD_32_C, COMPLEX_STANDARD_32_FFTS,
    COMPLEX_FLASH_22_A, COMPLEX_FLASH_22_B, COMPLEX_FLASH_22_C, COMPLEX_FLASH_22_FFTS,
    BCCHM_6_A, BCCHM_6_B, BCCHM_6_FFTS,
    COMPLEX_STENCIL_12_A, COMPLEX_STENCIL_12_B, COMPLEX_STENCIL_12_C, COMPLEX_STENCIL_12_FFTS,
    CASTELLA_BLANES_A, CASTELLA_BLANES_B, CASTELLA_BLANES_FFTS,
    SIMPLEX_A_C, SIMPLEX_B_C, SIMPLEX_A_R, SIMPLEX_B_R, SIMPLEX_KINETIC_STAGES
)

from src.solvers_spectral import RealGPESolver, ComplexParabolicSolver
from src.simplex_mesh import SimplexMesh2D
from src.simplex_solver import SimplexSolver
from src.simplex_policy import SimplexEvolutionPolicy, make_policy
from src.family import S6_FAMILY_MEMBERS, list_family_members

__all__ = [
    # Solvers
    "RealGPESolver",
    "ComplexParabolicSolver",
    "SimplexMesh2D",
    "SimplexSolver",
    "SimplexEvolutionPolicy",
    "make_policy",
    "S6_FAMILY_MEMBERS",
    "list_family_members",
    # Real Branch Coefficients
    "REAL_STANDARD_33_A", "REAL_STANDARD_33_B", "REAL_STANDARD_33_C", "REAL_STANDARD_33_FFTS",
    "YOSHIDA_6_A", "YOSHIDA_6_B", "YOSHIDA_6_FFTS",
    "REAL_STENCIL_12_A", "REAL_STENCIL_12_B", "REAL_STENCIL_12_C", "REAL_STENCIL_12_FFTS",
    "STENCIL_D", "STENCIL_D2", "STENCIL_D2_CENTER",
    # Complex Branch Coefficients
    "COMPLEX_STANDARD_32_A", "COMPLEX_STANDARD_32_B", "COMPLEX_STANDARD_32_C", "COMPLEX_STANDARD_32_FFTS",
    "COMPLEX_FLASH_22_A", "COMPLEX_FLASH_22_B", "COMPLEX_FLASH_22_C", "COMPLEX_FLASH_22_FFTS",
    "BCCHM_6_A", "BCCHM_6_B", "BCCHM_6_FFTS",
    "COMPLEX_STENCIL_12_A", "COMPLEX_STENCIL_12_B", "COMPLEX_STENCIL_12_C", "COMPLEX_STENCIL_12_FFTS",
    "CASTELLA_BLANES_A", "CASTELLA_BLANES_B", "CASTELLA_BLANES_FFTS",
    # Simplex Branch Coefficients
    "SIMPLEX_A_C", "SIMPLEX_B_C", "SIMPLEX_A_R", "SIMPLEX_B_R", "SIMPLEX_KINETIC_STAGES"
]
