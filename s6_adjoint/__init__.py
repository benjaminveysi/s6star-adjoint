"""
S6*-Adjoint: Sixth-Order Commutator-Corrected Geometric Splitting & Adjoint Engine.
Release v16 Unified Architecture:
- Branch I: Real Family (Hamiltonian Waves & Exact Reversible Adjoints)
- Branch II: Complex Family (Forward-Positive Parabolic Diffusion)
- Branch III: Simplex Family (2D Simplicial Finite-Element Meshes)
"""
__version__ = "1.1.0"

from .core.coefficients import (
    S6_REAL_A, S6_REAL_B, S6_REAL_C,
    REAL_STANDARD_33_A, REAL_STANDARD_33_B, REAL_STANDARD_33_C,
    REAL_MANIFOLD_33_A, REAL_MANIFOLD_33_B, REAL_MANIFOLD_33_C,
    REAL_STENCIL_12_A, REAL_STENCIL_12_B, REAL_STENCIL_12_C,
    S6_COMPLEX_A, S6_COMPLEX_B, S6_COMPLEX_C,
    COMPLEX_STANDARD_32_A, COMPLEX_STANDARD_32_B, COMPLEX_STANDARD_32_C,
    COMPLEX_FLASH_22_A, COMPLEX_FLASH_22_B, COMPLEX_FLASH_22_C,
    COMPLEX_STENCIL_12_A, COMPLEX_STENCIL_12_B, COMPLEX_STENCIL_12_C,
    S6_SIMPLEX_COMM_A, S6_SIMPLEX_COMM_B, S6_SIMPLEX_COMM_C,
    S6_A, S6_B, S6_C,
    BCCHM_GAMMAS, YOSHIDA4_W, YOSHIDA6_W,
    COEFFICIENTS_REGISTRY
)
from .core.integrator import S6HamiltonianCore
from .core.dynamics import HamiltonianDynamics

__all__ = [
    "S6_REAL_A", "S6_REAL_B", "S6_REAL_C",
    "REAL_STANDARD_33_A", "REAL_STANDARD_33_B", "REAL_STANDARD_33_C",
    "REAL_MANIFOLD_33_A", "REAL_MANIFOLD_33_B", "REAL_MANIFOLD_33_C",
    "REAL_STENCIL_12_A", "REAL_STENCIL_12_B", "REAL_STENCIL_12_C",
    "S6_COMPLEX_A", "S6_COMPLEX_B", "S6_COMPLEX_C",
    "COMPLEX_STANDARD_32_A", "COMPLEX_STANDARD_32_B", "COMPLEX_STANDARD_32_C",
    "COMPLEX_FLASH_22_A", "COMPLEX_FLASH_22_B", "COMPLEX_FLASH_22_C",
    "COMPLEX_STENCIL_12_A", "COMPLEX_STENCIL_12_B", "COMPLEX_STENCIL_12_C",
    "S6_SIMPLEX_COMM_A", "S6_SIMPLEX_COMM_B", "S6_SIMPLEX_COMM_C",
    "S6_A", "S6_B", "S6_C",
    "BCCHM_GAMMAS", "YOSHIDA4_W", "YOSHIDA6_W",
    "COEFFICIENTS_REGISTRY",
    "S6HamiltonianCore",
    "HamiltonianDynamics",
    "__version__"
]

from .simplex import (
    SimplexTorchBackend, NodalPotential, S6SimplexAdjoint, S6SimplexFunction,
    propagate_unrolled, simplex_hvp,
)
from .selector import S6Adjoint

from src.simplex_mesh import SimplexMesh2D
from src.simplex_solver import SimplexSolver
from src.simplex_policy import SimplexEvolutionPolicy, make_policy

__all__.extend([
    "SimplexMesh2D", "SimplexSolver", "SimplexEvolutionPolicy", "make_policy",
    "SimplexTorchBackend", "NodalPotential", "S6SimplexAdjoint",
    "S6SimplexFunction", "propagate_unrolled", "simplex_hvp", "S6Adjoint"
])

try:
    from .autograd.function import S6HamiltonianFunction
    from .autograd.module import S6HamiltonianODE
    from .general_ode import S6GeneralVectorFieldODE
    __all__.extend(["S6HamiltonianFunction", "S6HamiltonianODE", "S6GeneralVectorFieldODE"])
except (ImportError, AttributeError):
    pass

try:
    from .diff_md import (
        LennardJonesPotential,
        NeuralInteratomicPotential,
        DifferentiableMDSimulator,
        compute_temperature,
        compute_kinetic_energy,
        compute_total_energy,
        compute_radial_distribution_function,
        compute_mean_squared_displacement
    )
    __all__.extend([
        "LennardJonesPotential",
        "NeuralInteratomicPotential",
        "DifferentiableMDSimulator",
        "compute_temperature",
        "compute_kinetic_energy",
        "compute_total_energy",
        "compute_radial_distribution_function",
        "compute_mean_squared_displacement"
    ])
except (ImportError, AttributeError):
    pass

try:
    from .models.neural_forcefield import SchNetForceField
    from .diff_md.xyz_io import write_xyz_trajectory
    __all__.extend(["SchNetForceField", "write_xyz_trajectory"])
except (ImportError, AttributeError):
    pass
