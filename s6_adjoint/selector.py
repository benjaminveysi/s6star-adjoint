"""Unified user-facing selector for reversible S6* adjoint backends."""
from __future__ import annotations

import torch.nn as nn

from .autograd.module import S6HamiltonianODE
from .general_ode import S6GeneralVectorFieldODE
from .simplex.module import S6SimplexAdjoint


_REAL_ALIASES = {
    "real", "real_standard", "real_standard_33", "real_stencil", "real_stencil_12", "real_manifold", "real_manifold_33"
}
_GENERAL_ALIASES = {
    "general", "general_standard", "general_standard_33",
    "general_stencil", "general_stencil_12", "general_manifold", "general_manifold_33",
}
_SIMPLEX_ALIASES = {"simplex", "simplex_commutator", "s6_simplex_commutator", "simplex_s6"}


class S6Adjoint(nn.Module):
    """Select the real q/p adjoint, the general-vector-field adjoint, or the
    differentiable Simplex wave backend.

    Real branches use ``forward(q0, p0, h, num_steps)``.  General branches
    (Adjoint-G, arbitrary non-conservative F_theta) use the same
    ``forward(q0, p0, h, num_steps)`` with ``potential`` being the vector
    field F_theta.  Simplex uses ``forward(psi0, h, num_steps)`` and
    requires a 2D FE mesh.
    """

    def __init__(self, potential, scheme="real_standard_33", *, mesh=None, mass=1.0,
                 m_krylov=16, tol=1e-13, lanczos_check_every=4,
                 compile_kernels=False):
        super().__init__()
        s = str(scheme).lower()
        if s in _REAL_ALIASES:
            self.backend_name = "real"
            self.scheme = ("real_standard_33" if s in {"real", "real_standard"}
                         else "real_manifold_33" if s == "real_manifold" else s)
            self.engine = S6HamiltonianODE(potential, scheme=self.scheme, mass=mass)
        elif s in _GENERAL_ALIASES:
            self.backend_name = "general"
            self.scheme = ("general_standard_33" if s in
                           {"general", "general_standard", "general_standard_33"}
                           else "general_manifold_33" if s in
                           {"general_manifold", "general_manifold_33"}
                           else "general_stencil_12")
            self.engine = S6GeneralVectorFieldODE(potential, scheme=self.scheme)
        elif s in _SIMPLEX_ALIASES:
            if mesh is None:
                raise ValueError("scheme='simplex_commutator' requires mesh=SimplexMesh2D(...)")
            self.backend_name = "simplex"
            self.scheme = "simplex_commutator"
            self.engine = S6SimplexAdjoint(mesh, potential, m_krylov=m_krylov, tol=tol,
                                          lanczos_check_every=lanczos_check_every,
                                          compile_kernels=compile_kernels)
        else:
            raise ValueError(
                f"Unknown S6* adjoint scheme '{s}'. Supported: "
                "real_standard_33, real_stencil_12, real_manifold_33, "
                "general_standard_33, general_stencil_12, general_manifold_33, simplex_commutator"
            )

    def forward(self, *args, **kwargs):
        return self.engine(*args, **kwargs)
