"""Public S6*-Adjoint Simplex module."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .autograd import S6SimplexFunction
from .backend import SimplexTorchBackend


class NodalPotential(nn.Module):
    """Trainable nodal potential for cases where V is supplied directly."""

    def __init__(self, values, dtype=torch.float64):
        super().__init__()
        values = torch.as_tensor(values, dtype=dtype).reshape(-1)
        self.values = nn.Parameter(values.clone())

    def forward(self, coords):
        if self.values.numel() != coords.shape[0]:
            raise ValueError("NodalPotential size does not match mesh node count")
        return self.values


class S6SimplexAdjoint(nn.Module):
    """Differentiable S6* Simplex-Commutator propagator with O(1) step memory.

    Unlike the Hamiltonian q/p adjoint, Simplex evolves a complex wavefunction
    on a 2D finite-element mesh.  The potential module is evaluated once from
    mesh coordinates and differentiated stage-locally during the reverse pass.
    """

    def __init__(
        self,
        mesh,
        potential: nn.Module,
        m_krylov: int = 16,
        tol: float = 1e-13,
        lanczos_check_every: int = 4,
        compile_kernels: bool = False,
    ):
        super().__init__()
        if not isinstance(potential, nn.Module):
            raise TypeError("potential must be a torch.nn.Module")
        self.backend = SimplexTorchBackend(mesh, m_krylov=m_krylov, tol=tol,
                                          lanczos_check_every=lanczos_check_every,
                                          compile_kernels=compile_kernels)
        self.potential = potential

    @property
    def mesh_size(self) -> int:
        return self.backend.N

    def forward(self, psi0: torch.Tensor, h: float, num_steps: int) -> torch.Tensor:
        if psi0.ndim != 1:
            psi0 = psi0.reshape(-1)
        if psi0.numel() != self.backend.N:
            raise ValueError(f"psi0 must have {self.backend.N} nodal values; got {psi0.numel()}")
        param_names = tuple(name for name, _ in self.potential.named_parameters())
        params = tuple(self.potential.parameters())
        return S6SimplexFunction.apply(
            psi0, h, num_steps, self.potential, param_names, self.backend, *params
        )
