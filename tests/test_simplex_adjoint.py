"""Validation for the differentiable S6* Simplex-Commutator backend."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from s6_adjoint import S6Adjoint, S6SimplexAdjoint, SimplexMesh2D, SimplexSolver


class PolynomialPotential(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.05, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))

    def forward(self, coords):
        return self.w * (coords[:, 0] ** 2 + coords[:, 1] ** 2) + self.b


def _unrolled_reference(mesh, potential, psi0, h, steps, m_krylov):
    backend = S6SimplexAdjoint(mesh, potential, m_krylov=m_krylov).backend
    coords = backend.coords
    V = potential(coords)
    A = [float(x) for x in backend.A.detach().cpu().tolist()]
    B = [float(x) for x in backend.B.detach().cpu().tolist()]
    C = [float(x) for x in backend.C.detach().cpu().tolist()]

    u = psi0
    u = backend.potential_expm(u, V, h * B[0], h**3 * C[0])
    for step in range(steps):
        for k in range(5):
            u = backend.kinetic_expm(u, h * A[k])
            u = backend.potential_expm(u, V, h * B[k + 1], h**3 * C[k + 1])
        u = backend.kinetic_expm(u, h * A[5])
        factor = 2.0 if step < steps - 1 else 1.0
        u = backend.potential_expm(u, V, h * factor * B[6], h**3 * factor * C[6])
    return u


def test_simplex_adjoint_matches_unrolled_gradient():
    mesh = SimplexMesh2D(Lx=4.0, Ly=4.0, Nx=6, Ny=6, periodic=True)
    torch.manual_seed(7)
    p_ref = PolynomialPotential()
    p_adj = PolynomialPotential()
    p_adj.load_state_dict(p_ref.state_dict())

    psi_base = (torch.randn(mesh.N_nodes) + 1j * torch.randn(mesh.N_nodes)).to(torch.complex128)
    psi_ref = psi_base.clone().requires_grad_(True)
    psi_adj = psi_base.clone().requires_grad_(True)

    h = 0.03
    steps = 3
    out_ref = _unrolled_reference(mesh, p_ref, psi_ref, h, steps, m_krylov=16)
    out_adj = S6SimplexAdjoint(mesh, p_adj, m_krylov=16)(psi_adj, h, steps)

    coords = torch.as_tensor(mesh.nodes, dtype=torch.float64)
    mass = torch.as_tensor(mesh.M_diag, dtype=torch.float64)
    weight = 1.0 + 0.2 * coords[:, 0] - 0.11 * coords[:, 1]
    loss_ref = (weight * (out_ref.real**2 + 0.37 * out_ref.imag**2) * mass).sum()
    loss_adj = (weight * (out_adj.real**2 + 0.37 * out_adj.imag**2) * mass).sum()
    loss_ref.backward()
    loss_adj.backward()

    assert torch.allclose(out_adj, out_ref, atol=2e-13, rtol=2e-13)
    for a, b in ((psi_ref.grad, psi_adj.grad), (p_ref.w.grad, p_adj.w.grad), (p_ref.b.grad, p_adj.b.grad)):
        rel = torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(a) + 1e-30)
        assert float(rel) < 2e-12


def test_simplex_adjoint_reversal_and_mass_unitarity():
    mesh = SimplexMesh2D(Lx=6.0, Ly=6.0, Nx=8, Ny=8, periodic=True)
    potential = PolynomialPotential()
    engine = S6SimplexAdjoint(mesh, potential, m_krylov=16)
    psi = (torch.randn(mesh.N_nodes) + 1j * torch.randn(mesh.N_nodes)).to(torch.complex128)

    with torch.no_grad():
        final = engine(psi, 0.03, 3)
        recovered = engine(final, -0.03, 3)

    rev_err = torch.linalg.vector_norm(recovered - psi).item()
    assert rev_err < 5e-13

    mass = torch.as_tensor(mesh.M_diag, dtype=torch.float64)
    n0 = torch.real(torch.sum(torch.conj(psi) * mass * psi)).item()
    n1 = torch.real(torch.sum(torch.conj(final) * mass * final)).item()
    assert abs(n1 - n0) < 5e-12


def test_simplex_forward_matches_numpy_solver():
    mesh = SimplexMesh2D(Lx=6.0, Ly=6.0, Nx=8, Ny=8, periodic=True)
    V = 0.05 * (mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2) + 0.02
    rng = np.random.RandomState(11)
    psi_np = rng.randn(mesh.N_nodes) + 1j * rng.randn(mesh.N_nodes)
    psi_np /= mesh.norm(psi_np)

    baseline = SimplexSolver(mesh, field_type="complex", theta=0.0, V=V, scheme="s6_commutator")
    ref = baseline.integrate_simplex_s6(psi_np, n_steps=3, h=0.03, m_krylov=16, use_fsal=True)

    class FixedPotential(nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = nn.Parameter(torch.as_tensor(values, dtype=torch.float64), requires_grad=False)

        def forward(self, coords):
            return self.values

    engine = S6SimplexAdjoint(mesh, FixedPotential(V), m_krylov=16)
    psi = torch.as_tensor(psi_np, dtype=torch.complex128)
    with torch.no_grad():
        got = engine(psi, 0.03, 3)
    assert torch.linalg.vector_norm(got - torch.as_tensor(ref, dtype=torch.complex128)).item() < 1e-11


def test_unified_selector_exposes_three_user_selectable_backends():
    mesh = SimplexMesh2D(Lx=3.0, Ly=3.0, Nx=5, Ny=5, periodic=True)

    class RealPotential(nn.Module):
        def forward(self, q):
            return 0.5 * torch.sum(q * q, dim=-1)

    real_std = S6Adjoint(RealPotential(), scheme="real_standard_33")
    real_stencil = S6Adjoint(RealPotential(), scheme="real_stencil_12")
    simplex = S6Adjoint(mesh=mesh, potential=PolynomialPotential(), scheme="simplex_commutator", m_krylov=8)

    q = torch.tensor([0.4, -0.2], dtype=torch.float64)
    p = torch.tensor([0.1, 0.3], dtype=torch.float64)
    q_std, p_std = real_std(q, p, 0.01, 1)
    q_st, p_st = real_stencil(q, p, 0.01, 1)
    psi = (torch.randn(mesh.N_nodes) + 1j * torch.randn(mesh.N_nodes)).to(torch.complex128)
    psi_out = simplex(psi, 0.01, 1)

    assert q_std.shape == q_st.shape == q.shape
    assert p_std.shape == p_st.shape == p.shape
    assert psi_out.shape == psi.shape
    assert real_std.scheme == "real_standard_33"
    assert real_stencil.scheme == "real_stencil_12"
    assert simplex.scheme == "simplex_commutator"


if __name__ == "__main__":
    test_simplex_adjoint_matches_unrolled_gradient()
    test_simplex_adjoint_reversal_and_mass_unitarity()
    test_simplex_forward_matches_numpy_solver()
    test_unified_selector_exposes_three_user_selectable_backends()
    print("Simplex adjoint tests: PASS")
