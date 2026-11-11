"""Dtype / device / shape consistency tests for the S6* Simplex adjoint.

These pin down the input contract of S6SimplexAdjoint so that a dtype or
shape mistake fails loudly at the boundary instead of deep inside Lanczos.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from s6_adjoint import S6SimplexAdjoint, SimplexMesh2D  # noqa: E402
from s6_adjoint.simplex.module import NodalPotential  # noqa: E402


@pytest.fixture(scope="module")
def mesh():
    return SimplexMesh2D(Lx=4.0, Ly=4.0, Nx=6, Ny=6, periodic=True)


def test_complex128_forward_backward_finite(mesh):
    pot = NodalPotential(
        0.05 * (mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2) + 0.02,
        dtype=torch.float64,
    )
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=8)
    psi0 = (torch.randn(mesh.N_nodes, dtype=torch.float64)
            + 1j * torch.randn(mesh.N_nodes, dtype=torch.float64)).to(torch.complex128)
    psi = psi0.clone().requires_grad_(True)
    out = eng(psi, 0.03, 2)
    assert out.dtype == torch.complex128
    assert torch.isfinite(out).all()
    loss = (out.real ** 2 + out.imag ** 2).sum()
    loss.backward()
    assert torch.isfinite(psi.grad).all()
    assert torch.isfinite(pot.values.grad).all()


def test_complex64_matches_complex128(mesh):
    torch.manual_seed(3)
    vals = 0.05 * (mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2) + 0.02
    pot64 = NodalPotential(vals, dtype=torch.float32)
    pot128 = NodalPotential(vals, dtype=torch.float64)
    eng64 = S6SimplexAdjoint(mesh, pot64, m_krylov=8)
    eng128 = S6SimplexAdjoint(mesh, pot128, m_krylov=8)
    base = torch.randn(mesh.N_nodes, dtype=torch.float64)
    psi64 = (base + 1j * torch.randn(mesh.N_nodes, dtype=torch.float64)).to(torch.complex64)
    psi128 = (base + 1j * psi64.imag.to(torch.float64)).to(torch.complex128)
    with torch.no_grad():
        o64 = eng64(psi64, 0.03, 2)
        o128 = eng128(psi128, 0.03, 2)
    assert o64.dtype == torch.complex64 and o128.dtype == torch.complex128
    assert torch.allclose(o64.to(torch.complex128), o128, atol=1e-5, rtol=1e-5)


def test_float32_potential_is_cast_to_wavefunction_dtype(mesh):
    # The backend casts V to psi.real.dtype; a float32 potential with a
    # complex128 wavefunction must therefore just work.
    pot = NodalPotential(
        0.05 * (mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2) + 0.02,
        dtype=torch.float32,
    )
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=8)
    psi0 = (torch.randn(mesh.N_nodes, dtype=torch.float64)
            + 1j * torch.randn(mesh.N_nodes, dtype=torch.float64)).to(torch.complex128)
    with torch.no_grad():
        out = eng(psi0, 0.03, 1)
    assert out.dtype == torch.complex128 and torch.isfinite(out).all()


def test_real_psi0_raises(mesh):
    pot = NodalPotential(np.zeros(mesh.N_nodes), dtype=torch.float64)
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=8)
    with pytest.raises(ValueError, match="complex wavefunction"):
        eng(torch.randn(mesh.N_nodes, dtype=torch.float64), 0.03, 1)


def test_wrong_size_psi0_raises(mesh):
    pot = NodalPotential(np.zeros(mesh.N_nodes), dtype=torch.float64)
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=8)
    bad = torch.randn(mesh.N_nodes + 1, dtype=torch.complex128)
    with pytest.raises(ValueError, match="nodal values"):
        eng(bad, 0.03, 1)


def test_nonfinite_psi0_raises(mesh):
    pot = NodalPotential(np.zeros(mesh.N_nodes), dtype=torch.float64)
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=8)
    bad = torch.full((mesh.N_nodes,), float("nan"), dtype=torch.complex128)
    with pytest.raises(ValueError, match="finite"):
        eng(bad, 0.03, 1)


if __name__ == "__main__":
    import pytest as _p
    raise SystemExit(_p.main([__file__, "-q"]))
