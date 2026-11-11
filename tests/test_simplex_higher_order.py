"""Higher-order autograd through the unrolled Simplex propagator.

The O(1)-memory adjoint (S6SimplexFunction) supports first-order gradients
only.  ``propagate_unrolled`` runs the identical FSAL stage sequence with
native PyTorch ops, so the standard engine provides Hessian-vector products
and gradient-based meta-learning (MAML-style inner loops) via
``create_graph=True``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from s6_adjoint import S6SimplexAdjoint, SimplexMesh2D
from s6_adjoint.simplex.module import NodalPotential
from s6_adjoint.simplex.unrolled import propagate_unrolled


def _fixture():
    torch.manual_seed(11)
    np.random.seed(11)
    mesh = SimplexMesh2D(Lx=6.0, Ly=6.0, Nx=6, Ny=6, periodic=True)
    base = torch.as_tensor(
        0.05 * (mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2) + 0.02,
        dtype=torch.float64,
    )
    pot = NodalPotential(base, dtype=torch.float64)
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=8)
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    psi0 = torch.as_tensor(np.exp(-0.5 * (x ** 2 + y ** 2)) * np.exp(1j * x),
                           dtype=torch.complex128)
    psi0 = psi0 / float(torch.linalg.vector_norm(psi0))
    coords = torch.as_tensor(mesh.nodes, dtype=torch.float64)
    mass = torch.as_tensor(mesh.M_diag, dtype=torch.float64)
    weight = 1.0 + 0.2 * coords[:, 0] - 0.11 * coords[:, 1]
    return mesh, base, pot, eng, psi0, weight, mass


def _loss(pot, backend, psi0, weight, mass, params, h=0.02, steps=2):
    psi = propagate_unrolled(pot, backend, psi0, h, steps, param_override=params)
    return torch.sum(weight * (psi.real ** 2 + 0.37 * psi.imag ** 2) * mass)


def test_unrolled_matches_adjoint_gradient():
    _, base, pot, eng, psi0, weight, mass = _fixture()
    v = base.clone().requires_grad_(True)
    loss_u = _loss(pot, eng.backend, psi0, weight, mass, {"values": v})
    gu = torch.autograd.grad(loss_u, v)[0]

    psi_a = eng(psi0, 0.02, 2)
    loss_a = torch.sum(weight * (psi_a.real ** 2 + 0.37 * psi_a.imag ** 2) * mass)
    ga = torch.autograd.grad(loss_a, pot.values)[0]

    rel = float(torch.linalg.vector_norm(gu - ga) / torch.linalg.vector_norm(ga))
    assert rel < 1e-10


def test_hvp_matches_finite_differences():
    _, base, pot, eng, psi0, weight, mass = _fixture()
    n = base.numel()
    v = base.clone().requires_grad_(True)
    loss = _loss(pot, eng.backend, psi0, weight, mass, {"values": v})
    g = torch.autograd.grad(loss, v, create_graph=True)[0]
    d = torch.randn(n, dtype=torch.float64)
    d = d / float(torch.linalg.vector_norm(d))
    hvp = torch.autograd.grad((g * d).sum(), v, retain_graph=True)[0]
    hvp_val = float((hvp * d).sum())

    def grad_proj(delta):
        vv = (base + delta).requires_grad_(True)
        ll = _loss(pot, eng.backend, psi0, weight, mass, {"values": vv})
        gg = torch.autograd.grad(ll, vv)[0]
        return float((gg * d).sum())

    eps = 1e-5
    fd = (grad_proj(eps * d) - grad_proj(-eps * d)) / (2 * eps)
    rel = abs(hvp_val - fd) / max(abs(fd), 1e-30)
    assert rel < 2e-3


def test_maml_inner_loop_second_order():
    mesh, base, pot, eng, _, weight, mass = _fixture()
    nodes = np.asarray(mesh.nodes)
    xx, yy = nodes[:, 0], nodes[:, 1]

    def packet(cx):
        p = np.exp(-0.5 * ((xx - cx) ** 2 + yy ** 2)) * np.exp(1j * (xx - cx))
        p = torch.as_tensor(p, dtype=torch.complex128)
        return p / float(torch.linalg.vector_norm(p))
    psi_a, psi_b = packet(-1.0), packet(1.0)

    alpha = 0.05
    v0 = base.clone().requires_grad_(True)
    loss_in = _loss(pot, eng.backend, psi_a, weight, mass, {"values": v0})
    g_in = torch.autograd.grad(loss_in, v0, create_graph=True)[0]
    v_adapted = v0 - alpha * g_in  # graph-carrying inner update

    loss_out = _loss(pot, eng.backend, psi_b, weight, mass, {"values": v_adapted})
    meta_grad = torch.autograd.grad(loss_out, v0)[0]
    assert torch.isfinite(meta_grad).all()
    assert float(torch.linalg.vector_norm(meta_grad)) > 0
