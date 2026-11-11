"""Tests for the O(1)-memory second-order adjoint (simplex_hvp)."""
import gc
import resource

import pytest
import torch

from s6_adjoint import (
    S6SimplexAdjoint,
    S6SimplexFunction,
    SimplexMesh2D,
    propagate_unrolled,
    simplex_hvp,
)
from s6_adjoint.simplex.module import NodalPotential

PARAM_NAMES = ["values"]


@pytest.fixture(scope="module")
def fx():
    torch.manual_seed(0)
    mesh = SimplexMesh2D(Lx=6.0, Ly=6.0, Nx=8, Ny=8, periodic=True)
    n = mesh.N_nodes
    base = torch.as_tensor(
        0.05 * (mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2) + 0.02,
        dtype=torch.float64,
    )
    pot = NodalPotential(base, dtype=torch.float64)
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=6)
    psi0 = torch.randn(n, dtype=torch.complex128) + 1j * torch.randn(n, dtype=torch.complex128)
    psi0 = psi0 / psi0.norm()
    weight = torch.linspace(0.5, 1.5, n, dtype=torch.float64)
    loss_fn = lambda psi: torch.real(torch.sum(weight * torch.abs(psi) ** 2))
    return {
        "pot": pot, "backend": eng.backend, "psi0": psi0,
        "base": base, "loss_fn": loss_fn, "n": n,
    }


def _grad_at(fx, pp, h=0.02, steps=2):
    uu = S6SimplexFunction.apply(fx["psi0"], h, steps, fx["pot"], PARAM_NAMES,
                                 fx["backend"], pp)
    return torch.autograd.grad(fx["loss_fn"](uu), pp)[0]


def test_hvp_matches_finite_differences(fx):
    p = fx["base"].clone().requires_grad_(True)
    v = torch.randn_like(fx["base"])
    hv = simplex_hvp(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                     fx["backend"], (p,), (v,), fx["loss_fn"])[0]
    eps = 1e-6
    hv_fd = (_grad_at(fx, p + eps * v) - _grad_at(fx, p - eps * v)) / (2 * eps)
    rel = float(torch.linalg.vector_norm(hv - hv_fd) / torch.linalg.vector_norm(hv_fd))
    assert rel < 5e-5, f"HVP vs FD rel err {rel}"


def test_hvp_matches_unrolled_double_backward(fx):
    p = fx["base"].clone().requires_grad_(True)
    v = torch.randn_like(fx["base"])
    hv = simplex_hvp(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                     fx["backend"], (p,), (v,), fx["loss_fn"])[0]
    vv = fx["base"].clone().requires_grad_(True)
    uu = propagate_unrolled(fx["pot"], fx["backend"], fx["psi0"], 0.02, 2,
                            param_override={"values": vv})
    g1 = torch.autograd.grad(fx["loss_fn"](uu), vv, create_graph=True)[0]
    hvp_u = torch.autograd.grad((g1 * v).sum(), vv, retain_graph=True)[0]
    err = float(torch.max(torch.abs(hv - hvp_u)))
    assert err < 1e-8, f"HVP vs unrolled double-backward max abs err {err}"


def test_hvp_linear_loss(fx):
    # Loss linear in psi: the loss-Hessian seed is exactly zero; exercises
    # the lamd=None -> zeros path.
    torch.manual_seed(7)
    c = torch.randn(fx["n"], dtype=torch.complex128)
    lin_loss = lambda psi: torch.real(torch.sum(torch.conj(c) * psi))
    p = fx["base"].clone().requires_grad_(True)
    v = torch.randn_like(fx["base"])
    hv = simplex_hvp(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                     fx["backend"], (p,), (v,), lin_loss)[0]
    eps = 1e-6
    def _g(pp):
        uu = S6SimplexFunction.apply(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                                     fx["backend"], pp)
        return torch.autograd.grad(lin_loss(uu), pp)[0]
    hv_fd = (_g(p + eps * v) - _g(p - eps * v)) / (2 * eps)
    rel = float(torch.linalg.vector_norm(hv - hv_fd) / (torch.linalg.vector_norm(hv_fd) + 1e-30))
    assert rel < 5e-5, f"linear-loss HVP vs FD rel err {rel}"


def test_hvp_zero_direction_gives_zero(fx):
    p = fx["base"].clone()
    v = torch.zeros_like(fx["base"])
    hv = simplex_hvp(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                     fx["backend"], (p,), (v,), fx["loss_fn"])[0]
    assert float(hv.norm()) < 1e-12


def test_maml_outer_gradient_matches_fd(fx):
    # One-step MAML: meta_grad = g_out - alpha * H_in @ g_out, checked
    # against central FD of the full inner-adaptation objective.
    alpha = 0.05
    torch.manual_seed(11)
    p0 = fx["base"].clone()
    vin = torch.randn_like(fx["base"])

    def maml_objective(pp):
        uu_in = S6SimplexFunction.apply(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                                        fx["backend"], pp)
        g_in = torch.autograd.grad(fx["loss_fn"](uu_in), pp)[0]
        p1 = pp - alpha * g_in
        uu_out = S6SimplexFunction.apply(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                                         fx["backend"], p1)
        return fx["loss_fn"](uu_out)

    p0r = p0.clone().requires_grad_(True)
    uu_in = S6SimplexFunction.apply(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                                    fx["backend"], p0r)
    g_in = torch.autograd.grad(fx["loss_fn"](uu_in), p0r)[0]
    p1 = (p0 - alpha * g_in).detach().requires_grad_(True)
    uu_out = S6SimplexFunction.apply(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                                     fx["backend"], p1)
    g_out = torch.autograd.grad(fx["loss_fn"](uu_out), p1)[0]
    hvp_in = simplex_hvp(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                         fx["backend"], (p0r,), (g_out,), fx["loss_fn"])[0]
    meta = g_out - alpha * hvp_in

    eps = 2e-6
    mp = maml_objective((p0 + eps * vin).requires_grad_(True))
    mm = maml_objective((p0 - eps * vin).requires_grad_(True))
    fd_dir = float(mp - mm) / (2 * eps)
    an_dir = float((meta * vin).sum())
    rel = abs(an_dir - fd_dir) / abs(fd_dir)
    assert rel < 1e-4, f"MAML outer-gradient rel err {rel}"


def test_hvp_memory_flat_in_steps(fx):
    # O(1) in step count: 16x more stages must not grow peak RSS.
    p = fx["base"].clone()
    v = torch.randn_like(fx["base"])

    def rss_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    simplex_hvp(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                fx["backend"], (p,), (v,), fx["loss_fn"])
    gc.collect()
    simplex_hvp(fx["psi0"], 0.02, 2, fx["pot"], PARAM_NAMES,
                fx["backend"], (p,), (v,), fx["loss_fn"])
    gc.collect()
    r1 = rss_mb()
    simplex_hvp(fx["psi0"], 0.02, 32, fx["pot"], PARAM_NAMES,
                fx["backend"], (p,), (v,), fx["loss_fn"])
    gc.collect()
    r2 = rss_mb()
    assert r2 - r1 < 5.0, f"RSS grew by {r2 - r1:.1f} MB from 2 to 32 steps"
