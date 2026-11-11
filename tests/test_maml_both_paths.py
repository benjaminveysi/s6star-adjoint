"""Both meta-learning paths solve the same 1-step MAML problem and agree.

Path A: exact meta-gradient by double-backward through `propagate_unrolled`
        (O(num_steps) memory).
Path B: O(1)-memory meta-gradient via the first-order adjoint plus
        `simplex_hvp` for the H_in @ g_out correction.

Uses a nonlinear potential so both Hessian terms (propagation + module)
are exercised.
"""
import torch

from s6_adjoint import (
    S6SimplexAdjoint,
    S6SimplexFunction,
    SimplexMesh2D,
    propagate_unrolled,
    simplex_hvp,
)

PN = ["a", "b"]


class _SoftplusQuadPotential(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.05, dtype=dtype))
        self.b = torch.nn.Parameter(torch.tensor(0.02, dtype=dtype))

    def forward(self, coords):
        r2 = coords[:, 0] ** 2 + coords[:, 1] ** 2
        return torch.nn.functional.softplus(self.a * r2 + self.b)


def _setup():
    torch.manual_seed(0)
    mesh = SimplexMesh2D(Lx=6.0, Ly=6.0, Nx=8, Ny=8, periodic=True)
    n = mesh.N_nodes
    pot = _SoftplusQuadPotential()
    backend = S6SimplexAdjoint(mesh, pot, m_krylov=6).backend
    psi0 = torch.randn(n, dtype=torch.complex128) + 1j * torch.randn(n, dtype=torch.complex128)
    psi0 = psi0 / psi0.norm()
    w_in = torch.linspace(0.5, 1.5, n, dtype=torch.float64)
    w_out = torch.linspace(1.5, 0.5, n, dtype=torch.float64)
    loss_in = lambda psi: torch.real(torch.sum(w_in * torch.abs(psi) ** 2))
    loss_out = lambda psi: torch.real(torch.sum(w_out * torch.abs(psi) ** 2))
    p_init = [torch.tensor(0.05, dtype=torch.float64),
              torch.tensor(0.02, dtype=torch.float64)]
    return pot, backend, psi0, loss_in, loss_out, p_init


def _rel_diff(xs, ys):
    num = max(float(x.detach().sub(y).norm()) for x, y in zip(xs, ys))
    den = max(float(x.detach().norm()) for x in xs) + 1e-300
    return num / den


def test_both_paths_agree():
    pot, backend, psi0, loss_in, loss_out, p_init = _setup()
    h, steps, alpha = 0.02, 2, 0.05

    # Path A: unrolled exact.
    pa = [t.clone().requires_grad_(True) for t in p_init]
    u_in = propagate_unrolled(pot, backend, psi0, h, steps,
                              param_override=dict(zip(PN, pa)))
    g_in_a = torch.autograd.grad(loss_in(u_in), pa, create_graph=True)
    p1 = [p - alpha * g for p, g in zip(pa, g_in_a)]
    u_out = propagate_unrolled(pot, backend, psi0, h, steps,
                               param_override=dict(zip(PN, p1)))
    meta_a = torch.autograd.grad(loss_out(u_out), pa)

    # Path B: O(1) adjoint + O(1) HVP.
    pb = [t.clone().detach().requires_grad_(True) for t in p_init]
    u_in_b = S6SimplexFunction.apply(psi0, h, steps, pot, PN, backend, *pb)
    g_in_b = torch.autograd.grad(loss_in(u_in_b), pb)
    p1b = [(p - alpha * g).detach().requires_grad_(True) for p, g in zip(pb, g_in_b)]
    u_out_b = S6SimplexFunction.apply(psi0, h, steps, pot, PN, backend, *p1b)
    g_out = torch.autograd.grad(loss_out(u_out_b), p1b)
    hvp = simplex_hvp(psi0, h, steps, pot, PN, backend,
                      tuple(t.detach() for t in pb), tuple(g_out), loss_in)
    meta_b = [g - alpha * hv for g, hv in zip(g_out, hvp)]

    assert all(torch.isfinite(m).all() for m in list(meta_a) + list(meta_b))
    assert _rel_diff(g_in_a, g_in_b) < 1e-6
    assert _rel_diff(meta_a, meta_b) < 1e-6
