"""MAML few-shot adaptation through BOTH meta-learning paths.

The same 1-step MAML problem is solved two ways:

Path A -- unrolled (exact): the inner loop runs through `propagate_unrolled`
    with ``create_graph=True``; the meta-gradient backpropagates through the
    whole inner adaptation.  Exact, but O(num_steps) memory.

Path B -- adjoint + second-order adjoint (O(1) memory): inner/outer
    gradients come from the O(1) first-order adjoint and the outer correction
    ``H_in @ g_out`` from `simplex_hvp`.  No trajectory is ever stored.

The two meta-gradients must agree; the script prints both and their
relative disagreement.
"""
import sys
import torch

sys.path.insert(0, "/home/hatch/workspace/analysis_s6/adj_release_final_v163")

from s6_adjoint import (
    S6SimplexAdjoint,
    S6SimplexFunction,
    SimplexMesh2D,
    propagate_unrolled,
    simplex_hvp,
)

PN = ["a", "b"]


class SoftplusQuadPotential(torch.nn.Module):
    """V(x, y) = softplus(a*r^2 + b), nonlinear in (a, b).

    The nonlinearity exercises both Hessian terms of `simplex_hvp`: the
    propagation Hessian and the potential-module Hessian.
    """

    def __init__(self, a0=0.05, b0=0.02, dtype=torch.float64):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a0, dtype=dtype))
        self.b = torch.nn.Parameter(torch.tensor(b0, dtype=dtype))

    def forward(self, coords):
        r2 = coords[:, 0] ** 2 + coords[:, 1] ** 2
        return torch.nn.functional.softplus(self.a * r2 + self.b)


def path_a_unrolled(pot, backend, psi0, h, steps, alpha, p_init, loss_in, loss_out):
    """Exact meta-gradient by double-backward through the unrolled inner loop."""
    pa = [t.clone().requires_grad_(True) for t in p_init]
    u_in = propagate_unrolled(pot, backend, psi0, h, steps,
                              param_override=dict(zip(PN, pa)))
    g_in = torch.autograd.grad(loss_in(u_in), pa, create_graph=True)
    p1 = [p - alpha * g for p, g in zip(pa, g_in)]  # graph-carrying
    u_out = propagate_unrolled(pot, backend, psi0, h, steps,
                               param_override=dict(zip(PN, p1)))
    meta = torch.autograd.grad(loss_out(u_out), pa)
    return meta, g_in


def path_b_adjoint_hvp(pot, backend, psi0, h, steps, alpha, p_init, loss_in, loss_out):
    """Meta-gradient via O(1) adjoint + O(1) second-order adjoint (HVP)."""
    pb = [t.clone().detach().requires_grad_(True) for t in p_init]
    u_in = S6SimplexFunction.apply(psi0, h, steps, pot, PN, backend, *pb)
    g_in = torch.autograd.grad(loss_in(u_in), pb)
    p1 = [(p - alpha * g).detach().requires_grad_(True) for p, g in zip(pb, g_in)]
    u_out = S6SimplexFunction.apply(psi0, h, steps, pot, PN, backend, *p1)
    g_out = torch.autograd.grad(loss_out(u_out), p1)
    hvp = simplex_hvp(psi0, h, steps, pot, PN, backend,
                      tuple(t.detach() for t in pb), tuple(g_out), loss_in)
    meta = [g - alpha * hv for g, hv in zip(g_out, hvp)]
    return meta, g_in


def rel_diff(xs, ys):
    num = max(float(x.detach().sub(y).norm()) for x, y in zip(xs, ys))
    den = max(float(x.detach().norm()) for x in xs) + 1e-300
    return num / den


def main():
    torch.manual_seed(0)
    mesh = SimplexMesh2D(Lx=6.0, Ly=6.0, Nx=8, Ny=8, periodic=True)
    n = mesh.N_nodes
    pot = SoftplusQuadPotential()
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=6)
    backend = eng.backend

    psi0 = torch.randn(n, dtype=torch.complex128) + 1j * torch.randn(n, dtype=torch.complex128)
    psi0 = psi0 / psi0.norm()
    w_in = torch.linspace(0.5, 1.5, n, dtype=torch.float64)
    w_out = torch.linspace(1.5, 0.5, n, dtype=torch.float64)
    loss_in = lambda psi: torch.real(torch.sum(w_in * torch.abs(psi) ** 2))
    loss_out = lambda psi: torch.real(torch.sum(w_out * torch.abs(psi) ** 2))

    h, steps, alpha = 0.02, 2, 0.05
    p_init = [torch.tensor(0.05, dtype=torch.float64),
              torch.tensor(0.02, dtype=torch.float64)]

    meta_a, g_in_a = path_a_unrolled(pot, backend, psi0, h, steps, alpha,
                                     p_init, loss_in, loss_out)
    meta_b, g_in_b = path_b_adjoint_hvp(pot, backend, psi0, h, steps, alpha,
                                        p_init, loss_in, loss_out)

    print("inner grads agree (rel):", f"{rel_diff(g_in_a, g_in_b):.3e}")
    print("meta-grad path A (unrolled exact):", [f"{float(m):.6e}" for m in meta_a])
    print("meta-grad path B (adjoint + HVP): ", [f"{float(m):.6e}" for m in meta_b])
    rd = rel_diff(meta_a, meta_b)
    print("meta-grad relative disagreement:  ", f"{rd:.3e}")

    assert all(torch.isfinite(m).all() for m in list(meta_a) + list(meta_b))
    assert rel_diff(g_in_a, g_in_b) < 1e-6
    assert rd < 1e-6, f"paths disagree: {rd:.3e}"
    print("OK: both paths give the same MAML meta-gradient.")


if __name__ == "__main__":
    main()
