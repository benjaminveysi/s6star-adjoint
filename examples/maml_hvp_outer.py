"""One-step MAML outer gradient via the O(1) second-order adjoint.

Inner adaptation:  p1 = p0 - alpha * grad(L_in)(p0)   (O(1) adjoint)
Outer gradient:    dL_out/dp0 = g_out - alpha * H_in(p0) @ g_out,
where the Hessian-vector product comes from `simplex_hvp` -- O(1) memory,
no unrolled trajectory, no nested autograd graphs.
"""
import sys
import torch

sys.path.insert(0, "/home/hatch/workspace/analysis_s6/adj_release_final_v163")

from s6_adjoint import (
    S6SimplexAdjoint,
    S6SimplexFunction,
    SimplexMesh2D,
    simplex_hvp,
)
from s6_adjoint.simplex.module import NodalPotential

PN = ["values"]


def main():
    torch.manual_seed(0)
    mesh = SimplexMesh2D(Lx=6.0, Ly=6.0, Nx=8, Ny=8, periodic=True)
    n = mesh.N_nodes
    base = torch.as_tensor(
        0.05 * (mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2) + 0.02,
        dtype=torch.float64,
    )
    pot = NodalPotential(base, dtype=torch.float64)
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=6)
    backend = eng.backend

    psi0 = torch.randn(n, dtype=torch.complex128) + 1j * torch.randn(n, dtype=torch.complex128)
    psi0 = psi0 / psi0.norm()
    w_in = torch.linspace(0.5, 1.5, n, dtype=torch.float64)
    w_out = torch.linspace(1.5, 0.5, n, dtype=torch.float64)
    loss_in = lambda psi: torch.real(torch.sum(w_in * torch.abs(psi) ** 2))
    loss_out = lambda psi: torch.real(torch.sum(w_out * torch.abs(psi) ** 2))

    h, steps, alpha = 0.02, 2, 0.05
    p0 = base.clone().requires_grad_(True)

    # Inner loop (first-order O(1) adjoint).
    u_in = S6SimplexFunction.apply(psi0, h, steps, pot, PN, backend, p0)
    g_in = torch.autograd.grad(loss_in(u_in), p0)[0]
    p1 = (p0 - alpha * g_in).detach().requires_grad_(True)

    # Outer loss at the adapted parameters.
    u_out = S6SimplexFunction.apply(psi0, h, steps, pot, PN, backend, p1)
    g_out = torch.autograd.grad(loss_out(u_out), p1)[0]

    # Meta-gradient through the inner update, via the O(1) HVP.
    hvp_in = simplex_hvp(psi0, h, steps, pot, PN, backend, (p0,), (g_out,), loss_in)[0]
    meta_grad = g_out - alpha * hvp_in

    print(f"|g_in|     = {float(g_in.norm()):.6e}")
    print(f"|g_out|    = {float(g_out.norm()):.6e}")
    print(f"|H_in@g_out| = {float(hvp_in.norm()):.6e}")
    print(f"|meta_grad|  = {float(meta_grad.norm()):.6e}")
    assert torch.isfinite(meta_grad).all() and float(meta_grad.norm()) > 0


if __name__ == "__main__":
    main()
