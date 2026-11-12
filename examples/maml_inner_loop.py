"""MAML-style meta-learning through the S6* Simplex propagator.

Inner loop: adapt nodal potential values with one gradient step on task A.
Outer loop: differentiate the task-B loss w.r.t. the *initial* values,
flowing through the inner update (true second order).  Uses
``propagate_unrolled`` -- the native-autograd path that supports
``create_graph=True`` (the O(1) adjoint is first-order only).

Run:  python examples/maml_inner_loop.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from s6_adjoint import S6SimplexAdjoint, SimplexMesh2D
from s6_adjoint.simplex.module import NodalPotential
from s6_adjoint.simplex.unrolled import propagate_unrolled

torch.manual_seed(0)
np.random.seed(0)

mesh = SimplexMesh2D(Lx=6.0, Ly=6.0, Nx=8, Ny=8, periodic=True)
base = torch.as_tensor(
    0.05 * (mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2) + 0.02,
    dtype=torch.float64,
)
pot = NodalPotential(base, dtype=torch.float64)
eng = S6SimplexAdjoint(mesh, pot, m_krylov=8)
backend = eng.backend

nodes = np.asarray(mesh.nodes)
coords = torch.as_tensor(nodes, dtype=torch.float64)
mass = torch.as_tensor(mesh.M_diag, dtype=torch.float64)
weight = 1.0 + 0.2 * coords[:, 0] - 0.11 * coords[:, 1]


def packet(cx):
    x = nodes[:, 0] - cx
    y = nodes[:, 1]
    psi = np.exp(-0.5 * (x ** 2 + y ** 2)) * np.exp(1j * x)
    psi = torch.as_tensor(psi, dtype=torch.complex128)
    return psi / float(torch.linalg.vector_norm(psi))


def task_loss(psi0, params, h=0.02, steps=2):
    psi = propagate_unrolled(pot, backend, psi0, h, steps, param_override=params)
    return torch.sum(weight * (psi.real ** 2 + 0.37 * psi.imag ** 2) * mass)


psi_A = packet(-1.0)  # inner task
psi_B = packet(+1.0)  # meta task
alpha = 0.05

v0 = base.clone().requires_grad_(True)

# --- inner adaptation (differentiable) ---
loss_in = task_loss(psi_A, {"values": v0})
g_in = torch.autograd.grad(loss_in, v0, create_graph=True)[0]
v_adapted = v0 - alpha * g_in

# --- outer (meta) gradient through the inner update ---
loss_out = task_loss(psi_B, {"values": v_adapted})
meta_grad = torch.autograd.grad(loss_out, v0)[0]

print(f"inner loss:            {float(loss_in):.6e}")
print(f"outer loss (adapted):  {float(loss_out):.6e}")
print(f"meta-grad norm:        {float(meta_grad.norm()):.6e}")
print(f"meta-grad finite:      {bool(torch.isfinite(meta_grad).all())}")

# --- a few meta-descent steps on the initialization ---
v_meta = base.clone().requires_grad_(True)
opt = torch.optim.SGD([v_meta], lr=1.0)
for it in range(5):
    opt.zero_grad()
    li = task_loss(psi_A, {"values": v_meta})
    gi = torch.autograd.grad(li, v_meta, create_graph=True)[0]
    va = v_meta - alpha * gi
    lo = task_loss(psi_B, {"values": va})
    mg = torch.autograd.grad(lo, v_meta)[0]
    v_meta.grad = mg
    opt.step()
    print(f"meta-iter {it}: outer loss after adaptation = {float(lo):.6e}")
