"""Unrolled (native-autograd) S6* Simplex-Commutator propagation.

This is the higher-order companion to the O(1)-memory adjoint in
:mod:`s6_adjoint.simplex.autograd`:

- ``S6SimplexFunction`` (used by ``S6SimplexAdjoint``) implements the reverse
  pass by hand and only supports first-order gradients.  Making *its* backward
  differentiable would nest one autograd ``grad`` per stage inside the next,
  computing derivatives of order == number of stages instead of order 2 --
  correct but asymptotically intractable.
- This module instead runs the exact same FSAL stage sequence with native
  PyTorch ops, so the standard engine handles ``create_graph=True`` and
  second differentiation normally.  That is the supported path for
  Hessian-vector products and gradient-based meta-learning (MAML-style inner
  loops) through the Simplex propagator.

Cost model: the forward graph is O(num_steps) in memory (vs O(1) for the
adjoint).  Use this for few-step inner loops and HVPs; keep the adjoint for
first-order training rollouts.  First-order gradients agree with the adjoint
to numerical precision (pinned by ``tests/test_simplex_higher_order.py``).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.func import functional_call

from .backend import SimplexTorchBackend


def propagate_unrolled(potential: torch.nn.Module,
                       backend: SimplexTorchBackend,
                       psi0: torch.Tensor,
                       h: float,
                       num_steps: int,
                       param_override: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """Run the S6* Simplex-Commutator map with native autograd.

    Args:
        potential: potential module mapping coords -> nodal values.
        backend: SimplexTorchBackend with the assembled FE operators.
        psi0: complex initial wavefunction, shape (N,).
        h: macro step size.
        num_steps: number of macro-steps.
        param_override: optional ``{name: tensor}`` replacing the module's
            parameters (e.g. MAML-adapted, graph-carrying tensors).  When
            None the module's own parameters are used.

    Returns:
        Complex propagated wavefunction, shape (N,).
    """
    h = float(h)
    num_steps = int(num_steps)
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if not torch.isfinite(psi0).all():
        raise ValueError("psi0 must contain only finite values")
    if not psi0.is_complex():
        raise ValueError("S6* Simplex-Commutator propagation requires a complex wavefunction tensor")

    coords = backend.coords.to(device=psi0.device, dtype=psi0.real.dtype)
    if param_override is None:
        V = potential(coords)
    else:
        V = functional_call(potential, param_override, (coords,))
    V = V.reshape(-1).to(device=psi0.device, dtype=psi0.real.dtype)
    if V.numel() != backend.N:
        raise ValueError(f"Potential module must return {backend.N} nodal values; got {V.numel()}")

    A = [float(x) for x in backend.A.detach().cpu().tolist()]
    B = [float(x) for x in backend.B.detach().cpu().tolist()]
    C = [float(x) for x in backend.C.detach().cpu().tolist()]

    def Bstage(x, b, c):
        return backend.potential_expm(x, V, h * b, (h ** 3) * c, inverse=False)

    def Astage(x, a):
        return backend.kinetic_expm(x, h * a, inverse=False)

    u = psi0
    if num_steps > 0:
        # Exact FSAL form matching S6SimplexFunction.forward:
        # B1 | [A1 B2 A2 B3 A3 B4 A4 B5 A5 B6 A6] | B7/B1 fusion.
        u = Bstage(u, B[0], C[0])
        for step in range(num_steps):
            for k in range(5):
                u = Astage(u, A[k])
                u = Bstage(u, B[k + 1], C[k + 1])
            u = Astage(u, A[5])
            factor = 2.0 if step < num_steps - 1 else 1.0
            u = Bstage(u, factor * B[6], factor * C[6])
    return u
