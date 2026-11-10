"""O(1)-in-step-count PyTorch adjoint for S6* Simplex-Commutator."""
from __future__ import annotations

import torch
from torch.func import functional_call

from .backend import SimplexTorchBackend


class S6SimplexFunction(torch.autograd.Function):
    """Custom autograd.Function implementing the FSAL Simplex macro-map.

    Cotangent convention (read before touching the backward):
    PyTorch hands us an *Euclidean* complex cotangent ``lam = dL/dpsi`` with
    the pairing ``dL = Re(lam^H dpsi)``.  The two stage types need different
    treatment:

    - Potential stages ``exp(-i W)`` are diagonal-or-commutator exponentials
      whose VJP is evaluated with the plain Euclidean pairing
      ``Re(conj(lam) * u_check)``; no mass matrix appears because ``lam``
      already is ``dL/dpsi``.
    - The kinetic stage ``exp(-i tau T)`` with ``T = M^-1 K`` is *M-unitary*,
      not Euclidean-unitary.  Its Euclidean adjoint is therefore
      ``U^{-H}_E = M U^{-1} M^{-1}``: the pullback must map the Euclidean
      covector to an M-covector (``mu = lam / m``), apply the inverse kinetic
      propagator, then map back (``lam_prev = mu_prev * m``).  See
      ``reverse_A`` below.  Forgetting either map silently biases every
      gradient that flows through a kinetic stage.
    """

    @staticmethod
    def forward(ctx, psi0, h, num_steps, module, param_names, backend, *params):
        h = float(h)
        num_steps = int(num_steps)
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if not torch.isfinite(psi0).all():
            raise ValueError("psi0 must contain only finite values")
        if not psi0.is_complex():
            raise ValueError("S6* Simplex-Commutator adjoint requires a complex wavefunction tensor")

        ctx.h = h
        ctx.num_steps = num_steps
        ctx.module = module
        ctx.param_names = tuple(param_names)
        ctx.backend = backend

        def eval_V(pvals):
            pd = {name: p for name, p in zip(ctx.param_names, pvals)}
            with torch.no_grad():
                out = functional_call(
                    module, pd,
                    (backend.coords.to(device=psi0.device, dtype=psi0.real.dtype),)
                )
            out = out.reshape(-1).to(device=psi0.device, dtype=psi0.real.dtype)
            if out.numel() != backend.N:
                raise ValueError(f"Potential module must return {backend.N} nodal values; got {out.numel()}")
            return out

        V = eval_V(params)
        u = psi0.detach().clone()
        A = [float(x) for x in backend.A.detach().cpu().tolist()]
        B = [float(x) for x in backend.B.detach().cpu().tolist()]
        C = [float(x) for x in backend.C.detach().cpu().tolist()]

        if num_steps > 0:
            def Bstage(x, b, c):
                return backend.potential_expm(x, V, h * b, (h ** 3) * c, inverse=False)

            def Astage(x, a):
                return backend.kinetic_expm(x, h * a, inverse=False)

            # Optimized FSAL form matching src.simplex_solver:
            # B1 | [A1 B2 A2 B3 A3 B4 A4 B5 A5 B6 A6] | B7/B1 fusion.
            u = Bstage(u, B[0], C[0])
            for step in range(num_steps):
                for k in range(5):
                    u = Astage(u, A[k])
                    u = Bstage(u, B[k + 1], C[k + 1])
                u = Astage(u, A[5])
                if step < num_steps - 1:
                    u = Bstage(u, 2.0 * B[6], 2.0 * C[6])
                else:
                    u = Bstage(u, B[6], C[6])

        # Only the final state and parameters are retained: independent of num_steps.
        ctx.save_for_backward(u, *params)
        return u

    @staticmethod
    def backward(ctx, grad_psi_final):
        saved = ctx.saved_tensors
        u = saved[0].detach().clone()
        params = saved[1:]
        backend: SimplexTorchBackend = ctx.backend
        module = ctx.module

        if grad_psi_final is None:
            return (None, None, None, None, None, None) + tuple(None for _ in params)

        # PyTorch's complex gradient is an Euclidean covector: dL = Re(g^H dpsi).
        lam = grad_psi_final.detach().clone()
        m = backend.mass_diag.to(device=u.device, dtype=u.real.dtype)

        trainable = [(i, p) for i, p in enumerate(params) if p.requires_grad]
        grad_params = [None for _ in params]

        def eval_V_graph():
            pd = {name: p for name, p in zip(ctx.param_names, params)}
            out = functional_call(
                module, pd,
                (backend.coords.to(device=u.device, dtype=u.real.dtype),)
            )
            out = out.reshape(-1).to(device=u.device, dtype=u.real.dtype)
            if out.numel() != backend.N:
                raise ValueError(f"Potential module must return {backend.N} nodal values; got {out.numel()}")
            return out

        with torch.enable_grad():
            # V is static over the rollout, so retain only this one local parameter graph.
            V = eval_V_graph()
            A = [float(x) for x in backend.A.detach().cpu().tolist()]
            B = [float(x) for x in backend.B.detach().cpu().tolist()]
            C = [float(x) for x in backend.C.detach().cpu().tolist()]

            def pullback_B(u_out, lam_out, b, c):
                bh = ctx.h * b
                ch3 = (ctx.h ** 3) * c
                u_in = backend.potential_expm(
                    u_out, V, bh, ch3, inverse=True
                ).detach().requires_grad_(True)
                u_check = backend.potential_expm(u_in, V, bh, ch3, inverse=False)

                # Euclidean covector pairing.  No M factor here: lam is already dL/dpsi.
                target = torch.real(torch.sum(torch.conj(lam_out.detach()) * u_check))
                wrt = [u_in] + [p for _, p in trainable]
                grads = torch.autograd.grad(target, wrt, retain_graph=True, allow_unused=True)
                lam_prev = grads[0]
                for (idx, _), g in zip(trainable, grads[1:]):
                    if g is None:
                        continue
                    grad_params[idx] = g if grad_params[idx] is None else grad_params[idx] + g
                return u_in.detach(), lam_prev.detach()

            def reverse_A(u_out, lam_out, a):
                tau = ctx.h * a
                u_prev = backend.kinetic_expm(u_out, tau, inverse=True).detach()
                # A is M-unitary, so U^H_E = M U^{-1} M^{-1}.
                mu = lam_out.detach() / m
                mu_prev = backend.kinetic_expm(mu, tau, inverse=True).detach()
                lam_prev = mu_prev * m
                return u_prev, lam_prev

            # Reverse exact FSAL sequence from the saved final state.
            # Final macro-step: B7^-1, A6^-1, B6^-1, ..., B2^-1, A1^-1.
            u, lam = pullback_B(u, lam, B[6], C[6])
            for k in reversed(range(5)):
                u, lam = reverse_A(u, lam, A[k + 1])
                u, lam = pullback_B(u, lam, B[k + 1], C[k + 1])
            u, lam = reverse_A(u, lam, A[0])

            # Earlier macro-steps: first undo the fused B7+B1 boundary, then the step interior.
            for _ in reversed(range(ctx.num_steps - 1)):
                u, lam = pullback_B(u, lam, 2.0 * B[6], 2.0 * C[6])
                u, lam = reverse_A(u, lam, A[5])
                for k in reversed(range(5)):
                    u, lam = pullback_B(u, lam, B[k + 1], C[k + 1])
                    u, lam = reverse_A(u, lam, A[k])

            # Undo the initial B1 stage that sat before the first macro-step.
            if ctx.num_steps > 0:
                u, lam = pullback_B(u, lam, B[0], C[0])

        grad_psi0 = lam
        prefix = (grad_psi0, None, None, None, None, None)
        return prefix + tuple(grad_params)
