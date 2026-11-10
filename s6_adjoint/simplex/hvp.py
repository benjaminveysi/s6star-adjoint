"""O(1)-in-step-count second-order adjoint (HVP) for the S6* Simplex map.

This module implements Hessian-vector products ``Hv = d/dp[dL/dp] . v``
through :class:`S6SimplexFunction`'s FSAL macro-map without ever building the
nested per-stage autograd graphs that a naive ``create_graph=True`` double
backward would need (those compound exponentially across stages).

Method -- forward-over-reverse (second-order discrete adjoint)
--------------------------------------------------------------
Write the stages as ``u_{k+1} = S_k(u_k; V(p))`` with ``V`` the nodal
potential (static over the rollout).  Let ``g(p) = dL/dp`` be the first-order
adjoint gradient.  Then ::

    Hv = d/dp[g] . v
       = (dV/dp)^T . Hv_V + (d^2V/dp^2[v])^T . g_V ,

where ``g_V = dL/dV`` and ``Hv_V`` is the V-space HVP.  ``Hv_V`` is obtained
by *forward-mode differentiating the backward (adjoint) recurrence itself*:

- forward tangent sweep: ``ud_1 = 0``,
  ``ud_{k+1} = J^u_k . ud_k + J^V_k . Vd``  (``Vd = (dV/dp).v``);
- loss seed: ``lamd_{K+1} = H_ell . ud_{K+1}`` via a local double-grad;
- backward sweep carrying ``(u, ud, lam, lamd)``.  Per potential stage the
  second-order terms are the real-Jacobian-vector products of the stage VJP
  maps, evaluated with the local double-grad pattern
  ``grad(2*Re<grad(tgt), dir>)`` (the factor 2 matches PyTorch's complex
  ``dL = 2 Re<grad, du>`` convention); the kinetic stages are p-independent
  and linear, so they only transport the tangents.

Memory is O(1) in the step count: only the current ``(u, ud, lam, lamd)``
vectors plus stage-local graphs are ever alive.  No trajectory is stored and
no grad-of-grad chain crosses a stage boundary.

References: standard second-order discrete-adjoint theory; the stage VJP
convention (Euclidean ``Re(conj(lam) * u_check)`` for potential stages,
M-weighted pullback for kinetic stages) is documented on
``S6SimplexFunction``.
"""
from __future__ import annotations

import torch
from torch.func import functional_call, jvp


def _forward_stage_list(A, B, C, num_steps):
    """Stage list in forward order; mirrors S6SimplexFunction.forward exactly."""
    stages = []
    if num_steps > 0:
        stages.append(("B", B[0], C[0]))
        for step in range(num_steps):
            for k in range(5):
                stages.append(("A", A[k]))
                stages.append(("B", B[k + 1], C[k + 1]))
            stages.append(("A", A[5]))
            if step < num_steps - 1:
                stages.append(("B", 2.0 * B[6], 2.0 * C[6]))
            else:
                stages.append(("B", B[6], C[6]))
    return stages


def _reverse_stage_list(A, B, C, num_steps):
    """Stage list in reverse order; mirrors S6SimplexFunction.backward exactly."""
    stages = []
    if num_steps > 0:
        stages.append(("B", B[6], C[6]))
        for k in reversed(range(5)):
            stages.append(("A", A[k + 1]))
            stages.append(("B", B[k + 1], C[k + 1]))
        stages.append(("A", A[0]))
        for _ in reversed(range(num_steps - 1)):
            stages.append(("B", 2.0 * B[6], 2.0 * C[6]))
            stages.append(("A", A[5]))
            for k in reversed(range(5)):
                stages.append(("B", B[k + 1], C[k + 1]))
                stages.append(("A", A[k]))
        stages.append(("B", B[0], C[0]))
    return stages


def simplex_hvp(psi0, h, num_steps, module, param_names, backend, params, v_params, loss_fn):
    """Hessian-vector product through the S6* Simplex propagator, O(1) memory.

    Computes ``Hv = d^2L/dp^2 . v`` where ``L = loss_fn(u_final)`` and
    ``u_final`` is the FSAL macro-map applied to ``psi0`` (the same map as
    ``S6SimplexFunction``).  ``dL/dp`` here is exactly the gradient that
    ``S6SimplexFunction.backward`` computes.

    Parameters
    ----------
    psi0 : complex tensor
        Initial wavefunction (no grad required; it is detached internally).
    h, num_steps, module, param_names, backend :
        Same meaning as in ``S6SimplexFunction``.
    params : tuple of tensors
        Module parameters (the ``*params`` of ``S6SimplexFunction``).
    v_params : tuple of tensors
        HVP direction; same structure/shapes as ``params``.
    loss_fn : callable
        ``loss_fn(psi) -> real scalar``; the user's loss as a pure function
        of the final state.

    Returns
    -------
    tuple of tensors
        ``Hv`` matching the structure of ``params`` (detached).
    """
    h = float(h)
    num_steps = int(num_steps)
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if len(params) != len(v_params):
        raise ValueError("params and v_params must have the same structure")
    dev = psi0.device
    rdtype = psi0.real.dtype

    A = [float(x) for x in backend.A.detach().cpu().tolist()]
    B = [float(x) for x in backend.B.detach().cpu().tolist()]
    C = [float(x) for x in backend.C.detach().cpu().tolist()]
    m = backend.mass_diag.to(device=dev, dtype=rdtype)
    coords = backend.coords.to(device=dev, dtype=rdtype)
    param_names = tuple(param_names)

    def eval_V_nograd(pvals):
        pd = {n: p.detach() for n, p in zip(param_names, pvals)}
        with torch.no_grad():
            out = functional_call(module, pd, (coords,))
        out = out.reshape(-1).to(device=dev, dtype=rdtype)
        if out.numel() != backend.N:
            raise ValueError(f"Potential module must return {backend.N} nodal values")
        return out

    V = eval_V_nograd(params)

    # Vd = (dV/dp) . v : one forward-mode pass through the potential module.
    def _V_of(*pvals):
        pd = {n: p for n, p in zip(param_names, pvals)}
        return functional_call(module, pd, (coords,)).reshape(-1)

    with torch.enable_grad():
        _, Vd = jvp(_V_of,
                    tuple(p.detach() for p in params),
                    tuple(v.detach() for v in v_params))
    Vd = Vd.to(device=dev, dtype=rdtype)

    fwd_stages = _forward_stage_list(A, B, C, num_steps)

    # ---- Phase 1: forward sweep + forward tangent sweep (O(1) memory). ----
    u = psi0.detach().clone().to(dev)
    ud = torch.zeros_like(u)
    for st in fwd_stages:
        if st[0] == "A":
            tau = h * st[1]
            u = backend.kinetic_expm(u, tau, inverse=False)
            # Kinetic stage is linear and p-independent: JVP is the map itself.
            ud = backend.kinetic_expm(ud, tau, inverse=False)
        else:
            _, b, c = st
            bh, ch3 = h * b, (h ** 3) * c
            # (dP/dV) . Vd at fixed u (forward-mode through the stage).
            _, jv = jvp(lambda VV, uf=u: backend.potential_expm(uf, VV, bh, ch3, inverse=False),
                        (V,), (Vd,))
            ud = backend.potential_expm(ud, V, bh, ch3, inverse=False) + jv
            u = backend.potential_expm(u, V, bh, ch3, inverse=False)
    u_final = u.detach()
    ud_final = ud.detach()

    # ---- Phase 2: loss cotangent and loss Hessian-vector seed. ----
    with torch.enable_grad():
        uc = u_final.clone().requires_grad_(True)
        Lc = loss_fn(uc)
        lam_f = torch.autograd.grad(Lc, uc, create_graph=True)[0]
        # lamd_final = d/dp[dL/du] . v.  With PyTorch's complex convention
        # dL = Re<grad, du>,  grad(Re<grad(f), v>) is the real-HVP, so no
        # extra factor is needed.  A linear loss has zero Hessian: psi_l
        # then carries no graph and the seed is exactly zero.
        psi_l = torch.real(torch.sum(torch.conj(lam_f) * ud_final))
        if torch.is_tensor(psi_l) and psi_l.grad_fn is not None:
            lamd_f = torch.autograd.grad(psi_l, uc, allow_unused=True)[0]
        else:
            lamd_f = None
    lam = lam_f.detach().clone()
    lamd = lamd_f.detach().clone() if lamd_f is not None else torch.zeros_like(lam)

    # ---- Phase 3: backward sweep, forward-over-reverse (O(1) memory). ----
    rev_stages = _reverse_stage_list(A, B, C, num_steps)
    u = u_final.clone()
    ud = ud_final.clone()
    gV = torch.zeros_like(V)
    HvV = torch.zeros_like(V)

    for st in rev_stages:
        if st[0] == "A":
            tau = h * st[1]
            u = backend.kinetic_expm(u, tau, inverse=True).detach()
            ud = backend.kinetic_expm(ud, tau, inverse=True).detach()
            # M-unitary pullback is linear: same transport for lamd.
            lamd = (backend.kinetic_expm(lamd / m, tau, inverse=True) * m).detach()
            lam = (backend.kinetic_expm(lam / m, tau, inverse=True) * m).detach()
        else:
            _, b, c = st
            bh, ch3 = h * b, (h ** 3) * c
            lam_out = lam.detach()
            lamd_out = lamd.detach()

            # Recompute stage input; tangent through the inverse stage.
            u_in = backend.potential_expm(u, V, bh, ch3, inverse=True).detach()
            _, jv_inv = jvp(
                lambda VV, uo=u: backend.potential_expm(uo, VV, bh, ch3, inverse=True),
                (V,), (Vd,))
            ud_in = backend.potential_expm(ud, V, bh, ch3, inverse=True).detach() + jv_inv

            with torch.enable_grad():
                # First-order stage VJPs (same Euclidean convention as the adjoint).
                u_leaf = u_in.clone().requires_grad_(True)
                V_leaf = V.clone().requires_grad_(True)
                u_check = backend.potential_expm(u_leaf, V_leaf, bh, ch3, inverse=False)
                tgt = torch.real(torch.sum(torch.conj(lam_out) * u_check))
                lam_in, sV = torch.autograd.grad(tgt, [u_leaf, V_leaf], create_graph=True)
                gV = gV + sV.detach()

                # Second-order stage terms: real-Jacobian-vector products of
                # the stage VJP maps in the direction (ud_in, Vd).  Same
                # no-factor-2 HVP pattern as the loss seed above.
                psi_s = (torch.real(torch.sum(torch.conj(lam_in) * ud_in.detach()))
                         + torch.sum(sV * Vd))
                B_lam, B_s = torch.autograd.grad(psi_s, [u_leaf, V_leaf])
                HvV = HvV + B_s.detach()

                # (J^u)^T lamd_out and (J^V)^T lamd_out.
                u_l2 = u_in.clone().requires_grad_(True)
                V_l2 = V.clone().requires_grad_(True)
                uc2 = backend.potential_expm(u_l2, V_l2, bh, ch3, inverse=False)
                tgt_d = torch.real(torch.sum(torch.conj(lamd_out) * uc2))
                ld_u, ld_V = torch.autograd.grad(tgt_d, [u_l2, V_l2])
                HvV = HvV + ld_V.detach()

                lamd = (ld_u.detach() + B_lam.detach()).detach()
                lam = lam_in.detach()
            u = u_in.detach()
            ud = ud_in.detach()

    # ---- Phase 4: chain the V-space HVP back to parameter space. ----
    # Hv = (dV/dp)^T HvV + (d^2V/dp^2[v])^T gV.
    with torch.enable_grad():
        pg = [p.detach().requires_grad_(True) for p in params]
        pd = {n: p for n, p in zip(param_names, pg)}
        Vg = functional_call(module, pd, (coords,)).reshape(-1).to(device=dev, dtype=rdtype)
        t1 = torch.autograd.grad(torch.sum(HvV.detach() * Vg), pg,
                                 retain_graph=True, allow_unused=True)
        gp1 = torch.autograd.grad(torch.sum(gV.detach() * Vg), pg,
                                  create_graph=True, retain_graph=True, allow_unused=True)
        # Second-order term of the potential module.  For a linear module
        # (e.g. NodalPotential) gp1 carries no graph and the term is
        # identically zero -- the standard double-grad HVP then has nothing
        # to differentiate.
        t2 = [None for _ in pg]
        terms = [(g, vv.detach()) for g, vv in zip(gp1, v_params) if g is not None]
        if terms:
            psi2 = sum(torch.sum(g * vv) for g, vv in terms)
            if torch.is_tensor(psi2) and psi2.grad_fn is not None:
                t2 = torch.autograd.grad(psi2, pg, allow_unused=True)

    Hv = tuple(
        ((a if a is not None else torch.zeros_like(p))
         + (b if b is not None else torch.zeros_like(p)))
        for a, b, p in zip(t1, t2, params)
    )
    return Hv
