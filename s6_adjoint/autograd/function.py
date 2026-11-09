"""
PyTorch Autograd Integration for S6*-Adjoint (Release v16).
Features:
1. Strict O(1) memory with respect to integration-step count (stores only final state q_T, p_T and params).
2. Purely functional parameter evaluation via torch.func.functional_call.
3. In-place reverse state reconstruction; floating-point reconstruction error is tested near the machine-precision scale.
4. Nested higher-order VJP differentiation for the commutator correction G(q).
5. Configurable real solver branch: 'real_standard_33' (default) or 'real_stencil_12'.
"""
try:
    import torch
    import torch.nn as nn
    try:
        from torch.func import functional_call
    except ImportError:
        from torch.nn.utils.stateless import functional_call
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.coefficients import S6_REAL_A, S6_REAL_B, S6_REAL_C, COEFFICIENTS_REGISTRY

if TORCH_AVAILABLE:
    class S6HamiltonianFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q0, p0, h, num_steps, scheme_name, module, param_names, *args):
            ctx.h = float(h)
            ctx.num_steps = int(num_steps)
            ctx.param_names = tuple(param_names)

            # Backward-compatible low-level API: the original Function.apply signature
            # ended after param_names and passed only parameters. The v16.1 module may
            # additionally pass a scalar mass immediately before those parameters.
            n_params = len(ctx.param_names)
            if len(args) == n_params:
                ctx.mass = 1.0
                ctx.has_explicit_mass = False
                params = tuple(args)
            elif len(args) == n_params + 1:
                ctx.mass = float(args[0])
                ctx.has_explicit_mass = True
                params = tuple(args[1:])
            else:
                raise ValueError(
                    f"Expected {n_params} parameter tensors (legacy API) or {n_params + 1} tensors/values "
                    f"including mass; received {len(args)} arguments."
                )
            ctx.scheme_name = str(scheme_name).lower()
            if ctx.scheme_name in ['real', 'real_standard']:
                ctx.scheme_name = 'real_standard_33'
            elif ctx.scheme_name == 'real_stencil':
                ctx.scheme_name = 'real_stencil_12'
            elif ctx.scheme_name == 'real_manifold':
                ctx.scheme_name = 'real_manifold_33'

            ctx.module = module

            if ctx.num_steps < 0:
                raise ValueError("num_steps must be non-negative")
            if not (float("-inf") < ctx.h < float("inf")):
                raise ValueError("h must be finite")
            if not (float("-inf") < ctx.mass < float("inf")) or ctx.mass <= 0.0:
                raise ValueError("mass must be finite and strictly positive")

            # Select coefficients. Unknown schemes must fail loudly rather than
            # silently falling back to the real reference coefficients.
            if ctx.scheme_name in COEFFICIENTS_REGISTRY:
                reg = COEFFICIENTS_REGISTRY[ctx.scheme_name]
                if reg.get('is_complex', False):
                    raise ValueError(
                        f"Scheme '{ctx.scheme_name}' has complex coefficients (Branch II: Parabolic). "
                        "Only purely real symplectic schemes (Branch I: 'real_standard_33', 'real_stencil_12' or 'real_manifold_33') "
                        "preserve exact time-reversal bijectivity S(-h) o S(h) = I for O(1) adjoint state reconstruction."
                    )
                A_c, B_c, C_c = reg['A'], reg['B'], reg['C']
            else:
                raise ValueError(
                    f"Unknown S6* adjoint scheme '{ctx.scheme_name}'. "
                    "Supported schemes: real_standard_33, real_stencil_12, real_manifold_33, real, real_standard, real_stencil, real_manifold."
                )

            ctx.A = tuple(float(x) for x in A_c)
            ctx.B = tuple(float(x) for x in B_c)
            ctx.C = tuple(float(x) for x in C_c)

            q = q0.detach().clone()
            p = p0.detach().clone()

            def eval_V(q_tensor, p_tensors):
                param_dict = {name: p_val for name, p_val in zip(param_names, p_tensors)}
                return functional_call(module, param_dict, q_tensor)

            # Forward integration with ZERO activation caching (O(1) memory)
            with torch.no_grad():
                for step in range(ctx.num_steps):
                    for k in range(6):
                        h_b = ctx.h * ctx.B[k]
                        h3_c = (ctx.h ** 3) * ctx.C[k] / ctx.mass

                        with torch.enable_grad():
                            q_req = q.detach().requires_grad_(True)
                            V = eval_V(q_req, params)
                            F = -torch.autograd.grad(V.sum(), q_req, create_graph=True)[0]
                            half_F_sq = 0.5 * torch.sum(F ** 2)
                            G = torch.autograd.grad(half_F_sq, q_req, create_graph=False)[0]

                        p.add_(h_b * F - 2.0 * h3_c * G)
                        q.add_((ctx.h * ctx.A[k] / ctx.mass) * p)

                    # B7 stage
                    h_b = ctx.h * ctx.B[6]
                    h3_c = (ctx.h ** 3) * ctx.C[6] / ctx.mass
                    with torch.enable_grad():
                        q_req = q.detach().requires_grad_(True)
                        V = eval_V(q_req, params)
                        F = -torch.autograd.grad(V.sum(), q_req, create_graph=True)[0]
                        half_F_sq = 0.5 * torch.sum(F ** 2)
                        G = torch.autograd.grad(half_F_sq, q_req, create_graph=False)[0]
                    p.add_(h_b * F - 2.0 * h3_c * G)

            # Store strictly O(1) tensors in context
            ctx.save_for_backward(q, p, *params)
            return q, p

        @staticmethod
        def backward(ctx, grad_q_final, grad_p_final):
            saved = ctx.saved_tensors
            q_final = saved[0]
            p_final = saved[1]
            params = saved[2:]

            q = q_final.detach().clone()
            p = p_final.detach().clone()
            # PyTorch supplies None when an output is not used by the loss.
            # Treat that output's cotangent as zero so q-only/p-only losses work.
            l_q = torch.zeros_like(q_final) if grad_q_final is None else grad_q_final.detach().clone()
            l_p = torch.zeros_like(p_final) if grad_p_final is None else grad_p_final.detach().clone()

            grad_params = [torch.zeros_like(p_tensor) for p_tensor in params]

            def eval_V(q_tensor, p_tensors):
                param_dict = {name: p_val for name, p_val in zip(ctx.param_names, p_tensors)}
                return functional_call(ctx.module, param_dict, q_tensor)

            # Backward loop with in-place reverse state reconstruction and nested VJP
            with torch.enable_grad():
                def b_step_pullback_and_reverse(q_curr, p_curr, l_q_curr, l_p_curr, b_k, c_k):
                    h_b = ctx.h * b_k
                    h3_c = (ctx.h ** 3) * c_k / ctx.mass

                    q_v = q_curr.detach().requires_grad_(True)
                    V = eval_V(q_v, params)
                    F = -torch.autograd.grad(V.sum(), q_v, create_graph=True)[0]
                    half_F_sq = 0.5 * torch.sum(F ** 2)
                    G = torch.autograd.grad(half_F_sq, q_v, create_graph=True)[0]

                    kick_eff = h_b * F - 2.0 * h3_c * G
                    target = torch.sum(kick_eff * l_p_curr)

                    # State Pullback.
                    # NOTE: retain_graph=True here is load-bearing, not optional:
                    # the parameter pullback below differentiates the SAME
                    # `target` graph a second time.  Dropping retain_graph (or
                    # reordering these two grad calls) breaks the parameter
                    # gradients with "graph was freed" at backward time.
                    delta_l_q = torch.autograd.grad(target, q_v, retain_graph=True)[0]
                    l_q_curr.add_(delta_l_q)

                    # Parameter Pullback. Keep one output slot per supplied parameter,
                    # while asking autograd only about parameters that actually require grad.
                    trainable = [(i, p) for i, p in enumerate(params) if p.requires_grad]
                    if trainable:
                        trainable_idx, trainable_params = zip(*trainable)
                        p_grads = torch.autograd.grad(
                            target, trainable_params, retain_graph=False, allow_unused=True
                        )
                        for i, pg in zip(trainable_idx, p_grads):
                            if pg is not None:
                                grad_params[i].add_(pg)


                    # Exact reverse reconstruction
                    p_curr.sub_(kick_eff.detach())
                    return q_curr, p_curr, l_q_curr, l_p_curr

                def a_step_pullback_and_reverse(q_curr, p_curr, l_q_curr, l_p_curr, a_k):
                    tau = ctx.h * a_k / ctx.mass
                    l_p_curr.add_(tau * l_q_curr)
                    q_curr.sub_(tau * p_curr)
                    return q_curr, p_curr, l_q_curr, l_p_curr

                for step in range(ctx.num_steps):
                    q, p, l_q, l_p = b_step_pullback_and_reverse(q, p, l_q, l_p, ctx.B[6], ctx.C[6])
                    for k in reversed(range(6)):
                        q, p, l_q, l_p = a_step_pullback_and_reverse(q, p, l_q, l_p, ctx.A[k])
                        q, p, l_q, l_p = b_step_pullback_and_reverse(q, p, l_q, l_p, ctx.B[k], ctx.C[k])

            # Inputs are either the legacy form
            #   (q0, p0, h, num_steps, scheme_name, module, param_names, *params)
            # or v16.1 mass-aware form with mass inserted before *params.
            prefix = (l_q, l_p, None, None, None, None, None)
            if ctx.has_explicit_mass:
                prefix = prefix + (None,)
            return prefix + tuple(grad_params)
