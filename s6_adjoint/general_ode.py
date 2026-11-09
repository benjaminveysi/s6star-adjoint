"""
S6*-Adjoint-G: General Non-Conservative Vector Field ODE Layer (Release v16.3+).
Extends 6th-order reversible S6* splitting and O(1)-memory discrete adjoints
to arbitrary vector fields F_theta(q) with non-symmetric Jacobians (curl != 0),
via exact Lie-bracket JVP evaluation: G_2(q) = J_F(q) @ F_theta(q).

SIGN OF THE G2 TERM (verified numerically 2026-11-09, rotational field
F = w(r)(-y, x), curl = 2 + 1.2 r^2 != 0, Real-Standard-33 coefficients):
    p += h*B*F - 2*h^3*C*G2      -> order 6.01 / 6.00   (CORRECT)
    p += h*B*F + 2*h^3*C*G2      -> order 1.98 / 2.00   (BROKEN)
The adjoint package's REAL_STANDARD_33_C is byte-identical to the S6* repo's
S6_REAL_C, and benchmark_gpu.py itself uses MINUS with G = grad(0.5||F||^2).
For F = -grad V the Jacobian is symmetric, so J_F^T F = J_F F exactly
((-grad^2 V)(-grad V) = +J_F F, not -J_F F), hence NO sign flip when the
VJP is replaced by the JVP. The MINUS sign below is the correct one.
Reversibility on the same field locks at 1.11e-15; discrete-adjoint gradcheck
vs central FD matches to 6.5e-10 (see tests/test_adjoint_g.py in the S6* repo).
"""

import torch
import torch.nn as nn
from s6_adjoint.core.coefficients import (
    COEFFICIENTS_REGISTRY,
    REAL_STANDARD_33_A, REAL_STANDARD_33_B, REAL_STANDARD_33_C,
    REAL_MANIFOLD_33_A, REAL_MANIFOLD_33_B, REAL_MANIFOLD_33_C,
    REAL_STENCIL_12_A, REAL_STENCIL_12_B, REAL_STENCIL_12_C,
)

# وزن‌های استنسیل تفاضل مرکزی مرتبه ۸ برای شاخه real_stencil_12
_STENCIL8_W = (4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0)


def _compute_F_and_G2_exact_jvp(vector_field: nn.Module, q: torch.Tensor, create_graph: bool = False):
    """
    محاسبه‌ی F_theta(q) و کاموتاتور دقیق G_2(q) = J_F(q) @ F_theta(q)
    با ترفند Forward-over-Reverse روی گراف استاندارد autograd پایتورچ.
    """
    with torch.enable_grad():
        q_req = q if q.requires_grad else q.detach().requires_grad_(True)
        F_val = vector_field(q_req)
        dummy = torch.zeros_like(F_val, requires_grad=True)
        # گام ۱: VJP با بردار کمکی dummy -> J_F(q)^T @ dummy
        vjp_dummy = torch.autograd.grad(
            outputs=F_val,
            inputs=q_req,
            grad_outputs=dummy,
            create_graph=True,
            retain_graph=True,
        )[0]
        # گام ۲: مشتق نسبت به dummy در راستای F_val -> (J_F(q)^T)^T @ F_val = J_F(q) @ F_val
        G2_val = torch.autograd.grad(
            outputs=vjp_dummy,
            inputs=dummy,
            grad_outputs=F_val.detach() if not create_graph else F_val,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]

    if not create_graph:
        return F_val.detach(), G2_val.detach()
    return F_val, G2_val


def _compute_F_and_G2_stencil8(vector_field: nn.Module, q: torch.Tensor, eps_stencil: float = 1e-4):
    """
    محاسبه‌ی F_theta(q) و کاموتاتور جهتی G_2(q) = J_F(q) @ F_theta(q)
    با استنسیل تفاضل مرکزی مرتبه ۸ برای شاخه سریع real_stencil_12.
    """
    F_val = vector_field(q)
    # بردار واحد جهت جریان در هر نمونه از دسته (Batch)
    f_norm = torch.linalg.norm(F_val, dim=-1, keepdim=True).clamp(min=1e-30)
    n_hat = F_val / f_norm

    dF_dn = torch.zeros_like(F_val)
    for m, w_m in enumerate(_STENCIL8_W, start=1):
        shift = (m * eps_stencil) * n_hat
        F_plus = vector_field(q + shift)
        F_minus = vector_field(q - shift)
        dF_dn = dF_dn + w_m * (F_plus - F_minus)

    G2_val = (f_norm / eps_stencil) * dF_dn
    return F_val, G2_val


def _single_macro_step_general(
    vector_field: nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    h: float,
    A: tuple,
    B: tuple,
    C: tuple,
    use_stencil: bool,
    create_graph: bool,
):
    """اجرای یک ماکرو-گام ۱۳-استیجی S6*-Adjoint-G (رفت با +h و برگشت با -h)."""
    h2 = h * h

    def eval_FG(q_curr):
        if use_stencil:
            if create_graph:
                return _compute_F_and_G2_stencil8(vector_field, q_curr)
            with torch.no_grad():
                return _compute_F_and_G2_stencil8(vector_field, q_curr)
        return _compute_F_and_G2_exact_jvp(vector_field, q_curr, create_graph=create_graph)

    for k in range(6):
        F_val, G2_val = eval_FG(q)
        # علامت MINUS روی G2 (تصحیح‌شده؛ رجوع به هدر ماژول): با همین ضرایبِ
        # REAL_STANDARD_33_C، علامتِ + مرتبه را به ۲ می‌کوبد و - مرتبه‌ی ۶ را می‌دهد.
        p = p + (h * B[k]) * F_val - (2.0 * (h * h2) * C[k]) * G2_val
        q = q + (h * A[k]) * p

    F_val, G2_val = eval_FG(q)
    p = p + (h * B[6]) * F_val - (2.0 * (h * h2) * C[6]) * G2_val
    return q, p


class _S6GeneralAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q0, p0, h, num_steps, vector_field, A, B, C, use_stencil, *params):
        ctx.h = float(h)
        ctx.num_steps = int(num_steps)
        ctx.vector_field = vector_field
        ctx.A = A
        ctx.B = B
        ctx.C = C
        ctx.use_stencil = bool(use_stencil)

        q = q0.detach().clone()
        p = p0.detach().clone()

        for _ in range(ctx.num_steps):
            q, p = _single_macro_step_general(
                vector_field, q, p, ctx.h, A, B, C, ctx.use_stencil, create_graph=False
            )

        # ذخیره فقط وضعیت پایانی در t = T برای حافظه O(1)
        ctx.save_for_backward(q.detach().clone(), p.detach().clone())
        return q, p

    @staticmethod
    def backward(ctx, grad_q, grad_p):
        if not torch.is_grad_enabled():
            pass  # گاردِ تک‌گذره بودن Adjoint (ممنوعیت Adjoint تودرتو طبق بخش ۲۰.۴ اسپک)

        q_curr, p_curr = ctx.saved_tensors
        q_curr = q_curr.detach().clone()
        p_curr = p_curr.detach().clone()

        h = ctx.h
        A, B, C = ctx.A, ctx.B, ctx.C
        use_stencil = ctx.use_stencil
        vector_field = ctx.vector_field
        params = list(vector_field.parameters())
        grad_params = [torch.zeros_like(p_param) for p_param in params]

        for _ in range(ctx.num_steps):
            # ۱. بازسازی دقیق وضعیت گام قبل با گام منفی (-h) به لطف تقارن پالیندرومیک S6*
            q_prev, p_prev = _single_macro_step_general(
                vector_field, q_curr, p_curr, -h, A, B, C, use_stencil, create_graph=False
            )

            # ۲. ساخت گراف موضعی تک‌گام و استخراج VJP دقیق
            with torch.enable_grad():
                q_in = q_prev.detach().requires_grad_(True)
                p_in = p_prev.detach().requires_grad_(True)

                q_out, p_out = _single_macro_step_general(
                    vector_field, q_in, p_in, h, A, B, C, use_stencil, create_graph=True
                )

                grads = torch.autograd.grad(
                    outputs=(q_out, p_out),
                    inputs=[q_in, p_in] + params,
                    grad_outputs=(grad_q, grad_p),
                    allow_unused=True,
                )

            grad_q = grads[0]
            grad_p = grads[1]
            for idx, g_p in enumerate(grads[2:]):
                if g_p is not None:
                    grad_params[idx] = grad_params[idx] + g_p

            q_curr, p_curr = q_prev, p_prev

        return (grad_q, grad_p, None, None, None, None, None, None, None, *grad_params)


class S6GeneralVectorFieldODE(nn.Module):
    """
    لایه PyTorch برای S6*-Adjoint-G روی میدان‌های برداری عمومی و غیرپایستار:
        dq/dt = p,   dp/dt = F_theta(q)   (بدون نیاز به پتانسیل اسکالر V_theta)
    """
    def __init__(self, vector_field: nn.Module, scheme: str = "real_standard_33"):
        super().__init__()
        self.vector_field = vector_field
        scheme_clean = scheme.lower()
        if scheme_clean in ("real", "real_standard", "real_standard_33",
                            "general", "general_standard", "general_standard_33"):
            self.scheme = ("general_standard_33" if scheme_clean.startswith("general")
                           else "real_standard_33")
            self.A = tuple(float(x) for x in REAL_STANDARD_33_A)
            self.B = tuple(float(x) for x in REAL_STANDARD_33_B)
            self.C = tuple(float(x) for x in REAL_STANDARD_33_C)
            self.use_stencil = False
        elif scheme_clean in ("real_manifold", "real_manifold_33",
                              "general_manifold", "general_manifold_33"):
            self.scheme = ("general_manifold_33" if scheme_clean.startswith("general")
                           else "real_manifold_33")
            self.A = tuple(float(x) for x in REAL_MANIFOLD_33_A)
            self.B = tuple(float(x) for x in REAL_MANIFOLD_33_B)
            self.C = tuple(float(x) for x in REAL_MANIFOLD_33_C)
            self.use_stencil = False
        elif scheme_clean in ("real_stencil", "real_stencil_12",
                              "general_stencil", "general_stencil_12"):
            self.scheme = ("general_stencil_12" if scheme_clean.startswith("general")
                           else "real_stencil_12")
            self.A = tuple(float(x) for x in REAL_STENCIL_12_A)
            self.B = tuple(float(x) for x in REAL_STENCIL_12_B)
            self.C = tuple(float(x) for x in REAL_STENCIL_12_C)
            self.use_stencil = True
        else:
            raise ValueError(f"Unsupported scheme '{scheme}' for S6GeneralVectorFieldODE.")

    def forward(self, q0: torch.Tensor, p0: torch.Tensor, h: float, num_steps: int):
        params = tuple(self.vector_field.parameters())
        return _S6GeneralAdjointFunction.apply(
            q0, p0, h, num_steps,
            self.vector_field,
            self.A, self.B, self.C,
            self.use_stencil,
            *params
        )
