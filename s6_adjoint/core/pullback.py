"""
Single-Step Reversible Pullback Orchestrator across all 13 subflows.
Supports both Real-Standard, Real-Flash, and custom coefficient sets.
"""
from typing import Any, Tuple, List, Sequence, Optional
from .coefficients import S6_A, S6_B, S6_C
from .stages import a_step_reverse, a_step_pullback, b_step_reverse, b_step_pullback
from .dynamics import HamiltonianDynamics

def s6_single_step_pullback(
    q_end: Any,
    p_end: Any,
    lambda_q_end: Any,
    lambda_p_end: Any,
    h: float,
    dynamics: HamiltonianDynamics,
    params: Any,
    A: Optional[Sequence[Any]] = None,
    B: Optional[Sequence[Any]] = None,
    C: Optional[Sequence[Any]] = None,
) -> Tuple[Any, Any, Any, Any, List[Any]]:
    q = q_end
    p = p_end
    l_q = lambda_q_end
    l_p = lambda_p_end

    A_coeffs = S6_A if A is None else A
    B_coeffs = S6_B if B is None else B
    C_coeffs = S6_C if C is None else C

    # Stage B7
    l_q, l_p, d_p7 = b_step_pullback(q, l_q, l_p, h, B_coeffs[6], C_coeffs[6], dynamics, params)
    q, p = b_step_reverse(q, p, h, B_coeffs[6], C_coeffs[6], dynamics, params)
    accumulated_param_grads = list(d_p7)

    # Stages 6 down to 1
    for k in reversed(range(6)):
        l_q, l_p = a_step_pullback(l_q, l_p, h * A_coeffs[k])
        q, p = a_step_reverse(q, p, h * A_coeffs[k])

        l_q, l_p, d_pk = b_step_pullback(q, l_q, l_p, h, B_coeffs[k], C_coeffs[k], dynamics, params)
        q, p = b_step_reverse(q, p, h, B_coeffs[k], C_coeffs[k], dynamics, params)
        
        for idx, g in enumerate(d_pk):
            accumulated_param_grads[idx] = accumulated_param_grads[idx] + g

    return q, p, l_q, l_p, accumulated_param_grads
