"""
Stage Operators for A (Drift) and B (Kick + Commutator).
"""
from typing import Any, Tuple
from .dynamics import HamiltonianDynamics

def a_step_forward(q: Any, p: Any, tau: float) -> Tuple[Any, Any]:
    return q + tau * p, p

def a_step_reverse(q: Any, p: Any, tau: float) -> Tuple[Any, Any]:
    return q - tau * p, p

def a_step_pullback(lambda_q: Any, lambda_p: Any, tau: float) -> Tuple[Any, Any]:
    return lambda_q, lambda_p + tau * lambda_q

def b_step_forward(
    q: Any, p: Any, h: float, b_k: float, c_k: float, dynamics: HamiltonianDynamics, params: Any
) -> Tuple[Any, Any]:
    h_b = h * b_k
    h3_c = (h ** 3) * c_k
    f = dynamics.F(q, params)
    g = dynamics.compute_G(q, params)
    p_next = p + h_b * f - 2.0 * h3_c * g
    return q, p_next

def b_step_reverse(
    q: Any, p_next: Any, h: float, b_k: float, c_k: float, dynamics: HamiltonianDynamics, params: Any
) -> Tuple[Any, Any]:
    h_b = h * b_k
    h3_c = (h ** 3) * c_k
    f = dynamics.F(q, params)
    g = dynamics.compute_G(q, params)
    p_prev = p_next - h_b * f + 2.0 * h3_c * g
    return q, p_prev

def b_step_pullback(
    q: Any, lambda_q: Any, lambda_p: Any, h: float, b_k: float, c_k: float,
    dynamics: HamiltonianDynamics, params: Any
) -> Tuple[Any, Any, Tuple[Any, ...]]:
    h_b = h * b_k
    h3_c = (h ** 3) * c_k
    vjp_f = dynamics.vjp_F(q, lambda_p, params)
    vjp_g = dynamics.vjp_G(q, lambda_p, params)
    
    delta_lambda_q = h_b * vjp_f - 2.0 * h3_c * vjp_g
    new_lambda_q = lambda_q + delta_lambda_q
    d_params = dynamics.param_vjp_B(q, lambda_p, h_b, h3_c, params)
    return new_lambda_q, lambda_p, d_params
