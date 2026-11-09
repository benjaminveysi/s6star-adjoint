"""
Commutator Force and Pullback operators.
Contract: C_repo(q) = -2 * G(q) = -2 * J_F(q)^T F(q).
For conservative Hamiltonian forces J_F is symmetric, so J_F^T F = J_F F.
"""
from typing import Any
from .dynamics import HamiltonianDynamics

def compute_commutator_force(q: Any, dynamics: HamiltonianDynamics, params: Any) -> Any:
    g = dynamics.compute_G(q, params)
    return -2.0 * g


def commutator_pullback(q: Any, lambda_p: Any, dynamics: HamiltonianDynamics, params: Any) -> Any:
    vjp_g = dynamics.vjp_G(q, lambda_p, params)
    return -2.0 * vjp_g
