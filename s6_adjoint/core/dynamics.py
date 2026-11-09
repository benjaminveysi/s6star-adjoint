"""
Dynamics Protocol and Matrix-Free VJP Primitives.
"""
from typing import Callable, Tuple, Any


class HamiltonianDynamics:
    def __init__(
        self,
        force_fn: Callable[[Any, Any], Any],
        vjp_F_fn: Callable[[Any, Any, Any], Any],
        vjp_G_fn: Callable[[Any, Any, Any], Any],
        param_vjp_B_fn: Callable[[Any, Any, float, float, Any], Tuple[Any, ...]]
    ):
        self.F = force_fn
        self.vjp_F = vjp_F_fn
        self.vjp_G = vjp_G_fn
        self.param_vjp_B = param_vjp_B_fn

    def compute_G(self, q: Any, params: Any) -> Any:
        """Return G(q) = J_F(q)^T F(q) = grad_q(1/2 ||F(q)||^2).

        For conservative Hamiltonian forces F = -grad V, J_F is symmetric,
        so this agrees with the J_F F commutator form used by S6-star.
        """
        f = self.F(q, params)
        return self.vjp_F(q, f, params)
