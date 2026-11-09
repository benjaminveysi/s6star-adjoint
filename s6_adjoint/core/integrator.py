"""
S6HamiltonianCore: Orchestrates 13-stage reversible forward and backward adjoint integration.
Supports:
Branch I (Real Family - 100% Real, Symplectic & Exactly Reversible with machine precision < 1e-15):
  - 'real_standard_33' (Canonical spectral reference, 33 FFTs/step)
  - 'real_stencil_12'  (8th-order compact central difference stencil, 12 FFTs/step)
  - 'real' / 'real_standard' (Aliases)
"""
from typing import Any, Tuple, List, Sequence, Optional
import numpy as np
from .coefficients import S6_REAL_A, S6_REAL_B, S6_REAL_C, COEFFICIENTS_REGISTRY
from .stages import (
    a_step_forward, a_step_reverse,
    b_step_forward, b_step_reverse
)
from .pullback import s6_single_step_pullback
from .dynamics import HamiltonianDynamics


class S6HamiltonianCore:
    def __init__(
        self,
        dynamics: HamiltonianDynamics,
        scheme: str = 'real_standard_33',
        custom_coeffs: Optional[Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]] = None
    ):
        self.dynamics = dynamics
        self.scheme = scheme.lower()
        if self.scheme == 'real':
            self.scheme = 'real_standard_33'
        elif self.scheme == 'real_standard':
            self.scheme = 'real_standard_33'
        elif self.scheme == 'real_stencil':
            self.scheme = 'real_stencil_12'

        if custom_coeffs is not None:
            self.A, self.B, self.C = custom_coeffs
        elif self.scheme in COEFFICIENTS_REGISTRY:
            reg = COEFFICIENTS_REGISTRY[self.scheme]
            if reg.get('is_complex', False):
                raise ValueError(
                    f"Scheme '{self.scheme}' has complex coefficients (Branch II: Parabolic). "
                    "Only purely real symplectic schemes (Branch I: 'real_standard_33' or 'real_stencil_12') "
                    "preserve exact time-reversal bijectivity S(-h) o S(h) = I for O(1) adjoint state reconstruction."
                )
            self.A, self.B, self.C = reg['A'], reg['B'], reg['C']
        else:
            self.A, self.B, self.C = S6_REAL_A, S6_REAL_B, S6_REAL_C

    def step_forward(self, q: Any, p: Any, h: float, params: Any) -> Tuple[Any, Any]:
        """Advances state by one macro-step h across all 13 subflows."""
        for k in range(6):
            q, p = b_step_forward(q, p, h, self.B[k], self.C[k], self.dynamics, params)
            q, p = a_step_forward(q, p, h * self.A[k])
        q, p = b_step_forward(q, p, h, self.B[6], self.C[6], self.dynamics, params)
        return q, p

    def step_reverse(self, q: Any, p: Any, h: float, params: Any) -> Tuple[Any, Any]:
        """Reconstructs previous state by exact time-reversal operator S(-h)."""
        q, p = b_step_reverse(q, p, h, self.B[6], self.C[6], self.dynamics, params)
        for k in reversed(range(6)):
            q, p = a_step_reverse(q, p, h * self.A[k])
            q, p = b_step_reverse(q, p, h, self.B[k], self.C[k], self.dynamics, params)
        return q, p

    def integrate_forward(self, q0: Any, p0: Any, h: float, num_steps: int, params: Any) -> Tuple[Any, Any]:
        """Multi-step forward trajectory integration without storing intermediate activations."""
        q, p = q0, p0
        for _ in range(num_steps):
            q, p = self.step_forward(q, p, h, params)
        return q, p

    def integrate_adjoint(
        self,
        q_final: Any,
        p_final: Any,
        lambda_q_final: Any,
        lambda_p_final: Any,
        h: float,
        num_steps: int,
        params: Any
    ) -> Tuple[Any, Any, Any, Any, List[Any]]:
        """
        Reconstructs the full trajectory backwards in time with machine precision (< 1e-15)
        and accumulates analytical vector-Jacobian products (VJP) for parameters.
        """
        q = q_final
        p = p_final
        l_q = lambda_q_final
        l_p = lambda_p_final
        total_param_grads = None

        for _ in range(num_steps):
            q, p, l_q, l_p, step_grads = s6_single_step_pullback(
                q, p, l_q, l_p, h, self.dynamics, params,
                A=self.A, B=self.B, C=self.C
            )
            if total_param_grads is None:
                total_param_grads = list(step_grads)
            else:
                for idx, g in enumerate(step_grads):
                    total_param_grads[idx] = total_param_grads[idx] + g

        return q, p, l_q, l_p, total_param_grads
