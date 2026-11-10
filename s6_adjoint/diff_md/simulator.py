"""
Differentiable Molecular Dynamics Simulator Engine with S6*-Adjoint.
"""
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from ..autograd.module import S6HamiltonianODE

    class DifferentiableMDSimulator(nn.Module):
        """
        End-to-End Differentiable MD Simulator:
        Integrates M 3D particles across N time steps with O(1) memory in the number of integration steps for adjoint backpropagation.
        """
        def __init__(self, potential_module: nn.Module, mass: float = 1.0, scheme: str = 'real'):
            super().__init__()
            self.potential_module = potential_module
            self.mass = float(mass)
            if not (float('-inf') < self.mass < float('inf')) or self.mass <= 0.0:
                raise ValueError('mass must be finite and strictly positive')
            self.ode_layer = S6HamiltonianODE(potential_module, scheme=scheme, mass=self.mass)

        def forward(self, q0: torch.Tensor, p0: torch.Tensor, dt: float, num_steps: int):
            """
            Evolves atomic system (q0, p0) -> (q_T, p_T).
            dt: integration time step size.
            num_steps: number of macro-steps.
            """
            return self.ode_layer(q0, p0, dt, num_steps)
