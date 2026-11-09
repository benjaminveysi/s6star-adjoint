"""
High-Level PyTorch Layer: S6HamiltonianODE (Release v16).
Supports continuous time evolution with strict O(1) memory footprint.
Selectable real solver branches:
  - 'real_standard_33' (default)
  - 'real_stencil_12'
  - 'real_manifold_33' (manifold re-optimization, a1=-0.108)
"""
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .function import S6HamiltonianFunction

    class S6HamiltonianODE(nn.Module):
        """
        S6*-Adjoint Continuous Hamiltonian Layer:
        Transforms (q0, p0) -> (q(T), p(T)) for separable Hamiltonian:
            H(q, p) = ||p||^2 / (2m) + V_theta(q)
        Consumes strict O(1) VRAM with respect to the number of integration steps.
        
        Args:
            potential_net: nn.Module parameterizing the potential energy V_theta(q).
            scheme: 'real_standard_33' (default), 'real_stencil_12' or 'real_manifold_33'.
            mass: positive particle mass m; qdot = p/m.
        """
        def __init__(self, potential_net: nn.Module, scheme: str = 'real_standard_33', mass: float = 1.0):
            super().__init__()
            self.potential_net = potential_net
            self.scheme = scheme.lower()
            self.mass = float(mass)
            if not (float('-inf') < self.mass < float('inf')) or self.mass <= 0.0:
                raise ValueError('mass must be finite and strictly positive')
            if self.scheme in ['real', 'real_standard']:
                self.scheme = 'real_standard_33'
            elif self.scheme == 'real_stencil':
                self.scheme = 'real_stencil_12'
            elif self.scheme == 'real_manifold':
                self.scheme = 'real_manifold_33'
            valid_schemes = {'real_standard_33', 'real_stencil_12', 'real_manifold_33'}
            if self.scheme not in valid_schemes:
                raise ValueError(
                    f"Unknown or unsupported S6* adjoint scheme '{self.scheme}'. "
                    "Supported schemes: real_standard_33, real_stencil_12, real_manifold_33, real, real_standard, real_stencil, real_manifold."
                )

        def forward(self, q0: torch.Tensor, p0: torch.Tensor, h: float, num_steps: int):
            param_names = [name for name, _ in self.potential_net.named_parameters()]
            params = tuple(self.potential_net.parameters())
            return S6HamiltonianFunction.apply(
                q0, p0, h, num_steps, self.scheme, self.potential_net, param_names, self.mass, *params
            )
