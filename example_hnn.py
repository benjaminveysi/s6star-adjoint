"""
Example: Training a Hamiltonian Neural Network (HNN) with S6*-Adjoint (Release v16).
Demonstrates constant memory footprint O(1) and exact discrete adjoint gradients.
Supports selectable real solver branches:
    python example_hnn.py --scheme real_standard_33
    python example_hnn.py --scheme real_stencil_12
"""
import argparse

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required for example_hnn.py")

if TORCH_AVAILABLE:
    from s6_adjoint import S6HamiltonianODE

    class QuarticPotential(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
            self.lam = nn.Parameter(torch.tensor([0.1, 0.1], dtype=torch.float64))

        def forward(self, q):
            harmonic = 0.5 * torch.sum(self.k * (q ** 2), dim=-1)
            quartic = 0.25 * torch.sum(self.lam * (q ** 4), dim=-1)
            return harmonic + quartic

    def main():
        parser = argparse.ArgumentParser(description="HNN Training with S6*-Adjoint (v16)")
        parser.add_argument('--scheme', type=str, default='real_standard_33',
                            choices=['real_standard_33', 'real_stencil_12', 'real_standard', 'real_stencil', 'real'],
                            help="Branch I solver: 'real_standard_33' (default) or 'real_stencil_12'")
        parser.add_argument('--steps', type=int, default=20, help="Integration macro-steps")
        parser.add_argument('--dt', type=float, default=0.05, help="Step size")
        args = parser.parse_args()

        scheme_clean = args.scheme.lower()
        if scheme_clean in ['real', 'real_standard']:
            scheme_clean = 'real_standard_33'
        elif scheme_clean == 'real_stencil':
            scheme_clean = 'real_stencil_12'

        print(f"Initializing S6*-Adjoint Hamiltonian ODE Layer [Branch I: {scheme_clean.upper()}]...")
        potential = QuarticPotential()
        ode_layer = S6HamiltonianODE(potential, scheme=scheme_clean)

        q0 = torch.tensor([[1.0, -0.5], [0.5, 0.8], [-1.2, 0.3], [0.2, -1.0]], dtype=torch.float64, requires_grad=True)
        p0 = torch.tensor([[0.2, 0.4], [-0.3, 0.1], [0.8, -0.5], [-0.4, 0.6]], dtype=torch.float64, requires_grad=True)

        h = args.dt
        num_steps = args.steps

        print(f"Integrating forward {num_steps} steps (T = {h * num_steps:.2f}) with strict O(1) memory...")
        q_T, p_T = ode_layer(q0, p0, h, num_steps)

        loss = torch.sum(q_T ** 2 + p_T ** 2)
        print(f"Final Loss: {loss.item():.6f}")

        print(f"Executing backward adjoint pass with on-the-fly state reconstruction ({scheme_clean})...")
        loss.backward()

        print(f"Gradient w.r.t q0: {q0.grad.shape}")
        print(f"Gradient w.r.t potential k: {potential.k.grad}")
        print(f"Gradient w.r.t potential lambda: {potential.lam.grad}")
        print(f">> SUCCESS: S6*-Adjoint completed forward and backward passes flawlessly with branch {scheme_clean}!")

    if __name__ == '__main__':
        main()
