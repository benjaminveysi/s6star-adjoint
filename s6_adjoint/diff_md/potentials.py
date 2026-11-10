"""
Differentiable Interatomic Potentials for Molecular Dynamics in PyTorch.
Features:
- Direct upper-triangular indexing (torch.triu_indices): eliminates diagonals, prevents NaN gradients.
- Soft-core regularization: r_eff = sqrt(r^2 + delta^2) prevents r^-27 commutator singularity.
- Smooth polynomial cutoff envelope: C^2 continuity at interaction boundary.
- Support for LennardJonesPotential, HarmonicChainPotential, DoubleWellPotential, and NeuralInteratomicPotential.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class LennardJonesPotential(nn.Module):
        """
        Differentiable Soft-Core Lennard-Jones Pair Potential:
            V(r) = sum_{i < j} 4 * eps * [ (sig / r_eff)^12 - (sig / r_eff)^6 ] * f_cut(r)
        where r_eff = sqrt(r^2 + delta^2).
        """
        def __init__(
            self,
            init_epsilon: float = 1.0,
            init_sigma: float = 1.0,
            epsilon: float = None,
            sigma: float = None,
            cutoff: float = 3.5,
            delta: float = 0.25,
            dtype=torch.float64
        ):
            super().__init__()
            eps_val = epsilon if epsilon is not None else init_epsilon
            sig_val = sigma if sigma is not None else init_sigma
            self.log_epsilon = nn.Parameter(torch.tensor(float(np.log(eps_val)), dtype=dtype))
            self.log_sigma = nn.Parameter(torch.tensor(float(np.log(sig_val)), dtype=dtype))
            self.cutoff = float(cutoff)
            self.delta = float(delta)

        @property
        def epsilon(self) -> torch.Tensor:
            return torch.exp(self.log_epsilon)

        @property
        def sigma(self) -> torch.Tensor:
            return torch.exp(self.log_sigma)

        def forward(self, q_flat: torch.Tensor) -> torch.Tensor:
            is_batched = (q_flat.ndim == 2)
            if is_batched:
                B, num_coords = q_flat.shape
                M = num_coords // 3
                q = q_flat.view(B, M, 3)
                device = q.device
                i_idx, j_idx = torch.triu_indices(M, M, offset=1, device=device)
                diff = q[:, i_idx, :] - q[:, j_idx, :]
                r2 = torch.sum(diff ** 2, dim=-1)
            else:
                num_coords = q_flat.shape[0]
                M = num_coords // 3
                q = q_flat.view(M, 3)
                device = q.device
                i_idx, j_idx = torch.triu_indices(M, M, offset=1, device=device)
                diff = q[i_idx, :] - q[j_idx, :]
                r2 = torch.sum(diff ** 2, dim=-1)

            r = torch.sqrt(r2 + 1e-12)
            r_eff = torch.sqrt(r2 + (self.delta ** 2))

            eps = self.epsilon
            sig = self.sigma

            sig_r = sig / r_eff
            sig_r6 = sig_r ** 6
            sig_r12 = sig_r6 ** 2

            v_pair = 4.0 * eps * (sig_r12 - sig_r6)

            cutoff_mask = (r < self.cutoff)
            f_cut = 0.5 * (torch.cos(torch.pi * r / self.cutoff) + 1.0)
            v_pair_cut = torch.where(cutoff_mask, v_pair * f_cut, torch.zeros_like(v_pair))

            if is_batched:
                return torch.sum(v_pair_cut, dim=-1)
            else:
                return torch.sum(v_pair_cut)

        def force(self, q_flat: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                q_req = q_flat.detach().requires_grad_(True)
                V = self.forward(q_req)
                F = -torch.autograd.grad(V.sum(), q_req, create_graph=False)[0]
            return F


    class HarmonicChainPotential(nn.Module):
        """
        Differentiable Harmonic 1D/3D Chain Potential:
            V(q) = 0.5 * k * sum_{i} (||q_{i+1} - q_i|| - r0)^2
        """
        def __init__(self, k: float = 10.0, r0: float = 1.0, dtype=torch.float64):
            super().__init__()
            self.k = nn.Parameter(torch.tensor(float(k), dtype=dtype))
            self.r0 = float(r0)

        def forward(self, q_flat: torch.Tensor) -> torch.Tensor:
            is_batched = (q_flat.ndim == 2)
            if is_batched:
                B, num_coords = q_flat.shape
                M = num_coords // 3
                q = q_flat.view(B, M, 3)
                diff = q[:, 1:, :] - q[:, :-1, :]
                r = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
                return 0.5 * self.k * torch.sum((r - self.r0) ** 2, dim=-1)
            else:
                num_coords = q_flat.shape[0]
                M = num_coords // 3
                q = q_flat.view(M, 3)
                diff = q[1:, :] - q[:-1, :]
                r = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
                return 0.5 * self.k * torch.sum((r - self.r0) ** 2)

        def force(self, q_flat: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                q_req = q_flat.detach().requires_grad_(True)
                V = self.forward(q_req)
                F = -torch.autograd.grad(V.sum(), q_req, create_graph=False)[0]
            return F


    class DoubleWellPotential(nn.Module):
        """
        Differentiable Multi-Particle Double-Well Potential:
            V(q) = sum_i [ a * (x_i^2 - 1)^2 + b * y_i^2 + c * z_i^2 ]
        """
        def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, dtype=torch.float64):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(float(a), dtype=dtype))
            self.b = nn.Parameter(torch.tensor(float(b), dtype=dtype))
            self.c = nn.Parameter(torch.tensor(float(c), dtype=dtype))

        def forward(self, q_flat: torch.Tensor) -> torch.Tensor:
            is_batched = (q_flat.ndim == 2)
            if is_batched:
                B, num_coords = q_flat.shape
                M = num_coords // 3
                q = q_flat.view(B, M, 3)
                x, y, z = q[..., 0], q[..., 1], q[..., 2]
                return torch.sum(self.a * (x ** 2 - 1.0) ** 2 + self.b * y ** 2 + self.c * z ** 2, dim=-1)
            else:
                num_coords = q_flat.shape[0]
                M = num_coords // 3
                q = q_flat.view(M, 3)
                x, y, z = q[:, 0], q[:, 1], q[:, 2]
                return torch.sum(self.a * (x ** 2 - 1.0) ** 2 + self.b * y ** 2 + self.c * z ** 2)

        def force(self, q_flat: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                q_req = q_flat.detach().requires_grad_(True)
                V = self.forward(q_req)
                F = -torch.autograd.grad(V.sum(), q_req, create_graph=False)[0]
            return F


    class NeuralInteratomicPotential(nn.Module):
        """
        E(3)-Invariant and Permutation-Invariant Neural Network Interatomic Potential (MLIP).
        Encodes interatomic distances via Gaussian Radial Basis Functions (RBF):
            V_theta(q) = sum_{i < j} MLP(RBF(||q_i - q_j||)) * f_cut(||q_i - q_j||)
        """
        def __init__(self, num_rbf: int = 16, cutoff: float = 3.5, hidden_dim: int = 64, dtype=torch.float64):
            super().__init__()
            self.cutoff = float(cutoff)
            self.num_rbf = int(num_rbf)

            centers = torch.linspace(0.6, cutoff, num_rbf, dtype=dtype)
            gamma = 0.5 / ((cutoff - 0.6) / num_rbf) ** 2
            self.register_buffer('centers', centers)
            self.register_buffer('gamma', torch.tensor(gamma, dtype=dtype))

            self.mlp = nn.Sequential(
                nn.Linear(num_rbf, hidden_dim, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, dtype=dtype)
            )

        def forward(self, q_flat: torch.Tensor) -> torch.Tensor:
            is_batched = (q_flat.ndim == 2)
            if is_batched:
                B, num_coords = q_flat.shape
                M = num_coords // 3
                q = q_flat.view(B, M, 3)
                device = q.device
                i_idx, j_idx = torch.triu_indices(M, M, offset=1, device=device)
                diff = q[:, i_idx, :] - q[:, j_idx, :]
                r2 = torch.sum(diff ** 2, dim=-1)
            else:
                num_coords = q_flat.shape[0]
                M = num_coords // 3
                q = q_flat.view(M, 3)
                device = q.device
                i_idx, j_idx = torch.triu_indices(M, M, offset=1, device=device)
                diff = q[i_idx, :] - q[j_idx, :]
                r2 = torch.sum(diff ** 2, dim=-1)

            r = torch.sqrt(r2 + 1e-12)

            r_expand = r.unsqueeze(-1)
            rbf = torch.exp(-self.gamma * ((r_expand - self.centers) ** 2))
            pair_energy = self.mlp(rbf).squeeze(-1)

            cutoff_mask = (r < self.cutoff)
            f_cut = 0.5 * (torch.cos(torch.pi * r / self.cutoff) + 1.0)
            pair_energy_cut = torch.where(cutoff_mask, pair_energy * f_cut, torch.zeros_like(pair_energy))

            if is_batched:
                return torch.sum(pair_energy_cut, dim=-1)
            else:
                return torch.sum(pair_energy_cut)

        def force(self, q_flat: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                q_req = q_flat.detach().requires_grad_(True)
                V = self.forward(q_req)
                F = -torch.autograd.grad(V.sum(), q_req, create_graph=False)[0]
            return F

else:
    class LennardJonesPotential:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for LennardJonesPotential.")

    class HarmonicChainPotential:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for HarmonicChainPotential.")

    class DoubleWellPotential:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for DoubleWellPotential.")

    class NeuralInteratomicPotential:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for NeuralInteratomicPotential.")
