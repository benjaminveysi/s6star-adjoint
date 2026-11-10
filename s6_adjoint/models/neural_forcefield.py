"""
SchNetForceField: General E(3)-Invariant Continuous-Filter Convolutional Network (GNN).
Produces conservative atomic forces via exact autograd: F = -nabla_q V(q, Z).
Supports arbitrary molecules, multi-element species, and variable atom counts M.
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CFConv(nn.Module):
        """Continuous-Filter Convolution layer."""
        def __init__(self, num_rbf: int, hidden_dim: int, dtype=torch.float64):
            super().__init__()
            self.filter_net = nn.Sequential(
                nn.Linear(num_rbf, hidden_dim, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
            )

        def forward(self, h, rbf, f_cut, i_idx, j_idx, is_batched: bool):
            # h: (B, M, hidden_dim) or (M, hidden_dim)
            # rbf: (B, num_pairs, num_rbf) or (num_pairs, num_rbf)
            # f_cut: (B, num_pairs) or (num_pairs,)
            cfconv_weights = self.filter_net(rbf) * f_cut.unsqueeze(-1)
            if is_batched:
                B, M, H = h.shape
                # messages: (B, num_pairs, H)
                messages = h[:, j_idx, :] * cfconv_weights
                aggregated = torch.zeros(B, M, H, device=h.device, dtype=h.dtype)
                aggregated.index_add_(1, i_idx, messages)
                return aggregated
            else:
                M, H = h.shape
                messages = h[j_idx, :] * cfconv_weights
                aggregated = torch.zeros(M, H, device=h.device, dtype=h.dtype)
                aggregated.index_add_(0, i_idx, messages)
                return aggregated


    class SchNetInteractionBlock(nn.Module):
        """Interaction block with continuous-filter convolution and residual update."""
        def __init__(self, num_rbf: int, hidden_dim: int, dtype=torch.float64):
            super().__init__()
            self.cfconv = CFConv(num_rbf, hidden_dim, dtype=dtype)
            self.dense1 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
            self.dense2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)

        def forward(self, h, rbf, f_cut, i_idx, j_idx, is_batched: bool):
            v = self.cfconv(h, rbf, f_cut, i_idx, j_idx, is_batched)
            v = F.silu(self.dense1(v))
            v = self.dense2(v)
            return h + v


    class SchNetForceField(nn.Module):
        """
        General Molecular Machine Learning Force Field (MLFF) based on SchNet:
        - Input: Flat coordinates q (batch_size, 3 * M), optional species Z (M,)
        - Output: Conservative potential energy V(q) in scalar Float64.
        """
        def __init__(
            self,
            num_species: int = 10,
            hidden_dim: int = 48,
            num_interactions: int = 2,
            num_rbf: int = 16,
            cutoff: float = 4.0,
            default_species: int = 0,
            dtype=torch.float64
        ):
            super().__init__()
            self.cutoff = float(cutoff)
            self.num_rbf = int(num_rbf)
            self.hidden_dim = int(hidden_dim)
            self.num_species = int(num_species)
            self.default_species = int(default_species)

            # Atomic species embeddings
            self.embedding = nn.Embedding(num_species, hidden_dim, dtype=dtype)

            # RBF expansion parameters
            centers = torch.linspace(0.6, cutoff, num_rbf, dtype=dtype)
            gamma = 0.5 / ((cutoff - 0.6) / num_rbf) ** 2
            self.register_buffer('centers', centers)
            self.register_buffer('gamma', torch.tensor(gamma, dtype=dtype))

            # Optional fixed species buffer for specific molecular systems
            self.register_buffer('fixed_species', None)

            # Stacked interaction blocks
            self.interactions = nn.ModuleList([
                SchNetInteractionBlock(num_rbf, hidden_dim, dtype=dtype)
                for _ in range(num_interactions)
            ])

            # Atomic energy readout layer
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1, dtype=dtype)
            )

        def set_atomic_species(self, species_tensor: torch.Tensor):
            """Registers the atomic species IDs for the current molecule."""
            self.fixed_species = species_tensor.long()

        def forward(self, q_flat: torch.Tensor) -> torch.Tensor:
            is_batched = (q_flat.ndim == 2)
            if is_batched:
                B, num_coords = q_flat.shape
                M = num_coords // 3
                q = q_flat.view(B, M, 3)
            else:
                num_coords = q_flat.shape[0]
                M = num_coords // 3
                q = q_flat.view(M, 3)

            device = q.device
            # Default to atomic species 0 if not explicitly registered
            if self.fixed_species is None or len(self.fixed_species) != M:
                species = torch.full((M,), self.default_species, dtype=torch.long, device=device)
            else:
                species = self.fixed_species.to(device)

            # Initial embedding: (M, hidden_dim)
            h = self.embedding(species)
            if is_batched:
                h = h.unsqueeze(0).expand(B, -1, -1)

            # Pairwise directed graph indices (all i != j)
            i_idx, j_idx = torch.where(~torch.eye(M, device=device, dtype=torch.bool))

            if is_batched:
                diff = q[:, i_idx, :] - q[:, j_idx, :]
                r2 = torch.sum(diff ** 2, dim=-1)
            else:
                diff = q[i_idx, :] - q[j_idx, :]
                r2 = torch.sum(diff ** 2, dim=-1)

            r = torch.sqrt(r2 + 1e-12)

            # RBF expansion and smooth cosine cutoff envelope
            r_expand = r.unsqueeze(-1)
            rbf = torch.exp(-self.gamma * ((r_expand - self.centers) ** 2))
            cutoff_mask = (r < self.cutoff)
            f_cut = torch.where(
                cutoff_mask,
                0.5 * (torch.cos(torch.pi * r / self.cutoff) + 1.0),
                torch.zeros_like(r)
            )

            # Message passing iterations
            for interaction in self.interactions:
                h = interaction(h, rbf, f_cut, i_idx, j_idx, is_batched)

            # Atomic energies and total potential energy
            atomic_energies = self.readout(h).squeeze(-1) # (B, M) or (M,)
            if is_batched:
                return torch.sum(atomic_energies, dim=-1)
            else:
                return torch.sum(atomic_energies)

        def eval_pair_potential(self, r_grid: torch.Tensor, z1: int = 0, z2: int = 0) -> torch.Tensor:
            """Evaluates 1D pair energy curve for distance grid r between species z1 and z2."""
            device = r_grid.device
            # Construct a 2-atom system at distance r
            energies = []
            z_pair = torch.tensor([z1, z2], dtype=torch.long, device=device)
            self.set_atomic_species(z_pair)
            for dist in r_grid:
                q_pair = torch.tensor([[0.0, 0.0, 0.0], [float(dist), 0.0, 0.0]], dtype=torch.float64, device=device)
                energies.append(self.forward(q_pair.flatten()).item())
            return torch.tensor(energies, dtype=torch.float64, device=device)
