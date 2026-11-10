"""
Ensemble Initializer for Independent Parallel Replicas in Differentiable Molecular Dynamics.
Generates statistically decorrelated, thermalized configurations across batch dimension B.
"""
from typing import Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def generate_ensemble_configurations(
    num_replicas: int,
    num_particles: int,
    target_temperature: float = 1.0,
    density: float = 0.84,
    box_length: float = None,
    mass: float = 1.0,
    k_B: float = 1.0,
    seed: int = 42,
    dtype=None,
    device="cpu"
) -> Tuple["torch.Tensor", "torch.Tensor", float]:
    """
    Generates B independent, thermalized configurations for parallel ensemble simulations.
    
    Args:
        num_replicas: Number of parallel simulation boxes B.
        num_particles: Number of atoms N in each box.
        target_temperature: Desired equilibrium kinetic temperature.
        density: Number density rho = N / V (default 0.84, liquid argon state).
        box_length: Optional explicit cubic box length L. If None, L = (N / rho)^(1/3).
        mass: Particle mass (default 1.0).
        k_B: Boltzmann constant (default 1.0).
        seed: Base random seed for reproducibility across replicas.
        dtype: PyTorch tensor dtype (default torch.float64).
        device: PyTorch device ('cpu' or 'cuda').
        
    Returns:
        q_init: Tensor of shape (B, 3 * N) with particle positions.
        p_init: Tensor of shape (B, 3 * N) with Maxwell-Boltzmann momenta (zero COM momentum).
        box_length: The computed or specified cubic box length L.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for generate_ensemble_configurations.")

    if dtype is None:
        dtype = torch.float64

    # Determine cubic box length
    if box_length is None:
        volume = float(num_particles) / float(density)
        L = float(volume ** (1.0 / 3.0))
    else:
        L = float(box_length)

    # 1. Generate base lattice positions inside [-L/2, L/2]
    # Simple cubic grid arrangement
    n_per_dim = int(np.ceil(num_particles ** (1.0 / 3.0)))
    spacing = L / float(n_per_dim)
    
    grid_coords = []
    offset = -0.5 * L + 0.5 * spacing
    for ix in range(n_per_dim):
        for iy in range(n_per_dim):
            for iz in range(n_per_dim):
                if len(grid_coords) < num_particles:
                    grid_coords.append([offset + ix * spacing, offset + iy * spacing, offset + iz * spacing])
    
    base_pos = torch.tensor(grid_coords, dtype=dtype, device=device) # (N, 3)

    # 2. Build B independent replicas with small decorrelation jitter and Maxwell-Boltzmann velocities
    q_replicas = []
    p_replicas = []
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    vel_sigma = np.sqrt(k_B * target_temperature / mass)

    for b in range(num_replicas):
        # Independent jitter: 5% of lattice spacing to break initial crystalline symmetry
        jitter = (torch.rand((num_particles, 3), generator=gen, dtype=dtype, device=device) - 0.5) * (0.1 * spacing)
        qb = base_pos + jitter
        
        # Periodic wrap inside [-L/2, L/2]
        qb = qb - L * torch.round(qb / L)
        
        # Sample Maxwell-Boltzmann velocities
        vb = torch.randn((num_particles, 3), generator=gen, dtype=dtype, device=device) * vel_sigma
        
        # Zero out center-of-mass velocity: sum_i v_i = 0
        v_com = torch.mean(vb, dim=0, keepdim=True)
        vb = vb - v_com
        
        # Rescale velocities to match target kinetic temperature exactly:
        # T_curr = sum(v^2) * mass / (3 * (N - 1) * k_B)
        dof = 3.0 * (num_particles - 1.0)
        curr_temp = torch.sum(vb ** 2) * mass / (dof * k_B)
        scale_factor = torch.sqrt(torch.tensor(target_temperature, dtype=dtype, device=device) / (curr_temp + 1e-12))
        vb = vb * scale_factor
        
        pb = vb * mass
        q_replicas.append(qb.flatten())
        p_replicas.append(pb.flatten())

    q_init = torch.stack(q_replicas, dim=0) # (B, 3 * N)
    p_init = torch.stack(p_replicas, dim=0) # (B, 3 * N)

    return q_init, p_init, L
