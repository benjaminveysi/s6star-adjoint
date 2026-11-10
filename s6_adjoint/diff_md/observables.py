"""
Thermodynamic and Structural Observables for Differentiable Molecular Dynamics.
Includes:
1. Exact Energy and Temperature evaluations.
2. Fully Differentiable Radial Distribution Function g(r) via Gaussian Kernel Density Estimation (KDE).
3. Fully Differentiable Virial Pressure via Clausius Virial Theorem.
4. Mean Squared Displacement (MSD) for diffusion coefficients.
"""
from typing import Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_kinetic_energy(p_flat: "torch.Tensor", mass: float = 1.0) -> "torch.Tensor":
    """E_kin = 0.5 * sum_i ||p_i||^2 / m"""
    if p_flat.ndim == 2:
        return 0.5 * torch.sum(p_flat ** 2, dim=-1) / mass
    return 0.5 * torch.sum(p_flat ** 2) / mass


def compute_total_energy(
    q_flat: "torch.Tensor",
    p_flat: "torch.Tensor",
    potential_module: "torch.nn.Module",
    mass: float = 1.0
) -> "torch.Tensor":
    """H(q, p) = E_kin(p) + V(q)"""
    return compute_kinetic_energy(p_flat, mass) + potential_module(q_flat)


def compute_temperature(
    p_flat: "torch.Tensor",
    mass: float = 1.0,
    k_B: float = 1.0
) -> "torch.Tensor":
    """T = (2 * E_kin) / (3 * M * k_B)"""
    num_coords = p_flat.shape[-1]
    dof = float(num_coords)
    e_kin = compute_kinetic_energy(p_flat, mass)
    return (2.0 * e_kin) / (dof * k_B)


def compute_mean_squared_displacement(q_trajectory: "torch.Tensor") -> "torch.Tensor":
    """
    Computes MSD(t) = < ||q(t) - q(0)||^2 > across all particles.
    q_trajectory: (num_steps, 3 * M) or (num_steps, B, 3 * M)
    """
    q0 = q_trajectory[0]
    disp = q_trajectory - q0
    if disp.ndim == 3:  # (steps, B, coords)
        msd = torch.mean(torch.sum(disp ** 2, dim=-1), dim=-1)
    else:               # (steps, coords)
        msd = torch.mean(torch.sum(disp ** 2, dim=-1))
    return msd


def compute_differentiable_rdf(
    q_flat: "torch.Tensor",
    r_bins: "torch.Tensor",
    box_length: float,
    sigma: float = 0.08
) -> "torch.Tensor":
    """
    Differentiable Radial Distribution Function g(r) using Gaussian Kernel Density Estimation.
    Fully vectorized over batch dimension B (independent parallel replicas).
    
    Args:
        q_flat: Tensor of shape (B, 3 * M) or (3 * M,) with particle positions.
        r_bins: Tensor of shape (K,) with radial distance evaluation points (e.g. linspace(0.5, 4.0, 50)).
        box_length: Side length of the cubic simulation box L (Volume V = L^3).
        sigma: Gaussian kernel width for smooth continuous density projection (in distance units).
        
    Returns:
        g_r: Differentiable Tensor of shape (K,) containing the ensemble-averaged g(r).
    """
    if q_flat.ndim == 1:
        q = q_flat.unsqueeze(0)
    else:
        q = q_flat

    B, num_coords = q.shape
    M = num_coords // 3
    q_3d = q.view(B, M, 3)

    # Compute pairwise difference vectors: (B, M, M, 3)
    diff = q_3d.unsqueeze(2) - q_3d.unsqueeze(1)
    
    # Pairwise distances with safe numerical epsilon for gradient stability: (B, M, M)
    dist_sq = torch.sum(diff ** 2, dim=-1)
    dist = torch.sqrt(dist_sq + 1e-12)

    # Extract upper triangular pairs to avoid self-interaction and double counting: (B, num_pairs)
    triu_indices = torch.triu_indices(M, M, offset=1, device=q.device)
    pair_dists = dist[:, triu_indices[0], triu_indices[1]] # (B, P) where P = M * (M - 1) / 2
    P = pair_dists.shape[1]

    # Gaussian kernel evaluation against radial grid:
    # pair_dists: (B, P, 1) | r_bins: (1, 1, K) -> (B, P, K)
    r_bins_3d = r_bins.view(1, 1, -1)
    dists_3d = pair_dists.unsqueeze(-1)
    
    # Kernel: (1 / sqrt(2 * pi * sigma^2)) * exp(- (r_ij - r_k)^2 / (2 * sigma^2))
    norm_factor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    kernel_vals = norm_factor * torch.exp(-0.5 * ((dists_3d - r_bins_3d) / sigma) ** 2)

    # Sum kernel contributions over all pairs and average across batch B: (K,)
    # Total pair count density = sum_{i < j} kernel
    pair_density = torch.mean(torch.sum(kernel_vals, dim=1), dim=0) # (K,)

    # Normalization by ideal gas spherical shell volume:
    # V_shell(r) = 4 * pi * r^2
    # Ideal pair density for M particles in volume V = L^3 is: 0.5 * M * rho * 4 * pi * r^2
    # where rho = M / V.
    volume = float(box_length ** 3)
    rho = float(M) / volume
    shell_ideal = 0.5 * float(M) * rho * 4.0 * np.pi * (r_bins ** 2)

    g_r = pair_density / (shell_ideal + 1e-12)
    return g_r


def compute_differentiable_virial_pressure(
    q_flat: "torch.Tensor",
    p_flat: "torch.Tensor",
    forces_flat: "torch.Tensor",
    box_length: float,
    mass: float = 1.0
) -> "torch.Tensor":
    """
    Computes Differentiable Virial Pressure P = (2 * E_kin + Virial) / (3 * V)
    Averaged across the ensemble batch B.
    
    Args:
        q_flat: Positions (B, 3 * M) or (3 * M,)
        p_flat: Momenta (B, 3 * M) or (3 * M,)
        forces_flat: Forces (B, 3 * M) or (3 * M,)
        box_length: Cubic box length L (Volume V = L^3)
        mass: Particle mass (default 1.0)
        
    Returns:
        P: Differentiable scalar Tensor representing ensemble pressure.
    """
    if q_flat.ndim == 1:
        q = q_flat.unsqueeze(0)
        p = p_flat.unsqueeze(0)
        f = forces_flat.unsqueeze(0)
    else:
        q = q_flat
        p = p_flat
        f = forces_flat

    volume = float(box_length ** 3)
    
    # Kinetic energy per replica: (B,)
    e_kin = 0.5 * torch.sum(p ** 2, dim=-1) / mass
    
    # Internal virial per replica: sum_i q_i . F_i: (B,)
    virial = torch.sum(q * f, dim=-1)
    
    # Instantaneous pressure per replica: P_b = (2 * E_kin + Virial) / (3 * V)
    pressure_b = (2.0 * e_kin + virial) / (3.0 * volume)
    
    # Ensemble average pressure
    return torch.mean(pressure_b)


def compute_radial_distribution_function(
    q_flat: "torch.Tensor",
    r_max: float = 4.0,
    num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete binning RDF (evaluation only, NumPy backend).
    """
    coords = q_flat.detach().cpu().numpy()
    if coords.ndim == 2:
        coords = coords[0]  # Take first batch sample
    M = coords.shape[0] // 3
    q = coords.reshape(M, 3)

    diff = q[:, None, :] - q[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    i_idx, j_idx = np.triu_indices(M, k=1)
    pair_distances = r[i_idx, j_idx]

    counts, bin_edges = np.histogram(pair_distances, bins=num_bins, range=(0.1, r_max))
    r_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dr = bin_edges[1] - bin_edges[0]

    shell_volumes = 4.0 * np.pi * (r_centers ** 2) * dr
    density = float(M) / ((2.0 * r_max) ** 3)
    g_r = counts / (0.5 * M * density * shell_volumes + 1e-12)

    return r_centers, g_r


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
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for generate_ensemble_configurations.")

    if dtype is None:
        dtype = torch.float64

    if box_length is None:
        volume = float(num_particles) / float(density)
        L = float(volume ** (1.0 / 3.0))
    else:
        L = float(box_length)

    n_per_dim = int(np.ceil(num_particles ** (1.0 / 3.0)))
    spacing = L / float(n_per_dim)
    
    grid_coords = []
    offset = -0.5 * L + 0.5 * spacing
    for ix in range(n_per_dim):
        for iy in range(n_per_dim):
            for iz in range(n_per_dim):
                if len(grid_coords) < num_particles:
                    grid_coords.append([offset + ix * spacing, offset + iy * spacing, offset + iz * spacing])
    
    base_pos = torch.tensor(grid_coords, dtype=dtype, device=device)

    q_replicas = []
    p_replicas = []
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    vel_sigma = np.sqrt(k_B * target_temperature / mass)

    for b in range(num_replicas):
        jitter = (torch.rand((num_particles, 3), generator=gen, dtype=dtype, device=device) - 0.5) * (0.1 * spacing)
        qb = base_pos + jitter
        qb = qb - L * torch.round(qb / L)
        
        vb = torch.randn((num_particles, 3), generator=gen, dtype=dtype, device=device) * vel_sigma
        v_com = torch.mean(vb, dim=0, keepdim=True)
        vb = vb - v_com
        
        dof = 3.0 * (num_particles - 1.0)
        curr_temp = torch.sum(vb ** 2) * mass / (dof * k_B)
        scale_factor = torch.sqrt(torch.tensor(target_temperature, dtype=dtype, device=device) / (curr_temp + 1e-12))
        vb = vb * scale_factor
        
        pb = vb * mass
        q_replicas.append(qb.flatten())
        p_replicas.append(pb.flatten())

    q_init = torch.stack(q_replicas, dim=0)
    p_init = torch.stack(p_replicas, dim=0)

    return q_init, p_init, L
