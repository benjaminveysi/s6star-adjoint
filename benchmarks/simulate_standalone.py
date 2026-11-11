#!/usr/bin/env python3
"""
================================================================================
STANDALONE MOLECULAR DYNAMICS SIMULATOR (CLI)
Loads a pre-trained S6* Neural Force Field (.pt) and simulates ANY molecular system
Outputs full multi-frame .xyz trajectory for 3D visualization.
================================================================================
"""
import os
import sys
import time
import argparse
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required.")
    sys.exit(1)

from s6_adjoint import (
    S6HamiltonianODE,
    SchNetForceField,
    compute_temperature,
    compute_total_energy,
    write_xyz_trajectory
)


def create_system(system_name="argon_cluster", num_atoms=16, spacing=1.30, dtype=torch.float64, device='cpu'):
    """Creates initial coordinates and species for the specified system."""
    side = int(np.ceil(num_atoms ** (1/3)))
    coords = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                if len(coords) < num_atoms:
                    coords.append([x * spacing, y * spacing, z * spacing])
    coords = np.array(coords[:num_atoms])
    coords -= np.mean(coords, axis=0)

    np.random.seed(999)
    p_raw = np.random.randn(num_atoms, 3) * 0.20
    p_raw -= np.mean(p_raw, axis=0)

    q0 = torch.tensor(coords.flatten(), dtype=dtype, device=device)
    p0 = torch.tensor(p_raw.flatten(), dtype=dtype, device=device)
    species = [0] * num_atoms  # Default species
    return q0, p0, species


def run_standalone_simulation(
    checkpoint_path: str,
    output_xyz: str,
    num_steps: int = 200,
    dt: float = 0.005,
    num_atoms: int = 16,
    device_str: str = 'cuda',
    scheme: str = 'real_standard_33'
):
    if device_str == 'cuda' and not torch.cuda.is_available():
        device_str = 'cpu'
    device = torch.device(device_str)
    dtype = torch.float64

    print("=" * 80)
    print(f">> Standalone Molecular Dynamics Engine: S6*-Adjoint Loaded Model")
    print(f">> Device: {device} | Dtype: {dtype}")
    print(f">> Loading Checkpoint: {checkpoint_path}")
    print("=" * 80)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first with: python benchmarks/train_general_mlff.py")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['model_config']
    print(f">> Loaded Model Config: {cfg}")

    # Reconstruct general model
    model = SchNetForceField(
        num_species=cfg['num_species'],
        hidden_dim=cfg['hidden_dim'],
        num_interactions=cfg['num_interactions'],
        num_rbf=cfg['num_rbf'],
        cutoff=cfg['cutoff'],
        default_species=cfg['default_species'],
        dtype=dtype
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Build S6* High-Order Symplectic Simulator
    simulator = S6HamiltonianODE(model, scheme=scheme).to(device)

    # Setup atomic system
    q, p, species = create_system("cluster", num_atoms=num_atoms, spacing=1.30, dtype=dtype, device=device)
    M = num_atoms

    print(f"\n>> Simulating {M} atoms for {num_steps} macro-steps (dt = {dt} fs)...")
    print(f"{'Step':<10} | {'Kinetic Temp':<16} | {'Total Energy':<16} | {'Max Coord |q|':<14}")
    print("-" * 65)

    coords_history = [q.detach().cpu().numpy().reshape(M, 3)]
    temp_history = []
    energy_history = []

    # Run step-by-step to record frames and diagnose trajectory
    start_sim = time.time()
    chunk_size = 10
    total_chunks = num_steps // chunk_size

    with torch.no_grad():
        for ch in range(total_chunks):
            # Advance chunk_size steps with S6*
            q, p = simulator(q.unsqueeze(0), p.unsqueeze(0), dt, chunk_size)
            q = q.squeeze(0)
            p = p.squeeze(0)

            coords_history.append(q.detach().cpu().numpy().reshape(M, 3))
            
            temp = compute_temperature(p).item()
            energy = compute_total_energy(q, p, model).item()
            max_q = torch.max(torch.abs(q)).item()

            temp_history.append(temp)
            energy_history.append(energy)

            curr_step = (ch + 1) * chunk_size
            if curr_step % 20 == 0 or curr_step == chunk_size:
                print(f"{curr_step:<10} | {temp:>14.4f} | {energy:>14.4f} | {max_q:>12.4f}")

    sim_elapsed = time.time() - start_sim
    print("-" * 65)
    print(f">> Simulation completed in {sim_elapsed:.2f} s ({num_steps / sim_elapsed:.1f} steps/second)")

    # Write multi-frame XYZ trajectory
    comments = [f"Step {idx * chunk_size}, Time = {idx * chunk_size * dt:.4f} ps" for idx in range(len(coords_history))]
    write_xyz_trajectory(output_xyz, coords_history, species, comments)
    print(f">> XYZ Trajectory saved successfully to: {output_xyz}")
    print(f"   (Open this file in Ovito, VMD, or view in Colab!)")

    # Generate diagnostic plot
    fig_path = output_xyz.replace(".xyz", "_diagnostics.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=130)
    steps_arr = np.arange(1, len(temp_history) + 1) * chunk_size

    ax1.plot(steps_arr, temp_history, color='#D32F2F', lw=2)
    ax1.set_xlabel('Simulation Step', fontweight='bold')
    ax1.set_ylabel('Temperature (Kinetic)', fontweight='bold')
    ax1.set_title('Thermal Evolution Profile', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)

    e0 = energy_history[0]
    ax2.plot(steps_arr, np.array(energy_history) - e0, color='#1976D2', lw=2)
    ax2.set_xlabel('Simulation Step', fontweight='bold')
    ax2.set_ylabel('Total Energy Drift ΔH', fontweight='bold')
    ax2.set_title(r'S6* Symplectic Conservation ($\Delta H$)', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f">> Diagnostics figure saved to: {fig_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone MD Simulation CLI")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/general_mlff_checkpoint.pt")
    parser.add_argument('--output', type=str, default="trajectories/standalone_simulation.xyz")
    parser.add_argument('--num_steps', type=int, default=150)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--num_atoms', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--scheme', type=str, default='real_standard_33', choices=['real_standard_33', 'real_stencil_12', 'real', 'real_standard', 'real_stencil'])
    args = parser.parse_args()

    run_standalone_simulation(
        checkpoint_path=args.checkpoint,
        output_xyz=args.output,
        num_steps=args.num_steps,
        dt=args.dt,
        num_atoms=args.num_atoms,
        device_str=args.device,
        scheme=args.scheme
    )
