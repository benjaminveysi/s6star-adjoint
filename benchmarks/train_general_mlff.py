#!/usr/bin/env python3
"""
================================================================================
GENERAL MLFF: Training a Universal Neural Force Field with S6*-Adjoint
================================================================================

Objective:
    Trains an E(3)-invariant Continuous-Filter GNN Force Field (SchNet)
    directly on multi-particle 3D molecular dynamics trajectories.
    Saves an exportable PyTorch model checkpoint (.pt) for standalone deployment.
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
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required for training.")
    sys.exit(1)

from s6_adjoint import (
    S6HamiltonianODE,
    SchNetForceField,
    LennardJonesPotential,
    compute_temperature,
    compute_total_energy,
    write_xyz_trajectory
)


def generate_initial_lattice(num_particles=16, spacing=1.30, dtype=torch.float64, device='cpu'):
    side = int(np.ceil(num_particles ** (1/3)))
    coords = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                if len(coords) < num_particles:
                    coords.append([x * spacing, y * spacing, z * spacing])
    coords = np.array(coords[:num_particles])
    coords -= np.mean(coords, axis=0)
    
    np.random.seed(42)
    p_raw = np.random.randn(num_particles, 3) * 0.22
    p_raw -= np.mean(p_raw, axis=0)
    
    q0 = torch.tensor(coords.flatten(), dtype=dtype, device=device)
    p0 = torch.tensor(p_raw.flatten(), dtype=dtype, device=device)
    return q0, p0


def main(device_str='cuda', num_particles=16, num_steps=30, epochs=35, dt=0.005):
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable, switching to CPU.")
        device_str = 'cpu'

    device = torch.device(device_str)
    dtype = torch.float64
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    print("=" * 80)
    print(f">> Training General GNN Force Field (SchNet) via S6*-Adjoint")
    print(f">> Device: {device} ({gpu_name}) | Dtype: {dtype}")
    print(f">> Particles: {num_particles} atoms | Rollout Horizon: {num_steps} steps (dt = {dt})")
    print("=" * 80)

    # 1. Target Physical System (Ground Truth)
    true_potential = LennardJonesPotential(init_epsilon=1.00, init_sigma=1.00, cutoff=3.5, delta=0.25).to(device)
    true_sim = S6HamiltonianODE(true_potential, scheme='real').to(device)

    # Batch of initial states (Batch size = 4)
    batch_size = 4
    q0_list, p0_list = [], []
    for b in range(batch_size):
        q_b, p_b = generate_initial_lattice(num_particles, spacing=1.28 + b*0.02, dtype=dtype, device=device)
        q0_list.append(q_b)
        p0_list.append(p_b)
    q0_batch = torch.stack(q0_list)
    p0_batch = torch.stack(p0_list)

    print(f"\n[1/3] Generating Ground Truth Targets ({num_steps} steps)...")
    with torch.no_grad():
        q_target, p_target = true_sim(q0_batch, p0_batch, dt, num_steps)
        t_target = compute_temperature(p_target).mean().item()
        print(f"      Target State: Temperature={t_target:.4f}, Mean Norm={q_target.norm().item():.3f}")

    # 2. General SchNet GNN Model
    model = SchNetForceField(
        num_species=5,
        hidden_dim=48,
        num_interactions=2,
        num_rbf=16,
        cutoff=4.0,
        default_species=0,
        dtype=dtype
    ).to(device)

    # Zero initialization for readout to start with gentle forces
    for m in model.readout.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    sim = S6HamiltonianODE(model, scheme='real').to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[2/3] General SchNet Model Initialized: {num_params} trainable parameters.")
    print(f"      Starting Trajectory Optimization...")
    print("-" * 80)
    print(f"{'Epoch':<8} | {'Loss (MSE)':<16} | {'Mean Temp':<14} | {'Peak VRAM (MiB)':<16}")
    print("-" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.007, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    losses = []
    vrams = []
    start_t = time.time()

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        q_pred, p_pred = sim(q0_batch, p0_batch, dt, num_steps)

        loss_q = torch.mean((q_pred - q_target)**2)
        loss_p = torch.mean((p_pred - p_target)**2)
        loss = loss_q + 0.3 * loss_p

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        scheduler.step()

        curr_temp = compute_temperature(p_pred).mean().item()
        peak_vram = (torch.cuda.max_memory_allocated() / (1024**2)) if device.type == 'cuda' else 1.2

        losses.append(loss.item())
        vrams.append(peak_vram)

        if ep % 5 == 0 or ep == 1:
            print(f"{ep:<8} | {loss.item():<16.6e} | {curr_temp:>12.4f} | {peak_vram:>14.2f} MiB")

    elapsed = time.time() - start_t
    print("-" * 80)
    print(f">> Training completed in {elapsed:.2f} s ({elapsed/epochs:.2f} s/epoch)")

    # 3. Save Standalone Deployment Checkpoint
    checkpoint_dir = os.path.join(repo_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "general_mlff_checkpoint.pt")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_species': model.num_species,
            'hidden_dim': model.hidden_dim,
            'num_interactions': len(model.interactions),
            'num_rbf': model.num_rbf,
            'cutoff': model.cutoff,
            'default_species': model.default_species
        },
        'training_meta': {
            'final_loss': losses[-1],
            'epochs': epochs,
            'num_particles': num_particles,
            'dt': dt
        }
    }
    torch.save(checkpoint, ckpt_path)
    print(f">> [3/3] Model Checkpoint successfully saved for standalone deployment: {ckpt_path}")

    # Plot convergence
    fig_dir = os.path.join(repo_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "general_mlff_training.png")
    plt.figure(figsize=(7, 4), dpi=140)
    plt.semilogy(losses, 'o-', color='#1976D2', lw=2.2, ms=4)
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Trajectory Loss (MSE)', fontsize=11, fontweight='bold')
    plt.title('General SchNet Force Field Training Convergence', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f">> Training curve saved to: {fig_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_particles', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--dt', type=float, default=0.005)
    args = parser.parse_args()
    main(args.device, args.num_particles, args.num_steps, args.epochs, args.dt)
