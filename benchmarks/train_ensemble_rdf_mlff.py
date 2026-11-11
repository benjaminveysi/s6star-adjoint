#!/usr/bin/env python3
"""
train_ensemble_rdf_mlff.py:
Scale-Optimized Production Training for GNN Force Field (SchNet) from Macroscopic Observables.
- Radial Distribution Function g(r) via Differentiable Gaussian KDE.
- Virial Pressure P via Differentiable Clausius Virial.
- ZERO FORCE LABELS REQUIRED.
- Powered by S6*-Adjoint on a production-scale independent replica ensemble (default: 8,192 atoms total, 160,320 replica-steps).
"""
import os
import sys
import time
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def build_parser():
    parser = argparse.ArgumentParser(description="Train GNN Force Field from RDF and Pressure via S6*-Adjoint Ensemble.")
    parser.add_argument("--num_replicas", type=int, default=32, help="Number of independent parallel replicas B (e.g. 32 or 64).")
    parser.add_argument("--num_particles", type=int, default=256, help="Number of atoms N per replica (production 8192-atom profile uses 256).")
    parser.add_argument("--num_steps", type=int, default=30, help="Simulation trajectory horizon T in macro-steps.")
    parser.add_argument("--dt", type=float, default=0.005, help="Macro time step size dt.")
    parser.add_argument("--epochs", type=int, default=167, help="Training optimization epochs (167 gives 160,320 replica-steps with the default 32x256 profile).")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate for Adam optimizer.")
    parser.add_argument("--density", type=float, default=0.84, help="Target number density rho = N / V.")
    parser.add_argument("--scheme", type=str, default="real_standard_33", choices=["real_standard_33", "real_stencil_12", "real_standard", "real_stencil", "real"], help="Adjoint solver branch: real_standard_33 or real_stencil_12")
    parser.add_argument("--temperature", type=float, default=1.0, help="Target kinetic temperature.")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"], help="Floating point precision (float64 or float32).")
    parser.add_argument("--device", type=str, default="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension of SchNet GNN.")
    parser.add_argument("--num_interactions", type=int, default=2, help="Number of interaction blocks.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/ensemble_mlff_checkpoint.pt")
    parser.add_argument("--figure_path", type=str, default="figures/ensemble_training_diagnostics.png")
    return parser


def run_ensemble_training():
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required to run train_ensemble_rdf_mlff.py.")
        sys.exit(1)

    args = build_parser().parse_args()
    if args.num_replicas * args.num_particles == 8192:
        print(">> 8192-atom production profile confirmed: "
              f"{args.num_replicas} replicas x {args.num_particles} atoms/replica.")
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    os.makedirs(os.path.dirname(args.checkpoint_path) if os.path.dirname(args.checkpoint_path) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.figure_path) if os.path.dirname(args.figure_path) else ".", exist_ok=True)

    total_atoms = args.num_replicas * args.num_particles
    total_effective_steps = args.num_replicas * args.num_steps * args.epochs

    print("=" * 85)
    print(">> S6*-Adjoint High-Throughput Parallel Ensemble: Bulk Thermodynamic MLFF Training")
    print(f">> Device: {device} | Precision: {args.dtype} ({dtype})")
    print(f">> Scale: {args.num_replicas} parallel boxes x {args.num_particles} atoms = {total_atoms} total atoms evolving concurrently")
    print(f">> Trajectory Horizon: {args.num_steps} macro-steps/epoch | Epochs: {args.epochs}")
    print(f">> Total Phase-Space Sampling: {total_effective_steps:,} effective macro-steps")
    print("=" * 85)

    from s6_adjoint.models.neural_forcefield import SchNetForceField
    from s6_adjoint.autograd.function import S6HamiltonianFunction
    from s6_adjoint.diff_md.observables import (
        compute_differentiable_rdf,
        compute_differentiable_virial_pressure,
        compute_temperature
    )
    from s6_adjoint.diff_md.ensemble_init import generate_ensemble_configurations

    # 1. Setup Simulation Domain and Radial Grid
    volume = float(args.num_particles) / float(args.density)
    box_length = float(volume ** (1.0 / 3.0))
    # Cutoff covers up to 48% of the box length to capture multiple solvation shells without boundary violation
    r_cutoff = min(4.0, box_length * 0.48)
    num_bins = 50 if args.num_particles >= 128 else 40
    r_bins = torch.linspace(0.8, r_cutoff, num_bins, dtype=dtype, device=device)

    print(f">> Simulation Box: L = {box_length:.4f} (Volume = {volume:.2f}) | Cutoff r_c = {r_cutoff:.3f}")
    print(f">> Radial Grid: {num_bins} bins spanning r in [{r_bins[0]:.2f}, {r_bins[-1]:.2f}] Å")

    # 2. Synthesize Ground Truth Target Observables (Reference Liquid Argon Fluid)
    print(f"[1/3] Synthesizing Ground Truth Target Observables ({args.num_replicas} x {args.num_particles} atoms)...")
    from s6_adjoint.diff_md.potentials import LennardJonesPotential
    ref_potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=r_cutoff, dtype=dtype).to(device)

    with torch.no_grad():
        q_ref, p_ref, _ = generate_ensemble_configurations(
            num_replicas=args.num_replicas,
            num_particles=args.num_particles,
            target_temperature=args.temperature,
            density=args.density,
            box_length=box_length,
            seed=101,
            dtype=dtype,
            device=device
        )
        ref_params = [p for p in ref_potential.parameters()]
        ref_pnames = [n for n, _ in ref_potential.named_parameters()]
        q_ref_T, p_ref_T = S6HamiltonianFunction.apply(
            q_ref, p_ref, args.dt, args.num_steps, args.scheme, ref_potential, ref_pnames, *ref_params
        )
        target_rdf = compute_differentiable_rdf(q_ref_T, r_bins, box_length, sigma=0.08).detach()
        ref_forces = ref_potential.force(q_ref_T)
        target_pressure = compute_differentiable_virial_pressure(q_ref_T, p_ref_T, ref_forces, box_length).detach()

    peak_idx = target_rdf.argmax().item()
    print(f"      Ground Truth Established:")
    print(f"      - First Solvation Peak: g(r) = {target_rdf[peak_idx].item():.4f} at r = {r_bins[peak_idx].item():.3f} Å")
    print(f"      - Target Virial Pressure: P = {target_pressure.item():.4f}")

    # 3. Initialize Trainable SchNet Model
    print(f"[2/3] Initializing Trainable SchNet Force Field ({args.hidden_dim} dim, {args.num_interactions} layers)...")
    model = SchNetForceField(
        num_species=5,
        hidden_dim=args.hidden_dim,
        num_interactions=args.num_interactions,
        num_rbf=16,
        cutoff=r_cutoff,
        default_species=0,
        dtype=dtype
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Trainable Parameters: {total_params:,}")
    print("      Optimization Metric: Differentiable RDF MSE + Virial Pressure (ZERO force labels)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    param_list = [p for name, p in model.named_parameters() if p.requires_grad]

    history_loss = []
    history_rdf_err = []
    history_p_err = []
    history_vram = []
    history_temp = []

    print("-" * 92)
    print(f"{'Epoch':<8} | {'Total Loss':<14} | {'RDF Error':<12} | {'P Error':<12} | {'Temp':<10} | {'Peak VRAM':<14}")
    print("-" * 92)

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        optimizer.zero_grad()

        # Generate fresh decorrelated ensemble initial states
        q0, p0, _ = generate_ensemble_configurations(
            num_replicas=args.num_replicas,
            num_particles=args.num_particles,
            target_temperature=args.temperature,
            density=args.density,
            box_length=box_length,
            seed=epoch * 19 + 101,
            dtype=dtype,
            device=device
        )

        # Forward integration through S6*-Adjoint (O(1) memory)
        q_T, p_T = S6HamiltonianFunction.apply(
            q0, p0, args.dt, args.num_steps, args.scheme, model, param_names, *param_list
        )

        # Compute Differentiable RDF across all B replicas
        pred_rdf = compute_differentiable_rdf(q_T, r_bins, box_length, sigma=0.08)
        rdf_loss = torch.mean((pred_rdf - target_rdf) ** 2)

        # Compute Differentiable Virial Pressure
        with torch.enable_grad():
            q_req = q_T.detach().requires_grad_(True)
            V_eval = model(q_req)
            pred_forces = -torch.autograd.grad(V_eval.sum(), q_req, create_graph=True)[0]

        pred_pressure = compute_differentiable_virial_pressure(q_T, p_T, pred_forces, box_length)
        pressure_loss = (pred_pressure - target_pressure) ** 2

        # Combined thermodynamic objective
        total_loss = rdf_loss + 0.1 * pressure_loss

        # Reverse-time Adjoint Backpropagation (O(1) Memory Pullback)
        total_loss.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        scheduler.step()

        temp_val = compute_temperature(p_T).mean().item()
        rdf_err = rdf_loss.item()
        p_err = torch.abs(pred_pressure - target_pressure).item()
        loss_val = total_loss.item()

        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_vram = 0.0

        history_loss.append(loss_val)
        history_rdf_err.append(rdf_err)
        history_p_err.append(p_err)
        history_vram.append(peak_vram)
        history_temp.append(temp_val)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            ep_time = time.time() - epoch_start
            print(f"{epoch:<8} | {loss_val:<14.6e} | {rdf_err:<12.5e} | {p_err:<12.5e} | {temp_val:<10.4f} | {peak_vram:>8.2f} MiB ({ep_time:.1f}s)")

    total_time = time.time() - start_train_time
    print("-" * 92)
    print(f">> Bulk Ensemble Training finished in {total_time:.2f} s ({total_time / args.epochs:.2f} s/epoch)")

    # 4. Save Production Checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_species": 5,
            "hidden_dim": args.hidden_dim,
            "num_interactions": args.num_interactions,
            "num_rbf": 16,
            "cutoff": r_cutoff,
            "default_species": 0,
            "dtype": args.dtype
        },
        "training_meta": {
            "num_replicas": args.num_replicas,
            "num_particles": args.num_particles,
            "total_atoms": total_atoms,
            "total_effective_steps": total_effective_steps,
            "density": args.density,
            "temperature": args.temperature,
            "box_length": box_length,
            "dt": args.dt,
            "final_loss": history_loss[-1],
            "epochs": args.epochs,
            "history_loss": history_loss,
            "history_rdf_err": history_rdf_err,
            "history_p_err": history_p_err,
            "target_rdf": target_rdf.cpu().numpy(),
            "target_pressure": target_pressure.item(),
            "r_bins": r_bins.cpu().numpy()
        }
    }
    torch.save(checkpoint, args.checkpoint_path)
    print(f">> Production Checkpoint saved to: {args.checkpoint_path}")

    # 5. Generate Publication-Quality Diagnostics Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor("#0b0f19")
    for ax in axs.flat:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#374151")

    # Panel 1: Loss Convergence
    axs[0, 0].semilogy(range(1, args.epochs + 1), history_loss, color="#60a5fa", lw=2.2, label="Total Loss")
    axs[0, 0].semilogy(range(1, args.epochs + 1), history_rdf_err, color="#34d399", lw=1.8, ls="--", label="RDF MSE")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_title(f"Objective Convergence ({total_atoms:,} Atoms)")
    axs[0, 0].grid(True, alpha=0.2, ls=":")
    axs[0, 0].legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    # Panel 2: Predicted vs Target g(r)
    with torch.no_grad():
        final_pred_rdf = compute_differentiable_rdf(q_T, r_bins, box_length, sigma=0.08).cpu().numpy()
    axs[0, 1].plot(r_bins.cpu().numpy(), target_rdf.cpu().numpy(), color="#f59e0b", lw=2.5, label="Target g(r)")
    axs[0, 1].plot(r_bins.cpu().numpy(), final_pred_rdf, color="#38bdf8", lw=2, ls="--", label="Trained SchNet g(r)")
    axs[0, 1].set_xlabel("Distance r (Å)")
    axs[0, 1].set_ylabel("g(r)")
    axs[0, 1].set_title("Radial Distribution Function Matching")
    axs[0, 1].grid(True, alpha=0.2, ls=":")
    axs[0, 1].legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    # Panel 3: Pressure Discrepancy
    axs[1, 0].plot(range(1, args.epochs + 1), history_p_err, color="#f43f5e", lw=2)
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("|P_pred - P_target|")
    axs[1, 0].set_title("Virial Pressure Discrepancy")
    axs[1, 0].grid(True, alpha=0.2, ls=":")

    # Panel 4: Memory Scaling O(1)
    axs[1, 1].plot(range(1, args.epochs + 1), history_vram, color="#a855f7", lw=2.5, label="Peak VRAM (MiB)")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("VRAM (MiB)")
    axs[1, 1].set_title(f"Strict O(1) Memory ({args.num_replicas} Replicas x {args.num_particles} Atoms)")
    axs[1, 1].grid(True, alpha=0.2, ls=":")
    axs[1, 1].legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    plt.tight_layout()
    plt.savefig(args.figure_path, dpi=200)
    plt.close()
    print(f">> Training diagnostics plot saved to: {args.figure_path}")


if __name__ == "__main__":
    run_ensemble_training()
