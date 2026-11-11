#!/usr/bin/env python3
"""
simulate_ensemble_production.py:
High-Throughput Production Molecular Dynamics Simulation Engine powered by S6*-Adjoint.
Loads trained MLFF Checkpoint (trained from RDF/Pressure without force labels),
Runs long-horizon bulk molecular simulation (up to 1,000+ particles, 1,000+ steps),
Verifies symplectic energy conservation, computes structural RDF, and exports standard XYZ trajectory.
"""
import os
import sys
import time
import argparse
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def build_parser():
    parser = argparse.ArgumentParser(description="Production MD using S6* with Trained Checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ensemble_mlff_checkpoint.pt", help="Path to trained checkpoint.")
    parser.add_argument("--num_particles", type=int, default=128, help="Number of atoms in simulation box (e.g. 128, 256, 512).")
    parser.add_argument("--num_steps", type=int, default=250, help="Total simulation macro-steps.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step size.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Initial temperature.")
    parser.add_argument("--density", type=float, default=0.84, help="Target number density rho = N / V.")
    parser.add_argument("--scheme", type=str, default="real_standard_33", choices=["real_standard_33", "real_stencil_12", "real_standard", "real_stencil", "real"], help="Branch: real_standard or real_flash")
    parser.add_argument("--device", type=str, default="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--output_xyz", type=str, default="trajectories/ensemble_production_simulation.xyz")
    parser.add_argument("--output_fig", type=str, default="trajectories/ensemble_production_diagnostics.png")
    return parser


def run_production_simulation():
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required to run simulate_ensemble_production.py.")
        sys.exit(1)

    args = build_parser().parse_args()
    device = torch.device(args.device)

    os.makedirs(os.path.dirname(args.output_xyz) if os.path.dirname(args.output_xyz) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_fig) if os.path.dirname(args.output_fig) else ".", exist_ok=True)

    print("=" * 80)
    print(">> S6*-Adjoint High-Throughput Production Molecular Dynamics Engine")
    print(f">> Device: {device}")
    print(f">> Loading Checkpoint: {args.checkpoint}")
    print("=" * 80)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # PyTorch 2.6 safe loading for metadata dictionaries
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    cfg = ckpt.get("model_config", {})
    dtype_str = cfg.get("dtype", "float64")
    dtype = torch.float64 if dtype_str == "float64" else torch.float32

    print(f">> Loaded Model Architecture: {cfg}")
    print(f">> Precision: {dtype_str} ({dtype})")

    from s6_adjoint.models.neural_forcefield import SchNetForceField
    from s6_adjoint.core.coefficients import COEFFICIENTS_REGISTRY, S6_REAL_A, S6_REAL_B, S6_REAL_C
    scheme_clean = 'real_standard_33' if args.scheme in ['real', 'real_standard'] else ('real_stencil_12' if args.scheme == 'real_stencil' else args.scheme)
    coeffs = COEFFICIENTS_REGISTRY.get(scheme_clean, {'A': S6_REAL_A, 'B': S6_REAL_B, 'C': S6_REAL_C})
    A_coeffs = tuple(float(x.real) if isinstance(x, complex) else float(x) for x in coeffs['A'])
    B_coeffs = tuple(float(x.real) if isinstance(x, complex) else float(x) for x in coeffs['B'])
    C_coeffs = tuple(float(x.real) if isinstance(x, complex) else float(x) for x in coeffs['C']) if coeffs['C'] is not None else tuple(0.0 for _ in B_coeffs)
    from s6_adjoint.diff_md.observables import (
        compute_kinetic_energy,
        compute_temperature,
        compute_differentiable_rdf
    )
    from s6_adjoint.diff_md.ensemble_init import generate_ensemble_configurations
    from s6_adjoint.diff_md.xyz_io import save_xyz_trajectory

    volume = float(args.num_particles) / float(args.density)
    box_length = float(volume ** (1.0 / 3.0))
    r_cutoff = min(cfg.get("cutoff", 4.0), box_length * 0.48)

    model = SchNetForceField(
        num_species=cfg.get("num_species", 5),
        hidden_dim=cfg.get("hidden_dim", 32),
        num_interactions=cfg.get("num_interactions", 2),
        num_rbf=cfg.get("num_rbf", 16),
        cutoff=r_cutoff,
        default_species=cfg.get("default_species", 0),
        dtype=dtype
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f">> Initializing {args.num_particles} atoms in cubic box L = {box_length:.4f} (rho = {args.density:.3f})")
    q0, p0, _ = generate_ensemble_configurations(
        num_replicas=1,
        num_particles=args.num_particles,
        target_temperature=args.temperature,
        density=args.density,
        box_length=box_length,
        seed=2026,
        dtype=dtype,
        device=device
    )

    q = q0[0].detach().clone() # (3 * N,)
    p = p0[0].detach().clone() # (3 * N,)

    num_bins = 50 if args.num_particles >= 128 else 40
    r_bins = torch.linspace(0.8, r_cutoff, num_bins, dtype=dtype, device=device)

    # Pre-evaluate initial energy
    with torch.no_grad():
        v0 = model(q).item()
        e_kin0 = compute_kinetic_energy(p).item()
        e_tot0 = v0 + e_kin0

    trajectory_q = [q.detach().clone()]
    energies_tot = [e_tot0]
    temperatures = [compute_temperature(p).item()]
    times = [0.0]

    print(f">> Commencing S6* Symplectic Rollout for {args.num_steps} steps (dt = {args.dt})...")
    print("-" * 78)
    print(f"{'Step':<10} | {'Kinetic Temp':<16} | {'Total Energy':<16} | {'Relative Drift |dE/E0|':<20}")
    print("-" * 78)

    start_sim_time = time.time()

    for step in range(1, args.num_steps + 1):
        with torch.no_grad():
            for k in range(6):
                h_b = args.dt * B_coeffs[k]
                h3_c = (args.dt ** 3) * C_coeffs[k]

                with torch.enable_grad():
                    q_req = q.detach().requires_grad_(True)
                    V = model(q_req)
                    F = -torch.autograd.grad(V, q_req, create_graph=True)[0]
                    half_F_sq = 0.5 * torch.sum(F ** 2)
                    G = torch.autograd.grad(half_F_sq, q_req, create_graph=False)[0]

                p.add_(h_b * F - 2.0 * h3_c * G)
                q.add_(args.dt * A_coeffs[k] * p)

            # B7 stage
            h_b = args.dt * B_coeffs[6]
            h3_c = (args.dt ** 3) * C_coeffs[6]
            with torch.enable_grad():
                q_req = q.detach().requires_grad_(True)
                V = model(q_req)
                F = -torch.autograd.grad(V, q_req, create_graph=True)[0]
                half_F_sq = 0.5 * torch.sum(F ** 2)
                G = torch.autograd.grad(half_F_sq, q_req, create_graph=False)[0]
            p.add_(h_b * F - 2.0 * h3_c * G)

        if step % 10 == 0 or step == args.num_steps:
            with torch.no_grad():
                v_curr = model(q).item()
                e_k_curr = compute_kinetic_energy(p).item()
                e_t_curr = v_curr + e_k_curr
                temp_curr = compute_temperature(p).item()
                drift = abs(e_t_curr - e_tot0) / (abs(e_tot0) + 1e-12)

            trajectory_q.append(q.detach().clone())
            energies_tot.append(e_t_curr)
            temperatures.append(temp_curr)
            times.append(step * args.dt)

            if step % 50 == 0 or step == args.num_steps:
                print(f"{step:<10} | {temp_curr:<16.4f} | {e_t_curr:<16.4f} | {drift:<20.2e}")

    sim_duration = time.time() - start_sim_time
    print("-" * 78)
    print(f">> Simulation completed in {sim_duration:.2f} s ({args.num_steps / sim_duration:.1f} steps/second)")

    # Export XYZ trajectory
    save_xyz_trajectory(trajectory_q, args.output_xyz, element="Ar")
    print(f">> XYZ Trajectory saved to: {args.output_xyz}")

    # Compute final RDF from trajectory
    final_q = trajectory_q[-1]
    with torch.no_grad():
        pred_rdf = compute_differentiable_rdf(final_q, r_bins, box_length, sigma=0.08).cpu().numpy()

    # Generate Diagnostics Figure
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

    # Panel 1: Energy Conservation
    axs[0, 0].plot(times, energies_tot, color="#34d399", lw=2)
    axs[0, 0].set_xlabel("Time (ps)")
    axs[0, 0].set_ylabel("Total Energy H")
    axs[0, 0].set_title(f"S6* Symplectic Invariance ({args.num_particles} Atoms, E0={e_tot0:.4f})")
    axs[0, 0].grid(True, alpha=0.2, ls=":")

    # Panel 2: Temperature Dynamics
    axs[0, 1].plot(times, temperatures, color="#f59e0b", lw=2)
    axs[0, 1].set_xlabel("Time (ps)")
    axs[0, 1].set_ylabel("Kinetic Temperature")
    axs[0, 1].set_title("Equipartition Temperature Evolution")
    axs[0, 1].grid(True, alpha=0.2, ls=":")

    # Panel 3: Radial Distribution Function g(r)
    target_rdf = ckpt.get("training_meta", {}).get("target_rdf", None)
    target_r_bins = ckpt.get("training_meta", {}).get("r_bins", None)
    if target_rdf is not None and target_r_bins is not None:
        axs[1, 0].plot(target_r_bins, target_rdf, color="#f59e0b", lw=2.5, label="Target g(r)")
    axs[1, 0].plot(r_bins.cpu().numpy(), pred_rdf, color="#38bdf8", lw=2, ls="--", label="Simulated g(r)")
    axs[1, 0].set_xlabel("Distance r (Å)")
    axs[1, 0].set_ylabel("g(r)")
    axs[1, 0].set_title(f"Bulk Structure Function ({args.num_particles} Atoms)")
    axs[1, 0].grid(True, alpha=0.2, ls=":")
    axs[1, 0].legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    # Panel 4: Relative Energy Drift
    drifts = [abs(e - e_tot0) / (abs(e_tot0) + 1e-12) for e in energies_tot]
    axs[1, 1].semilogy(times, drifts, color="#ec4899", lw=2)
    axs[1, 1].set_xlabel("Time (ps)")
    axs[1, 1].set_ylabel("|E(t) - E(0)| / |E(0)|")
    axs[1, 1].set_title("Symplectic Energy Drift Bound")
    axs[1, 1].grid(True, alpha=0.2, ls=":")

    plt.tight_layout()
    plt.savefig(args.output_fig, dpi=200)
    plt.close()
    print(f">> Production diagnostics plot saved to: {args.output_fig}")


if __name__ == "__main__":
    run_production_simulation()
