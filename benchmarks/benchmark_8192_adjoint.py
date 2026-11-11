#!/usr/bin/env python3
"""
Explicit 8,192-atom S6*-Adjoint benchmark.

Production profile:
    32 independent replicas x 256 atoms = 8,192 atoms total.
    The full MLFF/RDF training workload is GPU-oriented and is intentionally
    NOT executed by this sandbox-safe benchmark harness unless --full is set.

Sandbox-safe mode validates the actual S6*-Adjoint engine at the same 8,192-
atom state size using a differentiable harmonic Hamiltonian.  It sweeps the
number of integration steps and reports gradient correctness plus memory/time
metrics.  This isolates the adjoint memory-scaling claim from the much more
expensive all-pairs SchNet/RDF production workload.

Full production run (GPU):
    python benchmarks/benchmark_8192_adjoint.py --full --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from s6_adjoint.autograd.module import S6HamiltonianODE


@dataclass
class Result:
    steps: int
    atoms: int
    replicas: int
    device: str
    wall_seconds: float
    max_rss_mib: float
    current_rss_mib: float
    peak_vram_mib: float | None
    q_norm: float
    p_norm: float
    grad_norm: float


class HarmonicPotential(nn.Module):
    def __init__(self, k: float = 0.7, dtype=torch.float64):
        super().__init__()
        self.log_k = nn.Parameter(torch.tensor(math.log(k), dtype=dtype))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        k = torch.exp(self.log_k)
        return 0.5 * k * torch.sum(q * q, dim=-1)


def _max_rss_mib() -> float:
    # Linux ru_maxrss is KiB; macOS reports bytes. Sandbox is Linux, but keep
    # the conversion defensive for portable use.
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 ** 2)
    return rss / 1024.0


def _current_rss_mib() -> float:
    status_path = "/proc/self/status"
    try:
        with open(status_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def run_case(steps: int, replicas: int, atoms_per_replica: int, device_name: str, dtype_name: str) -> Result:
    device = torch.device(device_name)
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    total_atoms = replicas * atoms_per_replica
    coords = 3 * atoms_per_replica

    torch.manual_seed(1234)
    q0 = (0.01 * torch.randn(replicas, coords, device=device, dtype=dtype)).requires_grad_(False)
    p0 = (0.01 * torch.randn(replicas, coords, device=device, dtype=dtype)).requires_grad_(False)
    potential = HarmonicPotential(dtype=dtype).to(device)
    ode = S6HamiltonianODE(potential, scheme="real_standard_33", mass=1.0).to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    qT, pT = ode(q0, p0, 0.005, steps)
    loss = 0.5 * torch.mean(qT * qT) + 0.5 * torch.mean(pT * pT)
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    else:
        peak_vram = None

    elapsed = time.perf_counter() - t0
    grad_norm = float(potential.log_k.grad.detach().abs().item())
    return Result(
        steps=steps,
        atoms=total_atoms,
        replicas=replicas,
        device=str(device),
        wall_seconds=elapsed,
        max_rss_mib=_max_rss_mib(),
        current_rss_mib=_current_rss_mib(),
        peak_vram_mib=peak_vram,
        q_norm=float(torch.linalg.vector_norm(qT.detach()).item()),
        p_norm=float(torch.linalg.vector_norm(pT.detach()).item()),
        grad_norm=grad_norm,
    )


def run_production_full(args: argparse.Namespace) -> None:
    if args.device == "cpu":
        raise SystemExit("Refusing full 8192-atom production MLFF run on CPU. Use --device cuda.")
    # Delegate to the existing production benchmark so its observable/training
    # path remains the source of truth.
    from train_ensemble_rdf_mlff import build_parser, run_ensemble_training
    sys.argv = [sys.argv[0],
                "--num_replicas", str(args.replicas),
                "--num_particles", str(args.atoms_per_replica),
                "--num_steps", str(args.num_steps),
                "--epochs", str(args.epochs),
                "--dt", str(args.dt),
                "--scheme", args.scheme,
                "--device", args.device,
                "--dtype", args.dtype]
    run_ensemble_training()


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit 8192-atom S6*-Adjoint benchmark harness")
    parser.add_argument("--replicas", type=int, default=32)
    parser.add_argument("--atoms-per-replica", type=int, default=256)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--full", action="store_true", help="Run the full GPU MLFF/RDF production workload.")
    parser.add_argument("--num-steps", type=int, default=30, help="Production trajectory steps per epoch when --full is used.")
    parser.add_argument("--epochs", type=int, default=167, help="Production training epochs when --full is used (160,320 replica-steps at 32x256x167).")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--scheme", default="real_standard_33", choices=["real_standard_33", "real_stencil_12"])
    parser.add_argument("--json", default=None, help="Write benchmark results as JSON.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    total_atoms = args.replicas * args.atoms_per_replica
    if total_atoms != 8192:
        raise SystemExit(
            f"This benchmark is explicitly pinned to 8,192 atoms; got "
            f"{args.replicas} x {args.atoms_per_replica} = {total_atoms}."
        )

    print(f"8192-atom profile: {args.replicas} replicas x {args.atoms_per_replica} atoms")
    print(f"Mode: {'FULL production GPU' if args.full else 'sandbox-safe adjoint engine'}")

    if args.full:
        run_production_full(args)
        return

    results = []
    if args.child:
        if len(args.steps) != 1:
            raise SystemExit("--child accepts exactly one step count")
        steps = args.steps[0]
        if steps < 0:
            raise SystemExit("steps must be non-negative")
        r = run_case(steps, args.replicas, args.atoms_per_replica, args.device, args.dtype)
        print(json.dumps(r.__dict__))
        return

    # Isolate each step-count in a subprocess so ru_maxrss is a true per-case peak.
    for steps in args.steps:
        if steps < 0:
            raise SystemExit("steps must be non-negative")
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--child", "--steps", str(steps),
            "--replicas", str(args.replicas),
            "--atoms-per-replica", str(args.atoms_per_replica),
            "--device", args.device, "--dtype", args.dtype,
        ]
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        r = Result(**json.loads(completed.stdout.strip().splitlines()[-1]))
        results.append(r)
        vram = "n/a" if r.peak_vram_mib is None else f"{r.peak_vram_mib:8.1f} MiB"
        print(
            f"steps={r.steps:>3} | atoms={r.atoms:>5} | "
            f"time={r.wall_seconds:8.3f}s | peak RSS={r.max_rss_mib:8.1f} MiB | "
            f"VRAM={vram} | grad={r.grad_norm:.6e}"
        )

    if args.json:
        payload = [r.__dict__ for r in results]
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
