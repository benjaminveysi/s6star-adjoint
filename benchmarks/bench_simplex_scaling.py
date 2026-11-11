#!/usr/bin/env python3
"""Scaling benchmark for the S6* Simplex-Commutator adjoint backend.

Sweeps mesh resolution and Krylov subspace depth, timing forward and
backward passes separately.  This is the harness to run on the *target
accelerator* before quoting throughput numbers: the in-repo smoke test only
validates correctness, not production GPU performance.

Example:
    python benchmarks/bench_simplex_scaling.py --device cpu \\
        --grids 16,32,64 --m-krylov 8,16 --steps 2
    python benchmarks/bench_simplex_scaling.py --device cuda \\
        --grids 64,128 --m-krylov 16 --steps 4 --csv scaling_cuda.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from s6_adjoint.simplex.module import NodalPotential, S6SimplexAdjoint
from src.simplex_mesh import SimplexMesh2D


def _parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def bench_one(mesh, steps, m_krylov, device, repeats, compile_kernels):
    vals = 0.05 * (mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2) + 0.02
    pot = NodalPotential(vals, dtype=torch.float64).to(device)
    eng = S6SimplexAdjoint(mesh, pot, m_krylov=m_krylov,
                           compile_kernels=compile_kernels).to(device)
    h = 0.03
    N = mesh.N_nodes
    psi0 = (torch.randn(N, dtype=torch.float64, device=device)
            + 1j * torch.randn(N, dtype=torch.float64, device=device)).to(torch.complex128)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # Warmup (includes torch.compile on first call when enabled).
    with torch.no_grad():
        eng(psi0, h, 1)
    sync()

    fwd_times, bwd_times = [], []
    for _ in range(repeats):
        with torch.no_grad():
            sync()
            t0 = time.perf_counter()
            out = eng(psi0, h, steps)
            sync()
            fwd_times.append(time.perf_counter() - t0)

        psi = psi0.clone().requires_grad_(True)
        for p in pot.parameters():
            if p.grad is not None:
                p.grad = None
        sync()
        t0 = time.perf_counter()
        out_b = eng(psi, h, steps)
        loss = (out_b.real ** 2 + out_b.imag ** 2).sum()
        loss.backward()
        sync()
        bwd_times.append(time.perf_counter() - t0)

    nnz = int(mesh.K.nnz)
    return {
        "nodes": N,
        "triangles": int(mesh.N_elements),
        "nnz": nnz,
        "m_krylov": m_krylov,
        "steps": steps,
        "fwd_ms_per_step": 1000.0 * min(fwd_times) / steps,
        "bwd_ms_per_step": 1000.0 * min(bwd_times) / steps,
    }


def main():
    ap = argparse.ArgumentParser(description="S6* Simplex scaling benchmark")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--grids", default="16,32,64",
                    help="comma-separated Nx=Ny grid resolutions")
    ap.add_argument("--m-krylov", default="8,16",
                    help="comma-separated Krylov depths")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--compile", action="store_true",
                    help="enable torch.compile kernels")
    ap.add_argument("--csv", default=None, help="write results to CSV file")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    rows = []
    for n in _parse_int_list(args.grids):
        mesh = SimplexMesh2D(Lx=6.0, Ly=6.0, Nx=n, Ny=n, periodic=True)
        for m in _parse_int_list(args.m_krylov):
            row = bench_one(mesh, args.steps, m, args.device,
                            args.repeats, args.compile)
            rows.append(row)
            print(
                f"grid {n:4d} | nodes {row['nodes']:6d} | tri {row['triangles']:6d} "
                f"| nnz {row['nnz']:7d} | krylov {m:2d} | "
                f"fwd {row['fwd_ms_per_step']:8.1f} ms/step | "
                f"bwd {row['bwd_ms_per_step']:8.1f} ms/step",
                flush=True,
            )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
