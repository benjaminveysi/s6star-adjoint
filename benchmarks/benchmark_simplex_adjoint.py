#!/usr/bin/env python3
"""Benchmark the differentiable S6* Simplex-Commutator backend.

The default profile is exactly 8,192 FE nodes (64 x 128).  It intentionally
runs a small step sweep only; a full scientific/GPU workload is left to the
user's target accelerator.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import torch
import torch.nn as nn

from s6_adjoint import S6SimplexAdjoint, SimplexMesh2D


class SmoothPotential(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.01, dtype=dtype))
        self.b = nn.Parameter(torch.tensor(0.02, dtype=dtype))

    def forward(self, coords):
        return self.w * (coords[:, 0] ** 2 + 0.5 * coords[:, 1] ** 2) + self.b


def peak_rss_mib():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def run_case(args):
    mesh = SimplexMesh2D(
        Lx=args.Lx, Ly=args.Ly, Nx=args.Nx, Ny=args.Ny,
        periodic=True, mode="grid"
    )
    if mesh.N_nodes != args.nodes:
        raise RuntimeError(f"expected {args.nodes} nodes, got {mesh.N_nodes}")

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    potential = SmoothPotential(dtype=dtype).to(device)
    engine = S6SimplexAdjoint(mesh, potential, m_krylov=args.m_krylov).to(device)

    torch.manual_seed(2026)
    psi = (torch.randn(mesh.N_nodes, device=device, dtype=dtype) +
           1j * torch.randn(mesh.N_nodes, device=device, dtype=dtype)).to(torch.complex128 if dtype == torch.float64 else torch.complex64)
    with torch.no_grad():
        mass = engine.backend.mass_diag.to(device=device, dtype=psi.real.dtype)
        psi /= torch.sqrt(torch.real(torch.sum(torch.conj(psi) * mass * psi)))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    psi.requires_grad_(True)
    out = engine(psi, args.dt, args.steps)
    mass = engine.backend.mass_diag.to(device=device, dtype=out.real.dtype)
    loss = torch.real(torch.sum(torch.conj(out) * mass * out))
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    else:
        peak_vram = None

    return {
        "nodes": mesh.N_nodes,
        "elements": mesh.N_elements,
        "nnz": mesh.K.nnz,
        "steps": args.steps,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_mib": peak_rss_mib(),
        "peak_vram_mib": peak_vram,
        "loss": float(loss.detach().cpu()),
        "parameter_grad_norm": float(potential.w.grad.detach().abs().cpu()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=8192)
    p.add_argument("--Nx", type=int, default=64)
    p.add_argument("--Ny", type=int, default=128)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--m-krylov", type=int, default=16)
    p.add_argument("--Lx", type=float, default=20.0)
    p.add_argument("--Ly", type=float, default=20.0)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be >= 1")
    result = run_case(args)
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
