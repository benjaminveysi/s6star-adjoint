"""
Unit test verifying Simplex dual-field physical integrity:
1. Hamiltonian wave: Exact unitary norm conservation |Delta N| < 1e-13 on both grid and Delaunay meshes
2. Parabolic diffusion: Non-expansive / contractive energy and norm evolution
"""
import sys, os
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.simplex_mesh import SimplexMesh2D
from src.simplex_solver import SimplexSolver

def test_simplex_hamiltonian_unitarity_grid():
    mesh = SimplexMesh2D(Lx=10.0, Ly=10.0, Nx=16, Ny=16, periodic=True)
    solver = SimplexSolver(mesh, field_type='complex', theta=0.0)
    r2 = mesh.nodes[:, 0]**2 + mesh.nodes[:, 1]**2
    psi0 = np.exp(-0.5 * r2) * np.exp(1j * 0.5 * mesh.nodes[:, 0])
    psi0 /= mesh.norm(psi0)

    psi_f = solver.integrate_simplex_s6(psi0, n_steps=20, h=0.01, use_fsal=True)
    norm_defect = abs(mesh.norm(psi_f) - 1.0)
    assert norm_defect < 1e-13, f"Unitary norm defect too large on grid: {norm_defect}"
    print(f"test_simplex_hamiltonian_unitarity_grid: PASSED (Delta N = {norm_defect:.2e})")

def test_simplex_hamiltonian_unitarity_delaunay():
    mesh = SimplexMesh2D(Lx=10.0, Ly=10.0, Nx=16, Ny=16, mode='delaunay', random_seed=42)
    solver = SimplexSolver(mesh, field_type='complex', theta=0.0)
    r2 = mesh.nodes[:, 0]**2 + mesh.nodes[:, 1]**2
    psi0 = np.exp(-0.5 * r2) * np.exp(1j * 0.5 * mesh.nodes[:, 0])
    psi0 /= mesh.norm(psi0)

    psi_f = solver.integrate_simplex_s6(psi0, n_steps=20, h=0.01, use_fsal=True)
    norm_defect = abs(mesh.norm(psi_f) - 1.0)
    assert norm_defect < 1e-13, f"Unitary norm defect too large on Delaunay: {norm_defect}"
    print(f"test_simplex_hamiltonian_unitarity_delaunay: PASSED (Delta N = {norm_defect:.2e})")

def test_simplex_parabolic_contraction():
    mesh = SimplexMesh2D(Lx=10.0, Ly=10.0, Nx=16, Ny=16, periodic=True)
    solver = SimplexSolver(mesh, field_type='real', theta=0.0)
    r2 = mesh.nodes[:, 0]**2 + mesh.nodes[:, 1]**2
    u0 = np.exp(-0.5 * r2)
    u0 /= mesh.norm(u0)

    u_f = solver.integrate_simplex_s6(u0, n_steps=20, h=0.01, use_fsal=False)
    contraction = mesh.norm(u_f) / mesh.norm(u0)
    assert 0.0 < contraction <= 1.0, f"Expected non-expansive / contractive, got ratio: {contraction}"
    print(f"test_simplex_parabolic_contraction: PASSED (Contraction ratio = {contraction:.4f})")

if __name__ == '__main__':
    test_simplex_hamiltonian_unitarity_grid()
    test_simplex_hamiltonian_unitarity_delaunay()
    test_simplex_parabolic_contraction()
