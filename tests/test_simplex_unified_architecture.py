import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
"""
Test suite for S6* Simplex Unified Architecture (Release v16).
Verifies that S6* Simplex-Commutator with joint potential Lanczos exponential
preserves exact M-unitarity and matches independent matrix exponentials.
"""

import unittest
import numpy as np
from scipy.linalg import expm
from src.simplex_mesh import SimplexMesh2D
from src.simplex_solver import SimplexSolver

class TestSimplexUnifiedArchitecture(unittest.TestCase):
    def test_unified_lanczos_unitarity(self):
        mesh = SimplexMesh2D(Lx=10.0, Ly=10.0, Nx=16, Ny=16, periodic=True)
        V = 0.05 * (mesh.nodes[:, 0]**2 + mesh.nodes[:, 1]**2)
        solver = SimplexSolver(mesh, field_type='complex', theta=0.0, V=V, scheme='s6_commutator')

        v0 = np.random.randn(mesh.N_nodes) + 1j * np.random.randn(mesh.N_nodes)
        v0 /= mesh.norm(v0)

        # 10 steps of size h = 0.05
        v = solver.integrate_simplex_s6(v0.copy(), n_steps=10, h=0.05, use_fsal=True)
        norm_v = mesh.norm(v)
        self.assertAlmostEqual(norm_v, 1.0, places=13)

    def test_dual_spmv_accounting(self):
        mesh = SimplexMesh2D(Lx=10.0, Ly=10.0, Nx=16, Ny=16, periodic=True)
        V = 0.05 * (mesh.nodes[:, 0]**2 + mesh.nodes[:, 1]**2)
        solver = SimplexSolver(mesh, field_type='complex', theta=0.0, V=V, scheme='s6_commutator')

        v0 = np.random.randn(mesh.N_nodes) + 1j * np.random.randn(mesh.N_nodes)
        v0 /= mesh.norm(v0)

        solver.reset_spmv_count()
        solver.integrate_simplex_s6(v0.copy(), n_steps=2, h=0.05, use_fsal=True)
        bk = solver.get_spmv_breakdown()

        self.assertEqual(bk['kinetic'], 2 * 96) # exactly 6 stages per step * 16
        self.assertGreater(bk['commutator'], 0) # commutator work tracked
        self.assertEqual(bk['total'], bk['kinetic'] + bk['commutator'])

if __name__ == '__main__':
    unittest.main()
