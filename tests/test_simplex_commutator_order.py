import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
"""
Asymptotic order test for S6* Simplex-Commutator with Unified Lanczos Potential Stage.
Verifies true 6th-order convergence (p >= 5.8) across consecutive step sizes.
"""

import unittest
import numpy as np
from scipy.linalg import expm
from src.simplex_mesh import SimplexMesh2D
from src.simplex_solver import SimplexSolver

class TestSimplexCommutatorOrder(unittest.TestCase):
    def test_structured_mesh_order_6(self):
        mesh = SimplexMesh2D(Lx=20.0, Ly=20.0, Nx=16, Ny=16, periodic=True, mode='grid')
        V = 0.05 * (mesh.nodes[:, 0]**2 + mesh.nodes[:, 1]**2)
        A = mesh.K.toarray() / mesh.M_diag[:, None] + np.diag(V)

        r2 = mesh.nodes[:, 0]**2 + mesh.nodes[:, 1]**2
        psi0 = (1.0 / np.sqrt(np.pi)) * np.exp(-0.5 * r2) * np.exp(1j * (mesh.nodes[:, 0] + 0.5 * mesh.nodes[:, 1]))
        psi0 /= mesh.norm(psi0)

        T_target = 0.40
        psi_exact = expm(-1j * T_target * A).dot(psi0)

        solver = SimplexSolver(mesh, field_type='complex', theta=0.0, V=V, scheme='s6_commutator')
        hs = [0.40, 0.20, 0.10]
        errs = []
        for h in hs:
            n_steps = int(round(T_target / h))
            v = solver.integrate_simplex_s6(psi0.copy(), n_steps=n_steps, h=h, m_krylov=16, use_fsal=True)
            errs.append(mesh.norm(v - psi_exact))

        order1 = np.log2(errs[0] / errs[1])
        order2 = np.log2(errs[1] / errs[2])

        self.assertGreaterEqual(order1, 5.8)
        self.assertGreaterEqual(order2, 5.8)

if __name__ == '__main__':
    unittest.main()
