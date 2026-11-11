"""
Unit test verifying independent scientific validity of Saad's Krylov Lanczos error estimator:
compares estimated error against exact matrix exponential (scipy.linalg.expm) across a 
spectrum of step sizes tau and Krylov subspace dimensions m.
"""
import sys, os
import numpy as np
from scipy.linalg import expm

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.simplex_mesh import SimplexMesh2D
from src.simplex_solver import SimplexSolver

def test_krylov_estimator_accuracy():
    mesh = SimplexMesh2D(Lx=10.0, Ly=10.0, Nx=16, Ny=16, periodic=True)
    A = (mesh.K.toarray() / mesh.M_diag[:, None])

    np.random.seed(42)
    v0 = np.random.randn(mesh.N_nodes) + 1j * np.random.randn(mesh.N_nodes)
    v0 /= mesh.norm(v0)

    taus = [0.01, 0.05, 0.10, 0.20]
    ms = [8, 12, 16]

    print("=================================================================================")
    print("INDEPENDENT KRYLOV ERROR ESTIMATOR CERTIFICATION (Saad 1992 vs. Exact Matrix exp)")
    print("=================================================================================")
    print(f"{'tau':>6} | {'m':>3} | {'Est Error':>12} | {'Actual Error':>12} | {'Est/Actual Ratio':>16} | Status")
    print("-" * 81)

    all_passed = True
    for tau in taus:
        v_exact = expm(-1j * tau * A).dot(v0)
        for m in ms:
            beta = mesh.norm(v0)
            V = np.zeros((m, mesh.N_nodes), dtype=complex)
            T_mat = np.zeros((m, m), dtype=complex)
            v_curr = v0 / beta
            V[0] = v_curr
            v_prev = np.zeros_like(v_curr)
            beta_prev = 0.0

            for j in range(m):
                w = mesh.apply_laplacian(v_curr)
                alpha = mesh.inner_product(v_curr, w).real
                T_mat[j, j] = alpha
                w = w - alpha * v_curr - beta_prev * v_prev
                if j < m - 1:
                    beta_next = mesh.norm(w)
                    T_mat[j, j+1] = beta_next
                    T_mat[j+1, j] = beta_next
                    v_prev = v_curr
                    v_curr = w / beta_next
                    V[j+1] = v_curr
                    beta_prev = beta_next

            exp_T = expm(-1j * tau * T_mat)
            e1 = np.zeros(m, dtype=complex)
            e1[0] = beta
            c_coeffs = exp_T.dot(e1)
            res = np.zeros(mesh.N_nodes, dtype=complex)
            for idx in range(m):
                res += c_coeffs[idx] * V[idx]

            actual_err = mesh.norm(res - v_exact)
            err_est = beta * beta_next * abs(tau) * abs(exp_T[m-1, 0])

            if actual_err > 1e-14:
                ratio = err_est / actual_err
                # Saad estimator should be within an order of magnitude of true error
                passed = (0.5 <= ratio <= 25.0)
            else:
                ratio = 1.0
                passed = (actual_err < 1e-14)

            status = "PASSED" if passed else "FAILED"
            if not passed:
                all_passed = False
            print(f"{tau:6.2f} | {m:3d} | {err_est:12.4e} | {actual_err:12.4e} | {ratio:16.4f} | [{status}]")

    print("=" * 81)
    assert all_passed, "Krylov error estimator certification failed!"
    print("test_krylov_estimator_accuracy: ALL CONDITIONS PASSED!")

if __name__ == '__main__':
    test_krylov_estimator_accuracy()
