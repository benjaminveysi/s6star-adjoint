"""
===================================================================================================
S6* SIMPLEX 2D SIMPLICIAL MESH & ASSEMBLY ENGINE
===================================================================================================
Constructs 2D triangular simplicial meshes:
1. Structured grids with optional coordinate perturbation
2. Genuine Unstructured Delaunay triangulations (via scipy.spatial.Delaunay)

Computes linear P1 shape function gradients on simplices, assembles the global sparse stiffness 
matrix K and diagonal lumped mass matrix M, and provides geometric element quality checks.

Runtime Execution Model:
    The computational primitive at runtime is global sparse matrix-vector multiplication (SpMV):
        w = M^-1 * (K @ v)
    Instrumented with separate runtime counters for kinetic K-SpMVs and commutator C-SpMVs so sparse operator work is empirical and auditable.
===================================================================================================
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay

class SimplexMesh2D:
    def __init__(self, Lx=40.0, Ly=40.0, Nx=64, Ny=64, periodic=True, perturbation=0.0, mode='grid', random_seed=42):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.periodic = periodic
        self.perturbation = perturbation
        self.mode = mode.lower()
        self.random_seed = random_seed
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.spmv_count = 0
        self.commutator_spmv_count = 0
        self._commutator_matrix = None

        self._build_mesh()
        self._assemble_matrices()


    def compute_discrete_commutator(self, V):
        """
        Computes the exact discrete double commutator C = [V, [M^-1 K, V]] on the simplicial mesh.
        Matches the sparsity pattern of the FE stiffness matrix K with zero diagonal.
        Self-adjoint in the lumped mass inner product: <u, C v>_M = <C u, v>_M.
        """
        import scipy.sparse as sp
        V_vec = np.asarray(V, dtype=float)
        V_sp = sp.diags(V_vec)
        M_inv_sp = sp.diags(self.inv_M_diag)
        T_sp = M_inv_sp.dot(self.K)
        comm_1 = T_sp.dot(V_sp) - V_sp.dot(T_sp)
        C_sp = (V_sp.dot(comm_1) - comm_1.dot(V_sp)).tocsr()
        return C_sp

    def reset_spmv_count(self):
        self.spmv_count = 0
        self.commutator_spmv_count = 0

    def get_spmv_count(self):
        """Return kinetic Laplacian K-SpMV count."""
        return self.spmv_count

    def get_commutator_spmv_count(self):
        """Return sparse discrete-commutator C-SpMV count."""
        return self.commutator_spmv_count

    def get_total_sparse_mv_count(self):
        """Return total sparse matvec work (K-SpMVs + C-SpMVs)."""
        return self.spmv_count + self.commutator_spmv_count

    def _build_mesh(self):
        if self.mode == 'delaunay':
            rng = np.random.RandomState(self.random_seed)
            n_bx = self.Nx
            n_by = self.Ny
            bx_top = np.column_stack([np.linspace(-self.Lx/2, self.Lx/2, n_bx), np.full(n_bx, self.Ly/2)])
            bx_bot = np.column_stack([np.linspace(-self.Lx/2, self.Lx/2, n_bx), np.full(n_bx, -self.Ly/2)])
            by_left = np.column_stack([np.full(n_by, -self.Lx/2), np.linspace(-self.Ly/2, self.Ly/2, n_by)])
            by_right = np.column_stack([np.full(n_by, self.Lx/2), np.linspace(-self.Ly/2, self.Ly/2, n_by)])
            
            n_interior = (self.Nx - 2) * (self.Ny - 2)
            interior_x = rng.uniform(-self.Lx/2 * 0.95, self.Lx/2 * 0.95, n_interior)
            interior_y = rng.uniform(-self.Ly/2 * 0.95, self.Ly/2 * 0.95, n_interior)
            interior = np.column_stack([interior_x, interior_y])
            
            all_pts = np.vstack([bx_top, bx_bot, by_left, by_right, interior])
            all_pts = np.unique(np.round(all_pts, decimals=8), axis=0)
            
            tri = Delaunay(all_pts)
            self.nodes = all_pts
            self.elements = tri.simplices.astype(np.int32)
            self.N_nodes = len(self.nodes)
            self.N_elements = len(self.elements)
            self.periodic = False
            return

        if self.periodic:
            x = np.linspace(-self.Lx / 2.0, self.Lx / 2.0, self.Nx, endpoint=False)
            y = np.linspace(-self.Ly / 2.0, self.Ly / 2.0, self.Ny, endpoint=False)
            X, Y = np.meshgrid(x, y, indexing='ij')

            if self.perturbation > 0.0:
                rng = np.random.RandomState(self.random_seed)
                X = X + rng.uniform(-self.perturbation * self.dx, self.perturbation * self.dx, X.shape)
                Y = Y + rng.uniform(-self.perturbation * self.dy, self.perturbation * self.dy, Y.shape)

            self.N_nodes = self.Nx * self.Ny
            self.nodes = np.column_stack([X.ravel(), Y.ravel()])

            triangles = []
            for i in range(self.Nx):
                i_next = (i + 1) % self.Nx
                for j in range(self.Ny):
                    j_next = (j + 1) % self.Ny
                    n00 = i * self.Ny + j
                    n10 = i_next * self.Ny + j
                    n01 = i * self.Ny + j_next
                    n11 = i_next * self.Ny + j_next
                    triangles.append([n00, n10, n11])
                    triangles.append([n00, n11, n01])
            self.elements = np.array(triangles, dtype=np.int32)
        else:
            x = np.linspace(-self.Lx / 2.0, self.Lx / 2.0, self.Nx, endpoint=True)
            y = np.linspace(-self.Ly / 2.0, self.Ly / 2.0, self.Ny, endpoint=True)
            X, Y = np.meshgrid(x, y, indexing='ij')

            if self.perturbation > 0.0:
                rng = np.random.RandomState(self.random_seed)
                X[1:-1, 1:-1] += rng.uniform(-self.perturbation * self.dx, self.perturbation * self.dx, X[1:-1, 1:-1].shape)
                Y[1:-1, 1:-1] += rng.uniform(-self.perturbation * self.dy, self.perturbation * self.dy, Y[1:-1, 1:-1].shape)

            self.N_nodes = self.Nx * self.Ny
            self.nodes = np.column_stack([X.ravel(), Y.ravel()])

            triangles = []
            for i in range(self.Nx - 1):
                for j in range(self.Ny - 1):
                    n00 = i * self.Ny + j
                    n10 = (i + 1) * self.Ny + j
                    n01 = i * self.Ny + (j + 1)
                    n11 = (i + 1) * self.Ny + (j + 1)
                    triangles.append([n00, n10, n11])
                    triangles.append([n00, n11, n01])
            self.elements = np.array(triangles, dtype=np.int32)

        self.N_elements = len(self.elements)

    def _assemble_matrices(self):
        rows_k = []
        cols_k = []
        vals_k = []
        m_lumped = np.zeros(self.N_nodes, dtype=float)

        for elem in self.elements:
            v = self.nodes[elem]
            v1, v2, v3 = v[0], v[1], v[2]

            d12 = v2 - v1
            d13 = v3 - v1
            if self.periodic:
                d12[0] = (d12[0] + self.Lx/2) % self.Lx - self.Lx/2
                d12[1] = (d12[1] + self.Ly/2) % self.Ly - self.Ly/2
                d13[0] = (d13[0] + self.Lx/2) % self.Lx - self.Lx/2
                d13[1] = (d13[1] + self.Ly/2) % self.Ly - self.Ly/2

            area = 0.5 * abs(d12[0] * d13[1] - d13[0] * d12[1])
            if area < 1e-14:
                area = 0.5 * self.dx * self.dy

            b = np.array([d12[1] - d13[1], d13[1], -d12[1]])
            c = np.array([d13[0] - d12[0], -d13[0], d12[0]])
            grad_phi = np.vstack([b, c]) / (2.0 * area)

            k_elem = area * (grad_phi.T @ grad_phi)

            for a_idx in range(3):
                m_lumped[elem[a_idx]] += area / 3.0
                for b_idx in range(3):
                    rows_k.append(elem[a_idx])
                    cols_k.append(elem[b_idx])
                    vals_k.append(k_elem[a_idx, b_idx])

        self.K = sp.coo_matrix((vals_k, (rows_k, cols_k)), shape=(self.N_nodes, self.N_nodes)).tocsc()
        self.M_diag = m_lumped
        self.inv_M_diag = 1.0 / np.where(m_lumped > 1e-15, m_lumped, 1.0)
        self.total_area = np.sum(m_lumped)

    def compute_quality_metrics(self):
        areas = []
        min_angles = []
        aspect_ratios = []

        for elem in self.elements:
            pts = self.nodes[elem]
            v1, v2, v3 = pts[0], pts[1], pts[2]

            d12 = v2 - v1
            d23 = v3 - v2
            d31 = v1 - v3

            if self.periodic:
                d12 = (d12 + [self.Lx/2, self.Ly/2]) % [self.Lx, self.Ly] - [self.Lx/2, self.Ly/2]
                d23 = (d23 + [self.Lx/2, self.Ly/2]) % [self.Lx, self.Ly] - [self.Lx/2, self.Ly/2]
                d31 = (d31 + [self.Lx/2, self.Ly/2]) % [self.Lx, self.Ly] - [self.Lx/2, self.Ly/2]

            a = np.linalg.norm(d23)
            b = np.linalg.norm(d31)
            c = np.linalg.norm(d12)

            area = 0.5 * abs(d12[0] * (-d31[1]) - (-d31[0]) * d12[1])
            if area < 1e-15:
                continue
            areas.append(area)

            cos_A = np.clip((b**2 + c**2 - a**2) / (2.0 * b * c + 1e-16), -1.0, 1.0)
            cos_B = np.clip((a**2 + c**2 - b**2) / (2.0 * a * c + 1e-16), -1.0, 1.0)
            cos_C = np.clip((a**2 + b**2 - c**2) / (2.0 * a * b + 1e-16), -1.0, 1.0)

            ang_A = np.degrees(np.arccos(cos_A))
            ang_B = np.degrees(np.arccos(cos_B))
            ang_C = np.degrees(np.arccos(cos_C))
            min_angles.append(min(ang_A, ang_B, ang_C))

            s = 0.5 * (a + b + c)
            r_in = area / (s + 1e-16)
            r_circ = (a * b * c) / (4.0 * area + 1e-16)
            aspect_ratio = r_circ / (2.0 * r_in + 1e-16)
            aspect_ratios.append(aspect_ratio)

        return {
            'min_area': float(np.min(areas)),
            'max_area': float(np.max(areas)),
            'min_angle_deg': float(np.min(min_angles)),
            'max_aspect_ratio': float(np.max(aspect_ratios)),
            'mean_aspect_ratio': float(np.mean(aspect_ratios))
        }

    def inner_product(self, u, v):
        return np.sum(u.conj() * (self.M_diag * v))

    def norm(self, u):
        return np.sqrt(np.abs(self.inner_product(u, u)))

    def apply_laplacian(self, u):
        self.spmv_count += 1
        return self.inv_M_diag * self.K.dot(u)

    def apply_commutator(self, u):
        """Apply the assembled discrete double commutator C to u."""
        if self._commutator_matrix is None:
            return np.zeros_like(u)
        self.commutator_spmv_count += 1
        return self._commutator_matrix.dot(u)
