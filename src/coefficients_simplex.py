"""
===================================================================================================
S6* SIMPLEX COEFFICIENTS: GENUINE S6* COMMUTATOR & CLASSICAL BASELINE MATRICES
===================================================================================================
Epistemological Delineation of Geometry-Native Splitting on Simplicial Meshes:

1. GENUINE S6* SIMPLEX-COMMUTATOR (13-Stage Commutator-Corrected Splitting on Simplicial Meshes):
   - Employs the canonical 13-stage palindromic S6* coefficients:
     B_1 A_1 B_2 A_2 B_3 A_3 B_4 A_3 B_3 A_2 B_2 A_1 B_1
     with 6 kinetic stages (a_j) and 7 potential stages (b_j, c_j).
   - In quantum wave dynamics (K = C), operates with S6_REAL_A, S6_REAL_B, S6_REAL_C.
     Evaluates the exact discrete double commutator C = [V, [M^-1 K, V]] as a sparse matrix
     sharing the exact sparsity pattern of the FE stiffness matrix K (identically self-adjoint).
   - Achieves 6th-order convergence (p = 5.90 - 6.00) and machine-precision unitarity on both
     structured periodic grids and unstructured Delaunay triangulations.
   - Cost: no kinetic FSAL; exact B-boundary fusion is used. With fixed m=16, asymptotic sparse work is 96 K-SpMVs + 96 C-SpMVs per macro-step.

2. CLASSICAL EXTERNAL BASELINES (Retained for Transparent Academic Comparison):
   - YOSHIDA-6 (1990, Phys. Lett. A): Classical 7-stage recursive fractal composition without
     commutators (pure splitting). 112 SpMVs/step with FSAL (formerly mislabeled as 'Real-Pure-14').
   - BCChM-6 P6S7 (Blanes et al. 2013, Math. Comp.): Classical 7-stage forward-positive complex
     composition for parabolic diffusion. 112 SpMVs/step with FSAL (formerly mislabeled as 'Complex-Bullet-14').
===================================================================================================
"""

import numpy as np
from src.coefficients import (
    S6_REAL_A, S6_REAL_B, S6_REAL_C,
    S6_COMPLEX_A, S6_COMPLEX_B, S6_COMPLEX_C,
    YOSHIDA6_W, BCCHM_GAMMAS
)

# 1. Genuine S6* Commutator-Corrected Simplex Manifold (Canonical 13-Stage)
S6_SIMPLEX_COMM_A = S6_REAL_A
S6_SIMPLEX_COMM_B = S6_REAL_B
S6_SIMPLEX_COMM_C = S6_REAL_C

S6_SIMPLEX_PARAB_A = S6_COMPLEX_A
S6_SIMPLEX_PARAB_B = S6_COMPLEX_B
S6_SIMPLEX_PARAB_C = S6_COMPLEX_C

# 2. Classical Pure Composition Baselines (Yoshida 1990 & BCChM 2013)
w_y = np.array(YOSHIDA6_W, dtype=float)
YOSHIDA_6_A = np.array([
    0.5 * w_y[0],
    0.5 * (w_y[0] + w_y[1]),
    0.5 * (w_y[1] + w_y[2]),
    0.5 * (w_y[2] + w_y[3]),
    0.5 * (w_y[3] + w_y[4]),
    0.5 * (w_y[4] + w_y[5]),
    0.5 * (w_y[5] + w_y[6]),
    0.5 * w_y[6],
], dtype=float)
YOSHIDA_6_B = w_y.copy()

w_b = np.array(BCCHM_GAMMAS, dtype=complex)
BCCHM_6_A = np.array([
    0.5 * w_b[0],
    0.5 * (w_b[0] + w_b[1]),
    0.5 * (w_b[1] + w_b[2]),
    0.5 * (w_b[2] + w_b[3]),
    0.5 * (w_b[3] + w_b[4]),
    0.5 * (w_b[4] + w_b[5]),
    0.5 * (w_b[5] + w_b[6]),
    0.5 * w_b[6],
], dtype=complex)
BCCHM_6_B = w_b.copy()

# Aliases for backward compatibility
SIMPLEX_A_REAL = S6_SIMPLEX_COMM_A
SIMPLEX_B_REAL = S6_SIMPLEX_COMM_B
SIMPLEX_C_REAL = S6_SIMPLEX_COMM_C

SIMPLEX_A_COMPLEX = S6_SIMPLEX_PARAB_A
SIMPLEX_B_COMPLEX = S6_SIMPLEX_PARAB_B
SIMPLEX_C_COMPLEX = S6_SIMPLEX_PARAB_C

SIMPLEX_A_R = SIMPLEX_A_REAL
SIMPLEX_B_R = SIMPLEX_B_REAL
SIMPLEX_A_C = SIMPLEX_A_COMPLEX
SIMPLEX_B_C = SIMPLEX_B_COMPLEX
