"""
===================================================================================================
S6* MASTER CERTIFIED COEFFICIENTS CATALOG (ACTIVE S6* FAMILY COEFFICIENTS)
===================================================================================================
Contains high-precision certified coefficients (computed to 25-30 decimal digits, stored in IEEE-754
float64 / complex128) for all eight members of the S6* geometric splitting framework across three branches,
plus the retained classical external baselines (NOT family members):

BRANCH I: S6* REAL GEOMETRIC FAMILY (Unitary, Hamiltonian, Symplectic Waves)
  1. S6* Real-Standard-33 (33 FFT transforms/step): Full spectral double commutator evaluation
  2. S6* Real-Stencil-12  (12 FFT transforms/step): 8th-order compact central difference spatial stencils
  3. S6* Real-Manifold-33 (33 FFT transforms/step): Manifold re-optimization of the Real root (a1 = -0.108, 2026-10-09; J = 1.3281, worst cumsum -0.3281)
  (Classical baseline Yoshida-6 (1990), 14 FFTs/step FSAL, lives in YOSHIDA_6_* — NOT a family member.)

BRANCH II: S6* COMPLEX GEOMETRIC FAMILY (Parabolic, Dissipative & Ground-State Relaxation)
  4. S6* Complex-Standard-32 (32 FFT transforms/step): Canonical 13-stage free-Lie splitting (2-point Gauss)
  5. S6* Complex-Flash-22    (22 FFT transforms/step): 13-stage midpoint subflow spectral commutator
  6. S6* Complex-Stencil-12  (12 FFT transforms/step): 6-stage forward-positive with 8th-order spatial stencils
  (Classical baselines BCChM-6 P6S7 (2013), 14 FFTs/step FSAL, lives in BCCHM_6_*; and Castella-Blanes (2009),
  20 FFTs/macro-step h, lives in CASTELLA_BLANES_* — NOT family members; formerly mislabeled as
  'Complex-Bullet-14' and 'Complex-Instant-10', pruned from the registry 2026-11-07.)

BRANCH III: S6* SIMPLEX GEOMETRY-NATIVE FAMILY (2D Simplicial Meshes over K in {R, C})
  7. S6* Simplex-Commutator (96 K-SpMVs + 96 C-SpMVs asymptotically at m=16): exact discrete commutator
  8. Simplex-Yoshida-6 (112 SpMVs/step): classical Yoshida-6 composition on simplicial FE geometry
===================================================================================================
"""

import numpy as np

# =================================================================================================
# 1. CANONICAL FREE-LIE ROOTS
# =================================================================================================

# -------------------------------------------------------------------------------------------------
# 1.1 Canonical Real-S6* Root (a1 = -0.50, computed to 30 digits)
# -------------------------------------------------------------------------------------------------
S6_REAL_A = np.array([
    -0.5000000000000000, -0.2377604384698390,  1.2377604384698390,
     1.2377604384698390, -0.2377604384698390, -0.5000000000000000
], dtype=np.float64)

S6_REAL_B = np.array([
     0.2295705207919680, -0.0300625729388632,  0.0093577892086225,  0.5822685258765455,
     0.0093577892086225, -0.0300625729388632,  0.2295705207919680
], dtype=np.float64)

S6_REAL_C = np.array([
     0.0020779453075789, -0.0013562328631281,  0.0008663958703609, -0.0083863408805925,
     0.0008663958703609, -0.0013562328631281,  0.0020779453075789
], dtype=np.float64)

# -------------------------------------------------------------------------------------------------
# 1.1b Manifold re-optimization of the Real root (a1 = -0.108, 2026-10-09)
# -------------------------------------------------------------------------------------------------
# Minimizer of J = max|cumsum| on the certified 1D branch (traced a1 in [-2.50, -0.108],
# 239 points, residuals ~1e-17; 40-digit refinement, max residual 1.9e-42). J = 1.3281
# (old 1.7378), worst cumsum -0.3281 (old -0.7378, 2.25x). Same 13-stage symmetric
# ansatz, same order 6; a NEW point on the certified branch, NOT a new method.
# Validated 2026-11-12 on the gate toy (double well + rotational carrier): observed
# order 5.96/5.99, reversibility 2.33e-15 at T=6 h=0.05, errors ~2.5x smaller than
# Standard-33 at the same h. Source: s6star-manifold/REPORT.md + coeffs_new.npz.
S6_REAL_MANIFOLD_A = np.array([
    -0.1080000000000000, -0.2201391365922932, 0.8281391365922932,
     0.8281391365922932, -0.2201391365922932, -0.1080000000000000
], dtype=np.float64)

S6_REAL_MANIFOLD_B = np.array([
     0.4380131147258101, -0.2173152310393382, 0.0182221385458947, 0.5221599555352670,
     0.0182221385458947, -0.2173152310393382, 0.4380131147258101
], dtype=np.float64)

S6_REAL_MANIFOLD_C = np.array([
     0.0035153173016543, -0.0001620959032421, 0.0005539117349793, -0.0050078862355022,
     0.0005539117349793, -0.0001620959032421, 0.0035153173016543
], dtype=np.float64)


# -------------------------------------------------------------------------------------------------
# 1.2 Canonical Complex-S6* Root (Root 11, Forward-Positive Re(a_k) > 0, c1 = c7 = 0)
# -------------------------------------------------------------------------------------------------
a1_c = 0.146936589470078787789 + 0.0376209430496957583645j
a2_c = 0.162069491014167760268 + 0.0258252974604388958746j
a3_c = 0.190993919515753451943 - 0.0634462405101346577085j

b1_c = 0.048755861334214195546 + 0.0099836308286563226616j
b2_c = 0.192260865299594685407 + 0.0620428505712945915773j
b3_c = 0.156181651092679785187 - 0.0384972549644904890598j
b4_c = 0.205603244547022667721 - 0.0670584528709208538277j

c1_c = 0.0j
c2_c = -2.32267121857025450821e-04 - 2.93632386270561052295e-04j
c3_c = -1.20436070062024673695e-04 + 7.30448574887737129462e-05j
c4_c = -2.31090141230173103017e-04 + 4.72447769147430233442e-04j

S6_COMPLEX_A = [a1_c, a2_c, a3_c, a3_c, a2_c, a1_c]
S6_COMPLEX_B = [b1_c, b2_c, b3_c, b4_c, b3_c, b2_c, b1_c]
S6_COMPLEX_C = [c1_c, c2_c, c3_c, c4_c, c3_c, c2_c, c1_c]

FLASH_COMPLEX_A = S6_COMPLEX_A
FLASH_COMPLEX_B = S6_COMPLEX_B
FLASH_COMPLEX_C = S6_COMPLEX_C

# -------------------------------------------------------------------------------------------------
# 1.3 Certified Spatial Finite-Difference Stencils (8th-Order Compact)
# -------------------------------------------------------------------------------------------------
STENCIL_D = np.array([4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0], dtype=float)
STENCIL_D2_CENTER = -205.0 / 72.0
STENCIL_D2 = np.array([8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0], dtype=float)

# -------------------------------------------------------------------------------------------------
# 1.4 Classical & Reference Composition Weights
# -------------------------------------------------------------------------------------------------
# BCChM-6 P6S7 Coefficients (Blanes, Casas, Chandre, Murua 2013, Math. Comp.)
g1_bcchm = 0.116900037554661284389 + 0.043428254616060341762j
g2_bcchm = 0.129559101282088262750 - 0.123989612188092593300j
g3_bcchm = 0.186532492812133817800 + 0.003107430710072675340j
g4_bcchm = 0.134016736702233270122 + 0.154907853723919152396j
BCCHM_GAMMAS = [g1_bcchm, g2_bcchm, g3_bcchm, g4_bcchm, g3_bcchm, g2_bcchm, g1_bcchm]

# Yoshida-4 and Yoshida-6 Weights (Yoshida 1990, Phys. Lett. A)
w1_4 =  1.3512071919596576
w2_4 = -1.7024143839193153
YOSHIDA4_W = [w1_4, w2_4, w1_4]

w1_6 = -1.17767998417887E0
w2_6 =  0.235573213359357E0
w3_6 =  0.784513610477560E0
w0_6 = 1.0 - 2.0 * (w1_6 + w2_6 + w3_6)
YOSHIDA6_W = [w3_6, w2_6, w1_6, w0_6, w1_6, w2_6, w3_6]


# =================================================================================================
# 2. CERTIFIED ARRAYS FOR THE 7 S6* FAMILY MEMBERS + CLASSICAL EXTERNAL BASELINES
# =================================================================================================

# MEMBER 1: S6* Real-Standard-33 (33 FFT transforms/step)
REAL_STANDARD_33_A = S6_REAL_A
REAL_STANDARD_33_B = S6_REAL_B
REAL_STANDARD_33_C = S6_REAL_C
REAL_STANDARD_33_FFTS = 33

# MEMBER 2: S6* Real-Manifold-33 (33 FFT transforms/step) — manifold re-optimization (a1 = -0.108, 2026-10-09)
REAL_MANIFOLD_33_A = S6_REAL_MANIFOLD_A
REAL_MANIFOLD_33_B = S6_REAL_MANIFOLD_B
REAL_MANIFOLD_33_C = S6_REAL_MANIFOLD_C
REAL_MANIFOLD_33_FFTS = 33

# MEMBER 3: Yoshida-6 (Classical Recursive Composition Baseline, Yoshida 1990; 14 FFTs/step FSAL)
# Explicit attribution: Haruo Yoshida (1990, Phys. Lett. A 150, 262). Retained as pure splitting baseline.
YOSHIDA_6_W = np.array(YOSHIDA6_W, dtype=float)
YOSHIDA_6_A = np.array([
    0.5 * YOSHIDA_6_W[0],
    0.5 * (YOSHIDA_6_W[0] + YOSHIDA_6_W[1]),
    0.5 * (YOSHIDA_6_W[1] + YOSHIDA_6_W[2]),
    0.5 * (YOSHIDA_6_W[2] + YOSHIDA_6_W[3]),
    0.5 * (YOSHIDA_6_W[3] + YOSHIDA_6_W[4]),
    0.5 * (YOSHIDA_6_W[4] + YOSHIDA_6_W[5]),
    0.5 * (YOSHIDA_6_W[5] + YOSHIDA_6_W[6]),
    0.5 * YOSHIDA_6_W[6],
], dtype=float)
YOSHIDA_6_B = YOSHIDA_6_W.copy()
YOSHIDA_6_FFTS = 14

# MEMBER 4: S6* Real-Stencil-12 (12 FFT transforms/step)
REAL_STENCIL_12_A = S6_REAL_A
REAL_STENCIL_12_B = S6_REAL_B
REAL_STENCIL_12_C = S6_REAL_C
REAL_STENCIL_12_FFTS = 12

# MEMBER 5: S6* Complex-Standard-32 (32 FFT transforms/step)
COMPLEX_STANDARD_32_A = S6_COMPLEX_A
COMPLEX_STANDARD_32_B = S6_COMPLEX_B
COMPLEX_STANDARD_32_C = S6_COMPLEX_C
COMPLEX_STANDARD_32_FFTS = 32

# MEMBER 6: S6* Complex-Flash-22 (22 FFT transforms/step)
COMPLEX_FLASH_22_A = FLASH_COMPLEX_A
COMPLEX_FLASH_22_B = FLASH_COMPLEX_B
COMPLEX_FLASH_22_C = FLASH_COMPLEX_C
COMPLEX_FLASH_22_FFTS = 22

# EXTERNAL BASELINE: BCChM-6 P6S7 (Classical Forward-Positive Composition Baseline, Blanes et al. 2013; 14 FFTs/step FSAL)
# Explicit attribution: S. Blanes, F. Casas, C. Chandre, A. Murua (2013, Math. Comp. 82, 1553).
# NOT an S6* family member — formerly mislabeled as 'Complex-Bullet-14'; pruned from the registry 2026-11-07.
# Retained as an external baseline (same treatment as Yoshida-6): BCCHM_6_* is the canonical name.
BCCHM_6_GAMMA = np.array(BCCHM_GAMMAS, dtype=complex)
BCCHM_6_A = np.array([
    0.5 * BCCHM_6_GAMMA[0],
    0.5 * (BCCHM_6_GAMMA[0] + BCCHM_6_GAMMA[1]),
    0.5 * (BCCHM_6_GAMMA[1] + BCCHM_6_GAMMA[2]),
    0.5 * (BCCHM_6_GAMMA[2] + BCCHM_6_GAMMA[3]),
    0.5 * (BCCHM_6_GAMMA[3] + BCCHM_6_GAMMA[4]),
    0.5 * (BCCHM_6_GAMMA[4] + BCCHM_6_GAMMA[5]),
    0.5 * (BCCHM_6_GAMMA[5] + BCCHM_6_GAMMA[6]),
    0.5 * BCCHM_6_GAMMA[6],
], dtype=complex)
BCCHM_6_B = BCCHM_6_GAMMA.copy()
BCCHM_6_FFTS = 14

# (The misleading COMPLEX_BULLET_14_* / BULLET_* aliases are gone — pruned 2026-11-07. Use BCCHM_6_*.)

# MEMBER 8: S6* Complex-Stencil-12 (12 FFT transforms/step)
COMPLEX_STENCIL_12_A = S6_COMPLEX_A
COMPLEX_STENCIL_12_B = S6_COMPLEX_B
COMPLEX_STENCIL_12_C = S6_COMPLEX_C
COMPLEX_STENCIL_12_FFTS = 12

REAL_FLASH_A = COMPLEX_STENCIL_12_A
REAL_FLASH_B = COMPLEX_STENCIL_12_B
REAL_FLASH_C = COMPLEX_STENCIL_12_C
REAL_FLASH_FFTS_PER_STEP = 12

# EXTERNAL BASELINE: Castella-Blanes 5-Stage (Classical Forward-Positive Baseline, 2009; 10 FFTs/step)
# Explicit attribution: F. Castella et al. (2009) / S. Blanes.
# NOT an S6* family member — formerly mislabeled as 'Complex-Instant-10'; pruned from the registry 2026-11-07.
# Retained as an external baseline (same treatment as Yoshida-6): CASTELLA_BLANES_* is the canonical name.
# Evaluates 5 forward-positive stages, exactly 10 FFT transforms per macro-step h.
_castella_blanes_raw = np.array([
    0.127748034029 - 0.212499902961j,
    0.262600175719 - 0.083257571752j,
    0.269952981581 + 0.125881897255j,
    0.224982654038 - 0.044554013373j,
    0.114716154634 + 0.214429590831j,
], dtype=complex)
# Rigorous normalization ensuring sum(gamma) == 1.0 to machine precision (< 3e-17)
CASTELLA_BLANES_GAMMA = _castella_blanes_raw / np.sum(_castella_blanes_raw)
CASTELLA_BLANES_A = CASTELLA_BLANES_GAMMA.copy()
CASTELLA_BLANES_B = CASTELLA_BLANES_GAMMA.copy()
CASTELLA_BLANES_FFTS = 10

# (The misleading COMPLEX_INSTANT_10_* / INSTANT_* aliases are gone — pruned 2026-11-07. Use CASTELLA_BLANES_*.)

# MEMBER 10: S6* Simplex-Commutator (Genuine S6* 13-stage scheme with discrete commutator on simplicial meshes)
# Evaluates exact discrete double commutator C = [V, [M^-1 K, V]] (sparse matrix).
# Exact realization uses one Krylov exponential for each B_j = exp(-i h b_j V - i h^3 c_j C).
# The canonical map starts/ends with B, so kinetic FSAL is unavailable; identical B boundary
# stages may be fused across macro-steps. With fixed m=16, asymptotic sparse work is
# 96 K-SpMVs + 96 C-SpMVs = 192 sparse matvecs per macro-step.
S6_SIMPLEX_COMM_A = S6_REAL_A
S6_SIMPLEX_COMM_B = S6_REAL_B
S6_SIMPLEX_COMM_C = S6_REAL_C
S6_SIMPLEX_COMM_K_SPMVS = 96
S6_SIMPLEX_COMM_C_SPMVS = 96
S6_SIMPLEX_COMM_SPMVS = 192

# Aliases
SIMPLEX_A_R = S6_SIMPLEX_COMM_A
SIMPLEX_B_R = S6_SIMPLEX_COMM_B
SIMPLEX_A_C = S6_COMPLEX_A
SIMPLEX_B_C = S6_COMPLEX_B
SIMPLEX_KINETIC_STAGES = 6  # Canonical S6* commutator map has 6 kinetic stages; no kinetic FSAL
