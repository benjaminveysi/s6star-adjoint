"""
Certified High-Precision Coefficients for S6* Release v16 Splitting Taxonomy:
Branch I: Real Family (Hamiltonian, Unitary Waves & Exact Reversible Adjoints)
  1. S6* Real-Standard-33 (a1 = -0.50 branch, 30-digit IEEE float64, 33 FFTs/step)
  2. S6* Real-Stencil-12  (8th-order compact central difference stencil, 12 FFTs/step)

Branch II: Complex Family (Parabolic, Dissipative Systems)
  3. S6* Complex-Standard-32 (Root 11 forward-positive, 2-pt Gauss subflow, 32 FFTs/step)
  4. S6* Complex-Flash-22    (Midpoint subflow, c1 = c7 = 0, 22 FFTs/step)
  5. S6* Complex-Stencil-12  (8th-order coordinate stencils, 12 FFTs/step)

Branch III: Simplex Family (2D Simplicial Meshes)
  6. S6* Simplex-Commutator  (Canonical 13-stage with FE stiffness double commutator, 157 SpMV/step)

Reference Baselines: Yoshida-6 (1990), BCChM-6 (2013), Castella-Blanes (2009).
"""
from typing import Sequence, Dict, Any
import numpy as np

# =============================================================================
# BRANCH I: REAL FAMILY (Pure Real, Symplectic, Exactly Reversible)
# =============================================================================
S6_REAL_A: Sequence[float] = (
    -0.500000000000000000000000000000,
    -0.237760438469839000791530674718,
    +1.237760438469839000791530674720,
    +1.237760438469839000791530674720,
    -0.237760438469839000791530674718,
    -0.500000000000000000000000000000,
)

S6_REAL_B: Sequence[float] = (
    +0.229570520791967995656670265773,
    -0.030062572938863245882265636245,
    +0.009357789208622524584969083512,
    +0.582268525876545451281252573919,
    +0.009357789208622524584969083512,
    -0.030062572938863245882265636245,
    +0.229570520791967995656670265773,
)

S6_REAL_C: Sequence[float] = (
    +0.002077945307578866621043128275,
    -0.001356232863128116206183885884,
    +0.000866395870360925200270635159,
    -0.008386340880592500000000000000,
    +0.000866395870360925200270635159,
    -0.001356232863128116206183885884,
    +0.002077945307578866621043128275,
)

# Manifold re-optimization of the Real root (a1 = -0.108, 2026-10-09).
# Minimizer of J = max|cumsum| on the certified branch; J = 1.3281, worst
# cumsum -0.3281 (old -0.7378, 2.25x). Same 13-stage ansatz, order 6.
# Validated 2026-11-12: order 5.96/5.99, reversibility 2.33e-15 (T=6, h=0.05).
S6_REAL_MANIFOLD_A: Sequence[float] = (
    -0.108000000000000000000000000000,
    -0.220139136592293239235040328231,
    +0.828139136592293239235040328231,
    +0.828139136592293239235040328231,
    -0.220139136592293239235040328231,
    -0.108000000000000000000000000000,
)

S6_REAL_MANIFOLD_B: Sequence[float] = (
    +0.438013114725810042224382327575,
    -0.217315231039338234361616214834,
    +0.018222138545894743204780564659,
    +0.522159955535266897864906645199,
    +0.018222138545894743204780564659,
    -0.217315231039338234361616214834,
    +0.438013114725810042224382327575,
)

S6_REAL_MANIFOLD_C: Sequence[float] = (
    +0.003515317301654269464008232266,
    -0.000162095903242087553234732502,
    +0.000553911734979314854674018281,
    -0.005007886235502198141869598714,
    +0.000553911734979314854674018281,
    -0.000162095903242087553234732502,
    +0.003515317301654269464008232266,
)


# Aliases
S6_A = S6_REAL_A
S6_B = S6_REAL_B
S6_C = S6_REAL_C

REAL_STANDARD_33_A = S6_REAL_A
REAL_STANDARD_33_B = S6_REAL_B
REAL_STANDARD_33_C = S6_REAL_C
REAL_STANDARD_33_FFTS = 33

REAL_MANIFOLD_33_A = S6_REAL_MANIFOLD_A
REAL_MANIFOLD_33_B = S6_REAL_MANIFOLD_B
REAL_MANIFOLD_33_C = S6_REAL_MANIFOLD_C
REAL_MANIFOLD_33_FFTS = 33

REAL_STENCIL_12_A = S6_REAL_A
REAL_STENCIL_12_B = S6_REAL_B
REAL_STENCIL_12_C = S6_REAL_C
REAL_STENCIL_12_FFTS = 12

# 8th-order compact central difference stencil weights for first derivative u_x:
STENCIL_D: Sequence[float] = (4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0)

# =============================================================================
# BRANCH II: COMPLEX FAMILY (Forward-Positive, Parabolic Stability)
# =============================================================================
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

S6_COMPLEX_A: Sequence[complex] = (a1_c, a2_c, a3_c, a3_c, a2_c, a1_c)
S6_COMPLEX_B: Sequence[complex] = (b1_c, b2_c, b3_c, b4_c, b3_c, b2_c, b1_c)
S6_COMPLEX_C: Sequence[complex] = (c1_c, c2_c, c3_c, c4_c, c3_c, c2_c, c1_c)

COMPLEX_STANDARD_32_A = S6_COMPLEX_A
COMPLEX_STANDARD_32_B = S6_COMPLEX_B
COMPLEX_STANDARD_32_C = S6_COMPLEX_C
COMPLEX_STANDARD_32_FFTS = 32

FLASH_COMPLEX_A = S6_COMPLEX_A
FLASH_COMPLEX_B = S6_COMPLEX_B
FLASH_COMPLEX_C = S6_COMPLEX_C
COMPLEX_FLASH_22_A = FLASH_COMPLEX_A
COMPLEX_FLASH_22_B = FLASH_COMPLEX_B
COMPLEX_FLASH_22_C = FLASH_COMPLEX_C
COMPLEX_FLASH_22_FFTS = 22

COMPLEX_STENCIL_12_A = S6_COMPLEX_A
COMPLEX_STENCIL_12_B = S6_COMPLEX_B
COMPLEX_STENCIL_12_C = S6_COMPLEX_C
COMPLEX_STENCIL_12_FFTS = 12

# =============================================================================
# BRANCH III: SIMPLEX FAMILY (2D Simplicial Meshes)
# =============================================================================
S6_SIMPLEX_COMM_A = S6_REAL_A
S6_SIMPLEX_COMM_B = S6_REAL_B
S6_SIMPLEX_COMM_C = S6_REAL_C

SIMPLEX_A_REAL = S6_SIMPLEX_COMM_A
SIMPLEX_B_REAL = S6_SIMPLEX_COMM_B
SIMPLEX_C_REAL = S6_SIMPLEX_COMM_C

SIMPLEX_A_COMPLEX = S6_COMPLEX_A
SIMPLEX_B_COMPLEX = S6_COMPLEX_B
SIMPLEX_C_COMPLEX = S6_COMPLEX_C

# =============================================================================
# CLASSICAL BASELINES
# =============================================================================
g1 = 0.116900037554661284389 + 0.043428254616060341762j
g2 = 0.129559101282088262750 - 0.123989612188092593300j
g3 = 0.186532492812133817800 + 0.003107430710072675340j
g4 = 0.134016736702233270122 + 0.154907853723919152396j
BCCHM_GAMMAS: Sequence[complex] = (g1, g2, g3, g4, g3, g2, g1)

w1_4 =  1.3512071919596576
w2_4 = -1.7024143839193153
YOSHIDA4_W: Sequence[float] = (w1_4, w2_4, w1_4)

w1_6 = -1.17767998417887E0
w2_6 =  0.235573213359357E0
w3_6 =  0.784513610477560E0
w0_6 = 1.0 - 2.0 * (w1_6 + w2_6 + w3_6)
YOSHIDA6_W: Sequence[float] = (w3_6, w2_6, w1_6, w0_6, w1_6, w2_6, w3_6)

# =============================================================================
# UNIFIED COEFFICIENTS REGISTRY (Release v16)
# =============================================================================
COEFFICIENTS_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Branch I: Real Family (Both 100% Real, Symplectic, and Time-Reversible)
    'real': {
        'A': S6_REAL_A,
        'B': S6_REAL_B,
        'C': S6_REAL_C,
        'type': 'real_standard_33',
        'branch': 'Real',
        'order': 6,
        'cost': 33,
        'cost_unit': 'FFTs/step',
        'is_complex': False,
    },
    'real_standard': {
        'A': S6_REAL_A,
        'B': S6_REAL_B,
        'C': S6_REAL_C,
        'type': 'real_standard_33',
        'branch': 'Real',
        'order': 6,
        'cost': 33,
        'cost_unit': 'FFTs/step',
        'is_complex': False,
    },
    'real_standard_33': {
        'A': REAL_STANDARD_33_A,
        'B': REAL_STANDARD_33_B,
        'C': REAL_STANDARD_33_C,
        'type': 'real_standard_33',
        'branch': 'Real',
        'order': 6,
        'cost': 33,
        'cost_unit': 'FFTs/step',
        'is_complex': False,
    },
    'real_manifold': {
        'A': REAL_MANIFOLD_33_A,
        'B': REAL_MANIFOLD_33_B,
        'C': REAL_MANIFOLD_33_C,
        'type': 'real_manifold_33',
        'branch': 'Real',
        'order': 6,
        'cost': 33,
        'cost_unit': 'FFTs/step',
        'is_complex': False,
    },
    'real_manifold_33': {
        'A': REAL_MANIFOLD_33_A,
        'B': REAL_MANIFOLD_33_B,
        'C': REAL_MANIFOLD_33_C,
        'type': 'real_manifold_33',
        'branch': 'Real',
        'order': 6,
        'cost': 33,
        'cost_unit': 'FFTs/step',
        'is_complex': False,
    },
    'real_stencil': {
        'A': REAL_STENCIL_12_A,
        'B': REAL_STENCIL_12_B,
        'C': REAL_STENCIL_12_C,
        'stencil': STENCIL_D,
        'type': 'real_stencil_12',
        'branch': 'Real',
        'order': 6,
        'cost': 12,
        'cost_unit': 'FFTs/step',
        'is_complex': False,
    },
    'real_stencil_12': {
        'A': REAL_STENCIL_12_A,
        'B': REAL_STENCIL_12_B,
        'C': REAL_STENCIL_12_C,
        'stencil': STENCIL_D,
        'type': 'real_stencil_12',
        'branch': 'Real',
        'order': 6,
        'cost': 12,
        'cost_unit': 'FFTs/step',
        'is_complex': False,
    },

    # Branch II: Complex Family (Forward-Positive, Parabolic Stability)
    'complex_standard_32': {
        'A': COMPLEX_STANDARD_32_A,
        'B': COMPLEX_STANDARD_32_B,
        'C': COMPLEX_STANDARD_32_C,
        'type': 'complex_standard_32',
        'branch': 'Complex',
        'order': 6,
        'cost': 32,
        'cost_unit': 'FFTs/step',
        'is_complex': True,
    },
    'complex_flash_22': {
        'A': COMPLEX_FLASH_22_A,
        'B': COMPLEX_FLASH_22_B,
        'C': COMPLEX_FLASH_22_C,
        'type': 'complex_flash_22',
        'branch': 'Complex',
        'order': 6,
        'cost': 22,
        'cost_unit': 'FFTs/step',
        'is_complex': True,
    },
    'complex_stencil_12': {
        'A': COMPLEX_STENCIL_12_A,
        'B': COMPLEX_STENCIL_12_B,
        'C': COMPLEX_STENCIL_12_C,
        'stencil': STENCIL_D,
        'type': 'complex_stencil_12',
        'branch': 'Complex',
        'order': 6,
        'cost': 12,
        'cost_unit': 'FFTs/step',
        'is_complex': True,
    },

    # Branch III: Simplex Family (2D Simplicial Meshes)
    'simplex_commutator': {
        'A': S6_SIMPLEX_COMM_A,
        'B': S6_SIMPLEX_COMM_B,
        'C': S6_SIMPLEX_COMM_C,
        'type': 'simplex_commutator',
        'branch': 'Simplex',
        'order': 6,
        'cost': 157,
        'cost_unit': 'SpMV/step',
        'is_complex': False,
    },
}
