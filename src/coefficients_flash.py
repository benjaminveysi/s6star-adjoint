"""
================================================================================
S6* COMPLEX-FLASH: HIGH-THROUGHPUT 6TH-ORDER GEOMETRIC SPLITTING ENGINE
================================================================================
Architecture:
    Forward-Positive Complex Commutator-Corrected Splitting (S6* Flash)
    Reduced FFT Footprint: 22 FFTs/macro-step (vs. 32 actual / 52 nominal baseline)
    Sectorial Bound: max |arg(a_j)| <= 18.38 degrees, max |arg(b_j)| <= 18.06 degrees
    Midpoint Subflow Evaluation for High-Throughput Smooth Flows

Mathematical Foundation:
    S_6^Flash(h) = B_1 A_1 B_2 A_2 B_3 A_3 B_4 A_3 B_3 A_2 B_2 A_1 B_1
    where:
        A_j = exp(h * a_j * Delta)
        B_j = exp(h * b_j * V + h^3 * c_j * [V, [T, V]])
    In Flash mode, active commutator substeps (c_j != 0) are integrated via a 
    symmetric 3-substep midpoint formulation:
        v <- Bernoulli(v, 0.5 * b_j * h)
        g_2 <- [V, [T, V]](v)
        v <- v + c_j * h^3 * g_2
        v <- Bernoulli(v, 0.5 * b_j * h)
    yielding an exact 6th-order global asymptotic scaling while slashing 
    commutator evaluations by 50% (from 2 calls to 1 call per active stage).
================================================================================
"""

import numpy as np

# Canonical Forward-Positive Coefficients (Root 11, c1 = c7 = 0)
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

FLASH_COMPLEX_A = [a1_c, a2_c, a3_c, a3_c, a2_c, a1_c]
FLASH_COMPLEX_B = [b1_c, b2_c, b3_c, b4_c, b3_c, b2_c, b1_c]
FLASH_COMPLEX_C = [c1_c, c2_c, c3_c, c4_c, c3_c, c2_c, c1_c]

# FFT accounting per macro-step:
# 6 kinetic stages * 2 = 12 FFTs
# 5 active commutator stages * 2 (1 call to compute_g2) = 10 FFTs
# Total = 22 FFTs
FLASH_FFTS_PER_STEP = 22
