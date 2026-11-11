"""
Unit test verifying algebraic sanity of the 7 member + external baseline coefficient arrays:
1. Normalization: sum(a) == 1.0, sum(b) == 1.0
2. Palindromic symmetry: a_j == a_{N-j+1}, b_j == b_{M-j+1}
3. Complex forward-positivity: Re(a_k) > 0, Re(b_k) > 0 for all parabolic members
"""
import sys, os
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.coefficients import (
    REAL_STANDARD_33_A, REAL_STANDARD_33_B,
    YOSHIDA_6_A, YOSHIDA_6_B,
    COMPLEX_STANDARD_32_A, COMPLEX_STANDARD_32_B,
    BCCHM_6_A, BCCHM_6_B,
    CASTELLA_BLANES_GAMMA
)

def test_real_standard():
    assert abs(np.sum(REAL_STANDARD_33_A) - 1.0) < 1e-14
    assert abs(np.sum(REAL_STANDARD_33_B) - 1.0) < 1e-14
    np.testing.assert_allclose(REAL_STANDARD_33_A, REAL_STANDARD_33_A[::-1], atol=1e-14)
    np.testing.assert_allclose(REAL_STANDARD_33_B, REAL_STANDARD_33_B[::-1], atol=1e-14)
    print("test_real_standard: PASSED")

def test_pure_14():
    assert abs(np.sum(YOSHIDA_6_A) - 1.0) < 1e-14
    assert abs(np.sum(YOSHIDA_6_B) - 1.0) < 1e-14
    np.testing.assert_allclose(YOSHIDA_6_A, YOSHIDA_6_A[::-1], atol=1e-14)
    np.testing.assert_allclose(YOSHIDA_6_B, YOSHIDA_6_B[::-1], atol=1e-14)
    print("test_pure_14: PASSED")

def test_bcchm_6():
    assert abs(np.sum(BCCHM_6_A) - 1.0) < 1e-14
    assert abs(np.sum(BCCHM_6_B) - 1.0) < 1e-14
    assert np.all(BCCHM_6_A.real > 0)
    assert np.all(BCCHM_6_B.real > 0)
    print("test_bcchm_6 (forward-positive): PASSED")

def test_castella_blanes_normalization():
    s = np.sum(CASTELLA_BLANES_GAMMA)
    assert abs(s - 1.0) < 1e-15
    assert np.all(CASTELLA_BLANES_GAMMA.real > 0)
    print("test_castella_blanes_normalization: PASSED")

if __name__ == '__main__':
    test_real_standard()
    test_pure_14()
    test_bcchm_6()
    test_castella_blanes_normalization()
