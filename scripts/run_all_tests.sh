#!/usr/bin/env bash
set -e
export PYTHONPATH=.
echo "Running S6* Scientific Verification Suite (Release v16.3)..."
python3 tests/test_all_7_members.py
python3 tests/test_coefficients.py
python3 tests/test_simplex_duality.py
python3 tests/test_simplex_adjoint.py
python3 tests/test_krylov_estimator.py
python3 tests/test_torch_gradcheck.py
echo "All tests passed successfully!"
