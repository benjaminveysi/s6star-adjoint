#!/usr/bin/env bash
set -e
export PYTHONPATH=.
echo "Executing Module 1: Matrix Audit..."
python3 benchmarks/module1_matrix_audit.py
echo "Executing Module 2: Soliton GPE..."
python3 benchmarks/module2_soliton_gpe.py
echo "Executing Module 3: Parabolic Diffusion..."
python3 benchmarks/module3_parabolic_diffusion.py
echo "Executing Module 4: Grand Suite..."
python3 benchmarks/module4_grand_suite.py
echo "Executing Module 5: Complex Flash Suite..."
python3 benchmarks/module5_complex_flash_suite.py
echo "Executing Module 6: Baselines Grand Suite..."
python3 benchmarks/module6_baselines_grand_suite.py
echo "Executing Module 7: Real Pareto Suite..."
python3 benchmarks/module7_real_pareto_suite.py
echo "Executing Module 8: Simplex Grand Suite..."
python3 benchmarks/module8_simplex_grand_suite.py
echo "All 8 benchmark modules completed successfully!"
