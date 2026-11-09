#!/usr/bin/env bash
set -e

echo "=================================================================="
echo "S6*-Adjoint: Unified Master Reproduction & Verification Suite"
echo "=================================================================="

echo ""
echo ">> [1/7] Installing s6_adjoint directly from source (editable mode)..."
pip install -e .

echo ""
echo ">> [2/7] Running Core Matrix Audit & Numerical Tiers..."
python tests/test_s6_core.py

echo ""
echo ">> [3/7] Running PyTorch Gradcheck & Exact Autograd Equivalence..."
python tests/test_torch_gradcheck.py

echo ""
echo ">> [4/7] Running Hamiltonian Neural Network (HNN) Example..."
python example_hnn.py

echo ""
echo ">> [5/7] Executing Dual-Arena Scientific Splitting Master Suites..."
python benchmarks/module1_matrix_audit.py
python benchmarks/module3_parabolic_diffusion.py

echo ""
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo ">> [6/7] Running GPU VRAM Memory Scaling Benchmark on CUDA..."
    python benchmark_gpu.py --device cuda
    echo ""
    echo ">> [7/7] Running Diff-MD Long-Horizon Trajectory Optimization Benchmark on CUDA..."
    python benchmarks/diff_md_long_horizon.py --device cuda
else
    echo ">> [6/7] CUDA GPU not detected; running Scaling Benchmark on CPU..."
    python benchmark_gpu.py --device cpu
    echo ""
    echo ">> [7/7] Running Diff-MD Long-Horizon Trajectory Optimization Benchmark on CPU..."
    python benchmarks/diff_md_long_horizon.py --device cpu --num_particles 16 --num_steps 30 --epochs 10
fi

echo ""
echo "=================================================================="
echo ">> All Master S6*-Adjoint test suites & Diff-MD completed successfully!"
echo "=================================================================="
