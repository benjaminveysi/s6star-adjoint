# S6*-Adjoint: Sixth-Order Differentiable Splitting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Differentiable adjoint for the **S6\* 13-substep geometric splitting family** — train Hamiltonian systems and general vector fields in **O(1) memory** at **6th order**, with a single backward pass.

Designed for training Neural ODEs and differentiable physics where backprop through the solver either eats all your memory or quietly loses the order.

## Why S6*-Adjoint?

- **One backward pass, O(1) memory:** The adjoint rebuilds the trajectory backward from the final state instead of storing every step. Peak VRAM stays flat at `12.13 MiB` from 20 to 400 steps.
- **Adjoint-H for Hamiltonians:** `H(q,p) = T(p) + V_θ(q)`. Exact 6th-order reversible adjoint for separable Hamiltonians, checked against a fully unrolled autograd graph.
- **Adjoint-G for general vector fields:** `ż = F_θ(z)` with no scalar potential — for one-way, asymmetric couplings where `curl ≠ 0`. The double commutator is a **JVP**, `G2 = (0, −2·J_F·F_θ)`: using the VJP (the old code's mistake) crushes the order from 6 to exactly 2 on rotational fields.
- **Reversible by construction:** Forward–backward error sits at `1.1e-15`, the double-precision floor. The backward pass is deliberately not differentiable — nesting adjoints blows up to K-th order, so one clean pass is the contract.

## Verified numbers

**1. Adjoint-G on a rotational field (`curl = 2.78` at the probe point)**
- Order with the JVP commutator: **6.01 / 6.00**; with the VJP (old code): **1.98 / 1.99** — the VJP trap.
- Reversibility: **1.11e-15** (h = 0.05).
- Gradient check vs. unrolled autograd: max relative diff **6.5e-10**.

**2. Adjoint-G vs. unrolled autograd on CUDA (RTX Pro 6000)**
- VJP difference: **5.68e-14**.
- Peak VRAM flat at **12.13 MiB** for 20 / 100 / 400 steps — O(1) in step count.

## Quickstart

```bash
git clone https://github.com/benjaminveysi/s6star-adjoint.git
cd s6star-adjoint
pip install -r requirements.txt

# Train a Hamiltonian neural network (O(1) memory)
python example_hnn.py --scheme real_standard_33

# Run the checks
python -m pytest tests/test_torch_gradcheck.py -v   # gradcheck + reversibility
python -m pytest tests/test_simplex_adjoint.py -v   # simplex adjoint
```
