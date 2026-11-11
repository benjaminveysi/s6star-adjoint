"""
Comprehensive PyTorch Autograd & Gradcheck Validation for S6HamiltonianFunction:
1. PyTorch gradcheck test against finite differences in float64.
2. Exact comparison against Naive Unrolled Autograd (ground truth computation graph).
"""
import sys
import os

try:
    import torch
    import torch.nn as nn
    from torch.autograd import gradcheck
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed in this environment. Test will be run on PyTorch-enabled environments.")

if TORCH_AVAILABLE:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from s6_adjoint import S6HamiltonianFunction, S6HamiltonianODE
    from s6_adjoint.core.coefficients import S6_REAL_A, S6_REAL_B, S6_REAL_C

    class SimpleParametricPotential(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.2, 0.8], dtype=torch.float64))
            self.b = nn.Parameter(torch.tensor([0.2], dtype=torch.float64))

        def forward(self, q):
            harmonic = 0.5 * torch.sum(self.w * (q ** 2), dim=-1)
            quartic = 0.1 * (torch.sum(q, dim=-1) + self.b) ** 4
            return harmonic + quartic

    def test_against_naive_unrolled_autograd():
        print("=== Test 1: S6*-Adjoint vs Naive Unrolled Computation Graph ===")
        potential_ref = SimpleParametricPotential()
        potential_adj = SimpleParametricPotential()
        potential_adj.load_state_dict(potential_ref.state_dict())

        q0_ref = torch.tensor([1.0, -0.5], dtype=torch.float64, requires_grad=True)
        p0_ref = torch.tensor([0.4, 0.7], dtype=torch.float64, requires_grad=True)

        q0_adj = q0_ref.detach().clone().requires_grad_(True)
        p0_adj = p0_ref.detach().clone().requires_grad_(True)

        h = 0.04
        num_steps = 5

        # 1. Naive Unrolled Forward with full computational graph
        q = q0_ref
        p = p0_ref
        for step in range(num_steps):
            for k in range(6):
                h_b = h * S6_REAL_B[k]
                h3_c = (h ** 3) * S6_REAL_C[k]
                V = potential_ref(q)
                F = -torch.autograd.grad(V.sum(), q, create_graph=True)[0]
                G = torch.autograd.grad(0.5 * torch.sum(F ** 2), q, create_graph=True)[0]
                p = p + h_b * F - 2.0 * h3_c * G
                q = q + h * S6_REAL_A[k] * p
            
            h_b = h * S6_REAL_B[6]
            h3_c = (h ** 3) * S6_REAL_C[6]
            V = potential_ref(q)
            F = -torch.autograd.grad(V.sum(), q, create_graph=True)[0]
            G = torch.autograd.grad(0.5 * torch.sum(F ** 2), q, create_graph=True)[0]
            p = p + h_b * F - 2.0 * h3_c * G

        loss_ref = torch.sum(q ** 2 + p ** 2)
        loss_ref.backward()

        # 2. S6*-Adjoint Forward & Backward (O(1) memory)
        ode = S6HamiltonianODE(potential_adj, scheme='real')
        q_T, p_T = ode(q0_adj, p0_adj, h, num_steps)
        loss_adj = torch.sum(q_T ** 2 + p_T ** 2)
        loss_adj.backward()

        # Compare State outputs
        q_diff = torch.norm(q.detach() - q_T.detach()).item()
        p_diff = torch.norm(p.detach() - p_T.detach()).item()
        print(f"Final State diff ||q_ref - q_adj||: {q_diff:.2e}")
        print(f"Final State diff ||p_ref - p_adj||: {p_diff:.2e}")

        # Compare Gradients
        grad_q0_diff = torch.norm(q0_ref.grad - q0_adj.grad).item() / torch.norm(q0_ref.grad).item()
        grad_p0_diff = torch.norm(p0_ref.grad - p0_adj.grad).item() / torch.norm(p0_ref.grad).item()
        grad_w_diff = torch.norm(potential_ref.w.grad - potential_adj.w.grad).item() / torch.norm(potential_ref.w.grad).item()
        grad_b_diff = torch.norm(potential_ref.b.grad - potential_adj.b.grad).item() / torch.norm(potential_ref.b.grad).item()

        print(f"Relative Grad diff q0: {grad_q0_diff:.2e}")
        print(f"Relative Grad diff p0: {grad_p0_diff:.2e}")
        print(f"Relative Grad diff w:  {grad_w_diff:.2e}")
        print(f"Relative Grad diff b:  {grad_b_diff:.2e}")

        assert grad_q0_diff < 1e-10, "q0 gradient mismatch!"
        assert grad_p0_diff < 1e-10, "p0 gradient mismatch!"
        assert grad_w_diff < 1e-10, "w gradient mismatch!"
        assert grad_b_diff < 1e-10, "b gradient mismatch!"
        print(">> PASS: S6*-Adjoint matches full unrolled graph to machine precision!")

    def test_torch_gradcheck():
        print("\n=== Test 2: PyTorch gradcheck against Finite Differences ===")
        potential = SimpleParametricPotential()
        q0 = torch.tensor([0.8, -0.4], dtype=torch.float64, requires_grad=True)
        p0 = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)
        h = 0.03
        num_steps = 3
        param_names = [name for name, _ in potential.named_parameters()]
        params = tuple(potential.parameters())

        test_inputs = (q0, p0, h, num_steps, 'real', potential, param_names, *params)
        passed = gradcheck(S6HamiltonianFunction.apply, test_inputs, eps=1e-6, atol=1e-4, rtol=1e-3)
        print(f">> PASS: PyTorch gradcheck status: {passed}")


    def test_output_cotangent_and_parameter_edge_cases():
        """Regression tests for unused outputs, parameterless nets, and frozen parameters."""
        # q-only and p-only losses must be valid: unused output cotangents arrive as None.
        for use_q, use_p in [(True, False), (False, True)]:
            pot = SimpleParametricPotential()
            q0 = torch.tensor([0.7, -0.2], dtype=torch.float64, requires_grad=True)
            p0 = torch.tensor([0.1, 0.3], dtype=torch.float64, requires_grad=True)
            q_T, p_T = S6HamiltonianODE(pot, scheme='real')(q0, p0, 0.02, 2)
            loss = (q_T.square().sum() if use_q else 0.0) + (p_T.square().sum() if use_p else 0.0)
            loss.backward()
            assert q0.grad is not None and p0.grad is not None
            assert pot.w.grad is not None and pot.b.grad is not None

        class FixedPotential(nn.Module):
            def forward(self, q):
                return 0.5 * torch.sum(q ** 2, dim=-1)

        q0 = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
        p0 = torch.tensor([0.2, 0.1], dtype=torch.float64, requires_grad=True)
        q_T, p_T = S6HamiltonianODE(FixedPotential())(q0, p0, 0.01, 2)
        (q_T.square().sum() + p_T.square().sum()).backward()
        assert q0.grad is not None and p0.grad is not None

        pot = SimpleParametricPotential()
        pot.w.requires_grad_(False)
        q0 = torch.tensor([0.4, -0.6], dtype=torch.float64, requires_grad=True)
        p0 = torch.tensor([0.2, 0.1], dtype=torch.float64, requires_grad=True)
        q_T, p_T = S6HamiltonianODE(pot)(q0, p0, 0.01, 2)
        (q_T.square().sum() + p_T.square().sum()).backward()
        assert pot.w.grad is None
        assert pot.b.grad is not None

    def test_mass_consistency_against_naive():
        """Mass scaling must match the canonical H = p^2/(2m) + V discretization."""
        mass = 2.5
        pot_ref = SimpleParametricPotential()
        pot_adj = SimpleParametricPotential()
        pot_adj.load_state_dict(pot_ref.state_dict())
        q_ref = torch.tensor([0.8, -0.3], dtype=torch.float64, requires_grad=True)
        p_ref = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
        q_adj = q_ref.detach().clone().requires_grad_(True)
        p_adj = p_ref.detach().clone().requires_grad_(True)
        h = 0.025
        steps = 3

        q, p = q_ref, p_ref
        for _ in range(steps):
            for k in range(6):
                hb = h * S6_REAL_B[k]
                hc = (h ** 3) * S6_REAL_C[k] / mass
                Fv = -torch.autograd.grad(pot_ref(q).sum(), q, create_graph=True)[0]
                Gv = torch.autograd.grad(0.5 * torch.sum(Fv ** 2), q, create_graph=True)[0]
                p = p + hb * Fv - 2.0 * hc * Gv
                q = q + (h * S6_REAL_A[k] / mass) * p
            hb = h * S6_REAL_B[6]
            hc = (h ** 3) * S6_REAL_C[6] / mass
            Fv = -torch.autograd.grad(pot_ref(q).sum(), q, create_graph=True)[0]
            Gv = torch.autograd.grad(0.5 * torch.sum(Fv ** 2), q, create_graph=True)[0]
            p = p + hb * Fv - 2.0 * hc * Gv

        q_T, p_T = S6HamiltonianODE(pot_adj, mass=mass)(q_adj, p_adj, h, steps)
        assert torch.allclose(q, q_T, atol=2e-14, rtol=2e-14)
        assert torch.allclose(p, p_T, atol=2e-14, rtol=2e-14)


    def test_time_reversal_and_invalid_scheme():
        """The discrete S6-star map must invert under h -> -h, and bad schemes must fail loudly."""
        pot = SimpleParametricPotential()
        q0 = torch.tensor([0.9, -0.35], dtype=torch.float64)
        p0 = torch.tensor([0.25, 0.55], dtype=torch.float64)
        ode = S6HamiltonianODE(pot, mass=1.7)
        q_T, p_T = ode(q0, p0, 0.017, 8)
        q_R, p_R = ode(q_T, p_T, -0.017, 8)
        assert torch.allclose(q_R, q0, atol=3e-15, rtol=3e-15)
        assert torch.allclose(p_R, p0, atol=3e-15, rtol=3e-15)

        try:
            S6HamiltonianODE(pot, scheme='not_a_real_scheme')
        except ValueError:
            pass
        else:
            raise AssertionError('invalid scheme must raise ValueError')
    if __name__ == '__main__':
        test_against_naive_unrolled_autograd()
        test_torch_gradcheck()
        test_output_cotangent_and_parameter_edge_cases()
        test_mass_consistency_against_naive()
        test_time_reversal_and_invalid_scheme()
