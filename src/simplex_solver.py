"""
===================================================================================================
S6* SIMPLEX V16 — FIELD-AGNOSTIC GEOMETRIC SOLVER ENGINE
===================================================================================================
One spatial/operator/stage engine over K in {R, C}. The scalar field is a representation
choice; the physical evolution is selected independently through SimplexEvolutionPolicy.

Shared machinery:
  geometry -> mass/stiffness -> M^{-1}K -> discrete commutators -> Krylov/Arnoldi -> stages

Built-in policies include:
  * hamiltonian: canonical real-coefficient S6* commutator splitting for complex wave fields.
  * parabolic: forward-positive BCChM-6 baseline for dissipative evolution.
  * s6star: the same canonical S6* commutator algebra specialized to either R or C.

Legacy field_type/scheme arguments remain accepted for compatibility.
===================================================================================================
"""

import numpy as np
from scipy.linalg import expm
from src.simplex_policy import make_policy

class SimplexSolver:
    """Unified geometry-native S6* solver over a common Simplex spatial engine.

    ``field`` selects the scalar representation (real/complex); ``evolution`` selects
    the physical evolution policy (Hamiltonian/parabolic). These are intentionally
    independent concepts. The mesh, operators, Krylov engine and stage engine are shared.

    ``field_type`` remains accepted as a backwards-compatible alias.
    """
    def __init__(
        self, mesh, field_type='complex', theta=0.0, V=None, scheme=None,
        *, field=None, evolution=None, real_projection_tol=1e-12
    ):
        self.mesh = mesh
        self.field = (field if field is not None else field_type).lower()
        if self.field in {'c', 'complex128'}:
            self.field = 'complex'
        elif self.field in {'r', 'float64'}:
            self.field = 'real'
        if self.field not in {'real', 'complex'}:
            raise ValueError(f"Unknown field: {self.field}")

        self.field_type = self.field  # backward-compatible attribute
        self.theta = theta
        self.real_projection_tol = real_projection_tol

        # Infer a legacy policy only when the new explicit evolution argument is omitted.
        if evolution is None:
            if scheme is None:
                evolution = 'hamiltonian' if self.field == 'complex' else 'parabolic'
                scheme = None if evolution == 'parabolic' else 's6_commutator'
            elif scheme.lower() in {'s6_commutator', 'commutator', 's6'}:
                evolution = 'hamiltonian' if self.field == 'complex' else 'parabolic'
            elif scheme.lower() in {'bcchm'}:
                evolution = 'legacy'
            elif scheme.lower() in {'yoshida', 'pure'}:
                evolution = 'legacy'
            else:
                raise ValueError(f"Cannot infer evolution from scheme: {scheme}")

        self.evolution = evolution.lower()
        self.scheme = (scheme or 's6_commutator').lower()
        self.policy = make_policy(self.evolution, self.field, self.scheme if evolution == 'legacy' else scheme)
        self.stages_a = self.policy.stages_a
        self.stages_b = self.policy.stages_b
        self.stages_c = self.policy.stages_c

        if self.policy.requires_complex_field and self.field != 'complex':
            raise ValueError(self.policy.name + ' requires field=complex')

        if V is None:
            self.V = np.zeros(mesh.N_nodes, dtype=float)
        elif callable(V):
            self.V = np.asarray(V(mesh.nodes[:, 0], mesh.nodes[:, 1]), dtype=float)
        else:
            self.V = np.asarray(V, dtype=float)
        if self.V.shape != (mesh.N_nodes,):
            raise ValueError(f"V must have shape ({mesh.N_nodes},), got {self.V.shape}")

        if self.policy.uses_commutator and np.any(self.V):
            self.C_mat = self.mesh.compute_discrete_commutator(self.V)
            self.mesh._commutator_matrix = self.C_mat
        else:
            self.C_mat = None
            self.mesh._commutator_matrix = None

    @property
    def uses_commutator(self):
        return self.policy.uses_commutator

    @property
    def scalar_dtype(self):
        return np.dtype(np.complex128 if self.field == 'complex' or self.policy.internal_complex else np.float64)

    def describe(self):
        return {
            'field': self.field,
            'evolution': self.evolution,
            'policy': self.policy.name,
            'uses_commutator': self.policy.uses_commutator,
            'kinetic_stages': self.policy.kinetic_stage_count,
            'potential_stages': self.policy.potential_stage_count,
            'generator': self.policy.generator,
            'internal_dtype': str(self.scalar_dtype),
        }

    def reset_spmv_count(self):
        self.mesh.reset_spmv_count()

    def get_spmv_count(self):
        return self.mesh.get_spmv_count()

    def potential_step(self, u, b_val, h, c_val=0.0, m_krylov=16):
        """Apply a potential stage under the current field/evolution policy."""
        rho = (np.conj(u) * u).real
        V_eff = self.V + self.theta * rho
        if self.policy.uses_commutator and self.C_mat is not None and abs(c_val) > 1e-18:
            return self.commutator_potential_step(u, b_val, h, c_val=c_val, m=m_krylov)

        v = np.asarray(u, dtype=self.scalar_dtype)
        return v * np.exp(self.policy.generator * (b_val * h) * V_eff)

    def _exponential_action_m_inner(self, u, tau, apply_operator, m=16, sign=-1j, hermitian=True):
        """Apply exp(sign*tau*L)u using an M-inner-product Krylov projection.

        Hermitian M-self-adjoint operators use the three-term Lanczos recurrence.
        General complex combinations use M-orthogonal Arnoldi. This distinction is
        essential for the parabolic complex-coefficient commutator stages.
        """
        beta = self.mesh.norm(u)
        if beta < 1e-15 or abs(tau) < 1e-18:
            return u.copy()

        N = self.mesh.N_nodes
        Q = np.zeros((m, N), dtype=complex)
        v = (u / beta).astype(complex)
        Q[0] = v

        if hermitian:
            T = np.zeros((m, m), dtype=float)
            v_prev = np.zeros_like(v)
            beta_prev = 0.0
            actual_m = m
            for j in range(m):
                w = apply_operator(v)
                alpha = float(self.mesh.inner_product(v, w).real)
                T[j, j] = alpha
                w = w - alpha * v - beta_prev * v_prev
                if j < m - 1:
                    beta_next = self.mesh.norm(w)
                    if beta_next < 1e-13:
                        actual_m = j + 1
                        T = T[:actual_m, :actual_m]
                        Q = Q[:actual_m]
                        break
                    T[j, j + 1] = beta_next
                    T[j + 1, j] = beta_next
                    v_prev, v = v, w / beta_next
                    Q[j + 1] = v
                    beta_prev = beta_next
        else:
            # Modified Gram-Schmidt Arnoldi in the M-inner product.
            H = np.zeros((m, m), dtype=complex)
            actual_m = m
            for j in range(m):
                w = apply_operator(Q[j])
                for i in range(j + 1):
                    hij = self.mesh.inner_product(Q[i], w)
                    H[i, j] += hij
                    w = w - hij * Q[i]
                beta_next = self.mesh.norm(w)
                if j < m - 1:
                    H[j + 1, j] = beta_next
                    if beta_next < 1e-13:
                        actual_m = j + 1
                        H = H[:actual_m, :actual_m]
                        Q = Q[:actual_m]
                        break
                    Q[j + 1] = w / beta_next
                else:
                    actual_m = m
            T = H

        E = expm(sign * tau * T)
        coeff = E[:, 0] * beta
        return coeff @ Q[:len(coeff)]

    def commutator_potential_step(self, u, b_val, h, c_val=0.0, m=16):
        """Apply the combined B-stage exponential in the common Krylov engine.

        B_j(h) = exp(generator * h * (b_j V + h^2 c_j C)).
        """
        if self.C_mat is None or abs(c_val) < 1e-18:
            return np.asarray(u, dtype=self.scalar_dtype) * np.exp(
                self.policy.generator * b_val * h * (self.V + self.theta * (np.conj(u) * u).real)
            )
        if abs(self.theta) > 1e-15:
            raise ValueError(
                'S6* Simplex-Commutator is currently validated only for theta=0. '
                'Use a pure-splitting policy for nonlinear GPE until the nonlinear commutator is implemented.'
            )

        needs_arnoldi = np.iscomplexobj(b_val) or np.iscomplexobj(c_val) or not np.isrealobj(b_val)
        # Hoist the scalar h^2*c_val out of the closure: apply_B is evaluated
        # once per Krylov iterate, and there is no reason to recompute it there.
        # (Pure scalar — bitwise identical result.)
        h2_c = (h * h) * c_val
        def apply_B(x):
            return b_val * (self.V * x) + h2_c * self.mesh.apply_commutator(x)

        # NOTE (do not "simplify" to Lanczos): with complex stage coefficients
        # the stage generator b_j*V + h^2*c_j*C is not M-self-adjoint, so the
        # three-term Lanczos recurrence is invalid and full M-orthogonal
        # Arnoldi is required.  This is what needs_arnoldi selects.
        out = self._exponential_action_m_inner(
            u, h, apply_B, m=m, sign=self.policy.generator, hermitian=not needs_arnoldi
        )
        return np.real(out) if self.field == 'real' else out

    def kinetic_step_lanczos(self, u, a_val, h, m=16, tol=1e-13, adaptive=False,
                             step_scaled_tol=True, adaptive_check_every=1):
        """Lanczos kinetic stage with optional adaptive early termination.

        The adaptive error estimate requires a dense sub-expm of size O(j^3)
        at each checked iteration, so ``adaptive_check_every`` controls how
        often it is evaluated: every N-th iteration past j=8.  This is the
        NumPy-side analog of avoiding a costly per-iteration check — the check
        can only delay termination slightly, never make it less accurate.
        ``adaptive_check_every=1`` reproduces the original every-iteration
        behavior exactly.
        """
        tau = a_val * h
        beta = self.mesh.norm(u)
        if beta < 1e-15 or abs(tau) < 1e-18:
            return np.asarray(u, dtype=self.scalar_dtype).copy()

        effective_tol = max(tol * abs(tau), 1e-15) if step_scaled_tol else tol
        needs_complex = self.scalar_dtype.kind == 'c' or np.iscomplexobj(a_val)
        dtype = complex if needs_complex else float
        basis = np.zeros((m, self.mesh.N_nodes), dtype=dtype)
        T_mat = np.zeros((m, m), dtype=complex if needs_complex else float)
        v_curr = (u / beta).astype(dtype)
        basis[0] = v_curr
        v_prev = np.zeros_like(v_curr)
        beta_prev = 0.0
        actual_m = m

        for j in range(m):
            w = self.mesh.apply_laplacian(v_curr)
            alpha = self.mesh.inner_product(v_curr, w).real
            T_mat[j, j] = alpha
            w = w - alpha * v_curr - beta_prev * v_prev
            if j < m - 1:
                beta_next = self.mesh.norm(w)
                if beta_next < 1e-14:
                    actual_m = j + 1
                    T_mat = T_mat[:actual_m, :actual_m]
                    basis = basis[:actual_m]
                    break
                T_mat[j, j + 1] = beta_next
                T_mat[j + 1, j] = beta_next
                v_prev, v_curr = v_curr, w / beta_next
                basis[j + 1] = v_curr
                beta_prev = beta_next
                if adaptive and j >= 8 and (j - 8) % max(adaptive_check_every, 1) == 0:
                    sub_exp = expm(self.policy.generator * tau * T_mat[:j + 1, :j + 1])
                    err_est = beta * beta_next * abs(tau) * abs(sub_exp[j, 0])
                    if err_est < effective_tol:
                        actual_m = j + 1
                        T_mat = T_mat[:actual_m, :actual_m]
                        basis = basis[:actual_m]
                        break

        E = expm(self.policy.generator * tau * T_mat)
        coeffs = E[:, 0] * beta
        # Single BLAS matvec instead of a Python-level axpy loop over the
        # basis rows.  One allocation instead of m; the Krylov loop above
        # still dominates, but this removes the per-stage Python overhead.
        # (Summation order differs from the loop at ~1e-16; every repo
        # tolerance is far above this.)
        return coeffs @ basis[:len(coeffs)]

    def _apply_stage(self, v, idx, h, m_krylov, tol, adaptive, step_scaled_tol,
                     adaptive_check_every=1):
        v = self.potential_step(v, self.stages_b[idx], h, c_val=self.stages_c[idx], m_krylov=m_krylov)
        return self.kinetic_step_lanczos(
            v, self.stages_a[idx], h, m=m_krylov, tol=tol, adaptive=adaptive,
            step_scaled_tol=step_scaled_tol, adaptive_check_every=adaptive_check_every
        )

    def step_simplex_s6(self, u, h, m_krylov=16, tol=1e-13, adaptive=False,
                        step_scaled_tol=True, adaptive_check_every=1):
        """Single macro-step through the policy-selected splitting on the common engine."""
        v = np.asarray(u, dtype=self.scalar_dtype).copy()
        if self.policy.uses_commutator:
            for idx in range(6):
                v = self._apply_stage(v, idx, h, m_krylov, tol, adaptive, step_scaled_tol,
                                      adaptive_check_every)
            v = self.potential_step(v, self.stages_b[6], h, c_val=self.stages_c[6], m_krylov=m_krylov)
        else:
            v = self.kinetic_step_lanczos(v, self.stages_a[0], h, m=m_krylov, tol=tol, adaptive=adaptive,
                                          step_scaled_tol=step_scaled_tol,
                                          adaptive_check_every=adaptive_check_every)
            for idx in range(7):
                v = self.potential_step(v, self.stages_b[idx], h, c_val=0.0, m_krylov=m_krylov)
                v = self.kinetic_step_lanczos(v, self.stages_a[idx + 1], h, m=m_krylov, tol=tol,
                                              adaptive=adaptive, step_scaled_tol=step_scaled_tol,
                                              adaptive_check_every=adaptive_check_every)
        return self._format_output(v)

    def _format_output(self, v):
        if self.field == 'real':
            imag_norm = self.mesh.norm(np.imag(v)) if np.iscomplexobj(v) else 0.0
            self.last_real_projection_error = float(imag_norm)
            if imag_norm > self.real_projection_tol and not self.policy.unsafe_real_projection:
                raise RuntimeError(
                    f"Real-field specialization received a non-real macro-step: "
                    f"||Im(v)||_M={imag_norm:.3e} > tolerance={self.real_projection_tol:.3e}. "
                    "Choose field='complex' or a field-preserving real evolution policy."
                )
            return np.real(v)
        self.last_real_projection_error = 0.0
        return v

    def _step_no_fusion(self, v, h, m_krylov, tol, adaptive, step_scaled_tol,
                          adaptive_check_every=1):
        return self.step_simplex_s6(
            v, h, m_krylov=m_krylov, tol=tol, adaptive=adaptive, step_scaled_tol=step_scaled_tol,
            adaptive_check_every=adaptive_check_every
        )

    def integrate_simplex_s6(self, u0, n_steps, h, m_krylov=16, tol=1e-13, use_fsal=True,
                             adaptive=False, step_scaled_tol=True, adaptive_check_every=1):
        """Integrate multiple macro-steps using exact boundary fusion where the policy permits it.

        FSAL exactness note (do not "fix" the 2.0 factors): the map starts and
        ends in a B stage, so the trailing B(h) of step k and the leading B(h)
        of step k+1 are the *same* operator exp(generator*h*(b6*V + h^2*c6*C)).
        Fusing them as one stage with (2*b6, 2*c6) is exact, not an
        approximation — halving the factors would silently drop half the
        boundary potential.
        """
        v = np.asarray(u0, dtype=self.scalar_dtype).copy()
        if n_steps <= 0:
            return self._format_output(v)

        # Hamiltonian S6* commutator map starts/ends in B: fuse B-boundaries exactly.
        if self.policy.uses_commutator:
            if not use_fsal or n_steps == 1:
                for _ in range(n_steps):
                    v = self._step_no_fusion(v, h, m_krylov, tol, adaptive, step_scaled_tol,
                                             adaptive_check_every)
                return v

            v = self.potential_step(v, self.stages_b[0], h, c_val=self.stages_c[0], m_krylov=m_krylov)
            for step in range(n_steps):
                for idx in range(6):
                    v = self.kinetic_step_lanczos(
                        v, self.stages_a[idx], h, m=m_krylov, tol=tol, adaptive=adaptive,
                        step_scaled_tol=step_scaled_tol, adaptive_check_every=adaptive_check_every
                    )
                    if idx < 5:
                        v = self.potential_step(
                            v, self.stages_b[idx + 1], h, c_val=self.stages_c[idx + 1], m_krylov=m_krylov
                        )
                    elif step < n_steps - 1:
                        v = self.potential_step(
                            v, 2.0 * self.stages_b[6], h, c_val=2.0 * self.stages_c[6], m_krylov=m_krylov
                        )
                    else:
                        v = self.potential_step(
                            v, self.stages_b[6], h, c_val=self.stages_c[6], m_krylov=m_krylov
                        )
            return self._format_output(v)

        # Pure compositions: kinetic FSAL boundary fusion remains available.
        if not use_fsal or n_steps == 1:
            for _ in range(n_steps):
                v = self._step_no_fusion(v, h, m_krylov, tol, adaptive, step_scaled_tol,
                                         adaptive_check_every)
            return v

        v = self.kinetic_step_lanczos(v, self.stages_a[0], h, m=m_krylov, tol=tol, adaptive=adaptive,
                                      step_scaled_tol=step_scaled_tol,
                                      adaptive_check_every=adaptive_check_every)
        for step in range(n_steps):
            for idx in range(6):
                v = self.potential_step(v, self.stages_b[idx], h, c_val=0.0, m_krylov=m_krylov)
                v = self.kinetic_step_lanczos(v, self.stages_a[idx + 1], h, m=m_krylov, tol=tol,
                                              adaptive=adaptive, step_scaled_tol=step_scaled_tol,
                                              adaptive_check_every=adaptive_check_every)
            v = self.potential_step(v, self.stages_b[6], h, c_val=0.0, m_krylov=m_krylov)
            if step < n_steps - 1:
                fused_a = self.stages_a[7] + self.stages_a[0]
                v = self.kinetic_step_lanczos(v, fused_a, h, m=m_krylov, tol=tol, adaptive=adaptive,
                                              step_scaled_tol=step_scaled_tol,
                                              adaptive_check_every=adaptive_check_every)
            else:
                v = self.kinetic_step_lanczos(v, self.stages_a[7], h, m=m_krylov, tol=tol, adaptive=adaptive,
                                              step_scaled_tol=step_scaled_tol,
                                              adaptive_check_every=adaptive_check_every)
        return self._format_output(v)

    def step_strang(self, u, h, m_krylov=16, tol=1e-13, adaptive_check_every=1):
        v = self.potential_step(u, 0.5, h, c_val=0.0)
        v = self.kinetic_step_lanczos(v, 1.0, h, m=m_krylov, tol=tol, adaptive=False,
                                      adaptive_check_every=adaptive_check_every)
        v = self.potential_step(v, 0.5, h, c_val=0.0)
        return self._format_output(v)
