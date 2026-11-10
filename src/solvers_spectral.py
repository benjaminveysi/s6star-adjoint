"""
===================================================================================================
S6* SPECTRAL PRODUCTION SOLVERS ENGINE (REAL & COMPLEX BRANCHES)
===================================================================================================
Provides production-grade spectral solver classes for the 7 Cartesian/Fourier members of the S6* family:

1. RealGPESolver (Branch I):
   - Real-Standard-33: Canonical 13-stage commutator-corrected splitting (33 FFT transforms)
   - Yoshida-6 (1990 baseline, NOT an S6* family member): 7-stage symmetric T-V-T pure splitting with FSAL stage fusion (14 FFT transforms)
   - Real-Stencil-12: 6-stage S6* with 8th-order spatial stencil commutators (12 FFT transforms)
   Includes complete analytical double commutator for arbitrary external potentials V(x),
   autonomous vs replay adaptive stepping, and real-time transform call instrumentation.

2. ComplexParabolicSolver (Branch II):
   - Complex-Standard-32: Canonical 13-stage splitting (2-point Gauss-Legendre quadrature)
   - Complex-Flash-22: 13-stage midpoint subflow spectral commutator
   - BCChM-6 P6S7 (2013 baseline, NOT an S6* family member): 7-stage pure complex T-V-T splitting with FSAL
   - Complex-Stencil-12: 6-stage forward-positive with 8th-order spatial stencils
   - Castella-Blanes (2009 baseline, NOT an S6* family member): 5-stage alternating-conjugate macro-pair scaled to exact time step h
   Employs spectral Fourier derivatives for smooth arbitrary potentials to preserve 6th-order rate.
===================================================================================================
"""

import numpy as np
from src.coefficients import (
    REAL_STANDARD_33_A, REAL_STANDARD_33_B, REAL_STANDARD_33_C,
    YOSHIDA_6_A, YOSHIDA_6_B, YOSHIDA6_W,
    REAL_STENCIL_12_A, REAL_STENCIL_12_B, REAL_STENCIL_12_C,
    STENCIL_D, STENCIL_D2, STENCIL_D2_CENTER,
    COMPLEX_STANDARD_32_A, COMPLEX_STANDARD_32_B, COMPLEX_STANDARD_32_C,
    COMPLEX_FLASH_22_A, COMPLEX_FLASH_22_B, COMPLEX_FLASH_22_C,
    BCCHM_6_A, BCCHM_6_B, BCCHM_GAMMAS,
    COMPLEX_STENCIL_12_A, COMPLEX_STENCIL_12_B, COMPLEX_STENCIL_12_C,
    CASTELLA_BLANES_GAMMA
)

class RealGPESolver:
    """1D/2D Focusing/Defocusing Gross-Pitaevskii Equation Solver for Hamiltonian Waves."""
    def __init__(self, N=2048, L=80.0, theta=-2.0, V=None, Vx=None):
        self.N = N
        self.L = L
        self.theta = theta
        self.dx = L / N
        self.x = np.linspace(-L / 2.0, L / 2.0, N, endpoint=False)
        self.k = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.disp_k = -1.0j * (self.k ** 2)
        self.ik = 1.0j * self.k
        self.minus_k2 = - (self.k ** 2)
        self.fft_count = 0

        if V is None:
            self.V = np.zeros(N, dtype=float)
            self.Vx = np.zeros(N, dtype=float)
        elif callable(V):
            self.V = V(self.x)
            if Vx is not None:
                self.Vx = Vx(self.x) if callable(Vx) else np.array(Vx, dtype=float)
            else:
                # Spectral derivative for smooth periodic potentials
                V_hat = np.fft.fft(self.V)
                self.Vx = np.fft.ifft(self.ik * V_hat).real
        else:
            self.V = np.array(V, dtype=float)
            if Vx is not None:
                self.Vx = Vx(self.x) if callable(Vx) else np.array(Vx, dtype=float)
            else:
                V_hat = np.fft.fft(self.V)
                self.Vx = np.fft.ifft(self.ik * V_hat).real

    def reset_fft_count(self):
        self.fft_count = 0

    def get_fft_count(self):
        return self.fft_count

    def fft(self, v):
        self.fft_count += 1
        return np.fft.fft(v)

    def ifft(self, v_hat):
        self.fft_count += 1
        return np.fft.ifft(v_hat)

    def compute_energy(self, psi):
        psi_hat = self.fft(psi)
        T_psi = self.ifft(-0.5 * self.minus_k2 * psi_hat).real
        E_kin = np.sum(psi.conj() * T_psi).real * self.dx
        E_pot = np.sum(self.V * np.abs(psi)**2).real * self.dx
        E_int = 0.5 * self.theta * np.sum(np.abs(psi)**4).real * self.dx
        norm = np.sum(np.abs(psi)**2).real * self.dx
        return norm, E_kin + E_pot + E_int

    def compute_f1_f2_spectral(self, psi):
        """Computes complete double commutator [V_eff, [T, V_eff]] for V_eff = V(x) + theta |psi|^2.
        
        Mathematical Formulation:
            T = -d^2/dx^2, V_eff = V(x) + theta * rho(x)
            [V_eff, [T, V_eff]] = 2 |d(V_eff)/dx|^2 + nonlinear GPE variation terms.
            - Pure potential contribution: + 2 * (V_x)^2
            - Cross contribution: + 4 * theta * V_x * rho_x
            - Pure nonlinear contribution: - 4 * theta * g6
        """
        psi_hat = self.fft(psi)
        grad_psi = self.ifft(self.ik * psi_hat)
        Delta_psi = self.ifft(self.minus_k2 * psi_hat)

        v_conj = np.conj(psi)
        rho = psi.real**2 + psi.imag**2
        rho_x = 2.0 * (v_conj * grad_psi).real

        g2 = (v_conj * Delta_psi).real
        g3 = (np.conj(grad_psi) * grad_psi).real
        g4 = ((v_conj**2) * (grad_psi**2)).real

        g5 = self.theta * (2.0 * g2 + 3.0 * g3)
        g6 = rho * g5 + self.theta * g4

        f1 = self.theta * rho + self.V
        # Complete commutator for arbitrary V(x) and theta:
        f2 = -4.0 * self.theta * g6 + 2.0 * (self.Vx ** 2) + 4.0 * self.theta * self.Vx * rho_x
        return f1, f2

    def compute_f1_f2_stencil(self, psi):
        grad_psi = (
            STENCIL_D[0] * (np.roll(psi, -1) - np.roll(psi, 1)) +
            STENCIL_D[1] * (np.roll(psi, -2) - np.roll(psi, 2)) +
            STENCIL_D[2] * (np.roll(psi, -3) - np.roll(psi, 3)) +
            STENCIL_D[3] * (np.roll(psi, -4) - np.roll(psi, 4))
        ) / self.dx

        Delta_psi = (
            STENCIL_D2_CENTER * psi +
            STENCIL_D2[0] * (np.roll(psi, -1) + np.roll(psi, 1)) +
            STENCIL_D2[1] * (np.roll(psi, -2) + np.roll(psi, 2)) +
            STENCIL_D2[2] * (np.roll(psi, -3) + np.roll(psi, 3)) +
            STENCIL_D2[3] * (np.roll(psi, -4) + np.roll(psi, 4))
        ) / (self.dx ** 2)

        v_conj = np.conj(psi)
        rho = psi.real**2 + psi.imag**2
        rho_x = 2.0 * (v_conj * grad_psi).real

        g2 = (v_conj * Delta_psi).real
        g3 = (np.conj(grad_psi) * grad_psi).real
        g4 = ((v_conj**2) * (grad_psi**2)).real

        g5 = self.theta * (2.0 * g2 + 3.0 * g3)
        g6 = rho * g5 + self.theta * g4

        f1 = self.theta * rho + self.V
        f2 = -4.0 * self.theta * g6 + 2.0 * (self.Vx ** 2) + 4.0 * self.theta * self.Vx * rho_x
        return f1, f2

    def step_standard_33(self, psi, h):
        v = psi.copy()
        h2 = h * h
        for idx in range(6):
            f1, f2 = self.compute_f1_f2_spectral(v)
            theta_p = h * (REAL_STANDARD_33_B[idx] * f1 + REAL_STANDARD_33_C[idx] * h2 * f2)
            v *= np.exp(-1j * theta_p)
            v = self.ifft(self.fft(v) * np.exp(REAL_STANDARD_33_A[idx] * h * self.disp_k))
        f1, f2 = self.compute_f1_f2_spectral(v)
        theta_p = h * (REAL_STANDARD_33_B[6] * f1 + REAL_STANDARD_33_C[6] * h2 * f2)
        v *= np.exp(-1j * theta_p)
        return v

    def step_stencil_12(self, psi, h):
        v = psi.copy()
        h2 = h * h
        for idx in range(6):
            f1, f2 = self.compute_f1_f2_stencil(v)
            theta_p = h * (REAL_STENCIL_12_B[idx] * f1 + REAL_STENCIL_12_C[idx] * h2 * f2)
            v *= np.exp(-1j * theta_p)
            v = self.ifft(self.fft(v) * np.exp(REAL_STENCIL_12_A[idx] * h * self.disp_k))
        f1, f2 = self.compute_f1_f2_stencil(v)
        theta_p = h * (REAL_STENCIL_12_B[6] * f1 + REAL_STENCIL_12_C[6] * h2 * f2)
        v *= np.exp(-1j * theta_p)
        return v

    def step_yoshida6(self, psi, h):
        """Single standalone step of Yoshida-6 (1990 baseline) using fused stages (16 transforms without inter-step FSAL)."""
        v = psi.copy()
        v = self.ifft(self.fft(v) * np.exp(YOSHIDA_6_A[0] * h * self.disp_k))
        for idx in range(7):
            f1 = self.theta * (v.real**2 + v.imag**2) + self.V
            v *= np.exp(-1j * YOSHIDA_6_B[idx] * h * f1)
            v = self.ifft(self.fft(v) * np.exp(YOSHIDA_6_A[idx + 1] * h * self.disp_k))
        return v

    def integrate_yoshida6(self, psi0, n_steps, h, use_fsal=True):
        """Multi-step integration of Yoshida-6 (1990 baseline) with FSAL fusion: exactly 14 transforms per step."""
        v = psi0.copy()
        if not use_fsal or n_steps <= 1:
            for _ in range(n_steps):
                v = self.step_yoshida6(v, h)
            return v

        # FSAL: Initial kinetic half-step
        v = self.ifft(self.fft(v) * np.exp(YOSHIDA_6_A[0] * h * self.disp_k))

        for step in range(n_steps):
            for idx in range(6):
                f1 = self.theta * (v.real**2 + v.imag**2) + self.V
                v *= np.exp(-1j * YOSHIDA_6_B[idx] * h * f1)
                v = self.ifft(self.fft(v) * np.exp(YOSHIDA_6_A[idx + 1] * h * self.disp_k))

            # Stage 6 potential
            f1 = self.theta * (v.real**2 + v.imag**2) + self.V
            v *= np.exp(-1j * YOSHIDA_6_B[6] * h * f1)

            if step < n_steps - 1:
                # Fused kinetic: a_8 + a_1
                fused_a = YOSHIDA_6_A[7] + YOSHIDA_6_A[0]
                v = self.ifft(self.fft(v) * np.exp(fused_a * h * self.disp_k))
            else:
                # Final kinetic stage
                v = self.ifft(self.fft(v) * np.exp(YOSHIDA_6_A[7] * h * self.disp_k))

        return v

    def compute_gradient_monitor(self, psi):
        """Computes M(psi) = ||grad psi||_L2 on the exact updated field without latency."""
        psi_hat = np.fft.fft(psi)
        grad_psi = np.fft.ifft(self.ik * psi_hat)
        return np.sqrt(np.sum(np.abs(grad_psi)**2) * self.dx)


class ComplexParabolicSolver:
    """Nonlinear Parabolic Reaction-Diffusion & Imaginary-Time GPE Solver."""
    def __init__(self, N=512, L=20.0, theta=2.0, V=None, Vx=None, Vxx=None):
        self.N = N
        self.L = L
        self.theta = theta
        self.dx = L / N
        self.x = np.linspace(-L / 2.0, L / 2.0, N, endpoint=False)
        self.k = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.disp_k = - (self.k ** 2)
        self.ik = 1.0j * self.k
        self.fft_count = 0

        if V is None:
            self.V = 0.5 * (self.x ** 2)
            self.Vx = self.x.copy()
            self.Vxx = np.ones_like(self.x)
        else:
            self.V = V(self.x) if callable(V) else np.array(V, dtype=float)
            if Vx is not None and Vxx is not None:
                self.Vx = Vx(self.x) if callable(Vx) else np.array(Vx, dtype=float)
                self.Vxx = Vxx(self.x) if callable(Vxx) else np.array(Vxx, dtype=float)
            else:
                # Spectral derivatives for smooth periodic potentials (exact to machine precision)
                V_hat = np.fft.fft(self.V)
                self.Vx = np.fft.ifft(self.ik * V_hat).real
                self.Vxx = np.fft.ifft(self.disp_k * V_hat).real

    def reset_fft_count(self):
        self.fft_count = 0

    def get_fft_count(self):
        return self.fft_count

    def fft(self, v):
        self.fft_count += 1
        return np.fft.fft(v)

    def ifft(self, v_hat):
        self.fft_count += 1
        return np.fft.ifft(v_hat)

    def bernoulli_step(self, u0, tau):
        exp_V_tau = np.exp(-self.V * tau)
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = np.where(np.abs(self.V) < 1e-12, 2.0 * tau, (1.0 - np.exp(-2.0 * self.V * tau)) / self.V)
        denom = np.sqrt(1.0 + self.theta * (u0 ** 2) * factor)
        return (u0 * exp_V_tau) / denom

    def compute_g2_spectral(self, u):
        u_hat = self.fft(u)
        u_x = self.ifft(self.ik * u_hat)
        u2 = u ** 2
        ux2 = u_x ** 2
        term = (self.Vx ** 2) - self.theta * u2 * self.Vxx + 6.0 * self.theta * u * self.Vx * u_x + 6.0 * self.theta * self.V * ux2 + 12.0 * (self.theta ** 2) * u2 * ux2
        return 2.0 * term * u

    def compute_g2_stencil(self, u):
        ux = (
            STENCIL_D[0] * (np.roll(u, -1) - np.roll(u, 1)) +
            STENCIL_D[1] * (np.roll(u, -2) - np.roll(u, 2)) +
            STENCIL_D[2] * (np.roll(u, -3) - np.roll(u, 3)) +
            STENCIL_D[3] * (np.roll(u, -4) - np.roll(u, 4))
        ) / self.dx
        u2 = u ** 2
        ux2 = ux ** 2
        term = (self.Vx ** 2) - self.theta * u2 * self.Vxx + 6.0 * self.theta * u * self.Vx * ux + 6.0 * self.theta * self.V * ux2 + 12.0 * (self.theta ** 2) * u2 * ux2
        return 2.0 * term * u

    def step_standard_32(self, u, h):
        alpha1 = 0.5 - np.sqrt(3.0) / 6.0
        alpha2 = np.sqrt(3.0) / 3.0
        v = u.copy()
        for idx in range(6):
            b_val = COMPLEX_STANDARD_32_B[idx]
            c_val = COMPLEX_STANDARD_32_C[idx]
            if abs(c_val) > 1e-15:
                v = self.bernoulli_step(v, alpha1 * b_val * h)
                v = v + 0.5 * c_val * (h ** 3) * self.compute_g2_spectral(v)
                v = self.bernoulli_step(v, alpha2 * b_val * h)
                v = v + 0.5 * c_val * (h ** 3) * self.compute_g2_spectral(v)
                v = self.bernoulli_step(v, alpha1 * b_val * h)
            else:
                v = self.bernoulli_step(v, b_val * h)
            v = self.ifft(self.fft(v) * np.exp(COMPLEX_STANDARD_32_A[idx] * h * self.disp_k))
        v = self.bernoulli_step(v, COMPLEX_STANDARD_32_B[6] * h)
        return v.real

    def step_flash_22(self, u, h):
        v = u.copy()
        for idx in range(6):
            b_val = COMPLEX_FLASH_22_B[idx]
            c_val = COMPLEX_FLASH_22_C[idx]
            if abs(c_val) > 1e-15:
                v = self.bernoulli_step(v, 0.5 * b_val * h)
                g2 = self.compute_g2_spectral(v)
                v = v + c_val * (h ** 3) * g2
                v = self.bernoulli_step(v, 0.5 * b_val * h)
            else:
                v = self.bernoulli_step(v, b_val * h)
            v = self.ifft(self.fft(v) * np.exp(COMPLEX_FLASH_22_A[idx] * h * self.disp_k))
        v = self.bernoulli_step(v, COMPLEX_FLASH_22_B[6] * h)
        return v.real

    def step_bcchm_6(self, u, h):
        """Single standalone step of BCChM-6 P6S7 (external baseline, formerly 'Complex-Bullet-14')."""
        v = u.copy()
        v = self.ifft(self.fft(v) * np.exp(BCCHM_6_A[0] * h * self.disp_k))
        for idx in range(7):
            v = self.bernoulli_step(v, BCCHM_6_B[idx] * h)
            v = self.ifft(self.fft(v) * np.exp(BCCHM_6_A[idx + 1] * h * self.disp_k))
        return v.real

    def integrate_bcchm_6(self, u0, n_steps, h, use_fsal=True):
        """Multi-step integration of BCChM-6 P6S7 with FSAL: exactly 14 transforms per step (external baseline)."""
        v = u0.copy()
        if not use_fsal or n_steps <= 1:
            for _ in range(n_steps):
                v = self.step_bcchm_6(v, h)
            return v.real

        # FSAL Initial kinetic stage
        v = self.ifft(self.fft(v) * np.exp(BCCHM_6_A[0] * h * self.disp_k))

        for step in range(n_steps):
            for idx in range(6):
                v = self.bernoulli_step(v, BCCHM_6_B[idx] * h)
                v = self.ifft(self.fft(v) * np.exp(BCCHM_6_A[idx + 1] * h * self.disp_k))

            # Stage 6 potential
            v = self.bernoulli_step(v, BCCHM_6_B[6] * h)

            if step < n_steps - 1:
                fused_a = BCCHM_6_A[7] + BCCHM_6_A[0]
                v = self.ifft(self.fft(v) * np.exp(fused_a * h * self.disp_k))
            else:
                v = self.ifft(self.fft(v) * np.exp(BCCHM_6_A[7] * h * self.disp_k))

        return v.real

    def step_stencil_12(self, u, h):
        v = u.copy()
        for idx in range(6):
            b_val = COMPLEX_STENCIL_12_B[idx]
            c_val = COMPLEX_STENCIL_12_C[idx]
            if abs(c_val) > 1e-15:
                v = self.bernoulli_step(v, 0.5 * b_val * h)
                g2 = self.compute_g2_stencil(v)
                v = v + c_val * (h ** 3) * g2
                v = self.bernoulli_step(v, 0.5 * b_val * h)
            else:
                v = self.bernoulli_step(v, b_val * h)
            v = self.ifft(self.fft(v) * np.exp(COMPLEX_STENCIL_12_A[idx] * h * self.disp_k))
        v = self.bernoulli_step(v, COMPLEX_STENCIL_12_B[6] * h)
        return v.real

    def step_castella_blanes(self, u, h):
        """Castella-Blanes 5-stage forward-positive splitting (external baseline, formerly 'Complex-Instant-10'): exactly 10 FFT transforms per macro-step h."""
        v = u.copy()
        for g in CASTELLA_BLANES_GAMMA:
            v = self.bernoulli_step(v, 0.5 * g * h)
            v = self.ifft(self.fft(v) * np.exp(g * h * self.disp_k))
            v = self.bernoulli_step(v, 0.5 * g * h)
        return v.real

    step_castella_blanes_pair = step_castella_blanes
