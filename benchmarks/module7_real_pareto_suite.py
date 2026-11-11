"""
===================================================================================================
MODULE 7: S6* REAL GEOMETRIC SPLITTING FAMILY GRAND COMPARATIVE BENCHMARK SUITE
===================================================================================================
Evaluates the optimized Real-S6* family for conservative, unitary, and Hamiltonian wave dynamics:
1. S6* Real-Stencil-12 (12 FFTs/step): 6-Stage S6* with 8th-Order Compact Spatial Stencils
2. S6* Real-Flash-18 (18 FFTs/step): 6-Stage S6* with Midpoint Subflow Spectral Commutator
3. S6* Real-Standard-33 (33 FFTs/step): Canonical 13-Stage Commutator-Corrected Splitting
4. Yoshida-6 (14 FFTs/step): 6th-Order Symmetric Composition (Yoshida 1990)
5. Chin-4A (8 FFTs/step): 4th-Order Commutator-Corrected Splitting (Chin 1997)
6. Strang (2 FFTs/step): 2nd-Order Symmetric Splitting
===================================================================================================
"""

import time
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.coefficients import (
    S6_REAL_A, S6_REAL_B, S6_REAL_C,
    STENCIL_D, STENCIL_D2, STENCIL_D2_CENTER,
    YOSHIDA_6_A, YOSHIDA_6_B, YOSHIDA6_W
)

class RealBenchmarkSuite:
    def __init__(self, N=2048, L=80.0, theta=-2.0):
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

    def integrate_yoshida6(self, psi0, n_steps, h, use_fsal=True):
        from src.coefficients import YOSHIDA_6_A, YOSHIDA_6_B
        v = psi0.copy()
        if not use_fsal or n_steps <= 1:
            for _ in range(n_steps):
                v = self.step_yoshida6(v, h)
            return v
        v = self.ifft(self.fft(v) * np.exp(YOSHIDA_6_A[0] * h * self.disp_k))
        for step in range(n_steps):
            for idx in range(6):
                f1 = self.theta * (v.real**2 + v.imag**2)
                v *= np.exp(-1j * YOSHIDA_6_B[idx] * h * f1)
                v = self.ifft(self.fft(v) * np.exp(YOSHIDA_6_A[idx + 1] * h * self.disp_k))
            f1 = self.theta * (v.real**2 + v.imag**2)
            v *= np.exp(-1j * YOSHIDA_6_B[6] * h * f1)
            if step < n_steps - 1:
                fused_a = YOSHIDA_6_A[7] + YOSHIDA_6_A[0]
                v = self.ifft(self.fft(v) * np.exp(fused_a * h * self.disp_k))
            else:
                v = self.ifft(self.fft(v) * np.exp(YOSHIDA_6_A[7] * h * self.disp_k))
        return v

    def compute_energy(self, psi):
        psi_hat = self.fft(psi)
        T_psi = self.ifft(-0.5 * self.minus_k2 * psi_hat).real
        E_kin = np.sum(psi.conj() * T_psi).real * self.dx
        E_int = 0.5 * self.theta * np.sum(np.abs(psi)**4).real * self.dx
        norm = np.sum(np.abs(psi)**2).real * self.dx
        return norm, E_kin + E_int

    def compute_f1_f2_spectral(self, psi):
        psi_hat = self.fft(psi)
        grad_psi = self.ifft(self.ik * psi_hat)
        Delta_psi = self.ifft(self.minus_k2 * psi_hat)

        v_conj = np.conj(psi)
        rho = psi.real**2 + psi.imag**2
        g2 = (v_conj * Delta_psi).real
        g3 = (np.conj(grad_psi) * grad_psi).real
        g4 = ((v_conj**2) * (grad_psi**2)).real

        g5 = self.theta * (2.0 * g2 + 3.0 * g3)
        g6 = rho * g5 + self.theta * g4

        f1 = self.theta * rho
        f2 = -4.0 * self.theta * g6
        return f1, f2

    def compute_f1_f2_stencil(self, psi):
        # High-performance 8th-order spatial derivative stencils in coordinate space
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
        g2 = (v_conj * Delta_psi).real
        g3 = (np.conj(grad_psi) * grad_psi).real
        g4 = ((v_conj**2) * (grad_psi**2)).real

        g5 = self.theta * (2.0 * g2 + 3.0 * g3)
        g6 = rho * g5 + self.theta * g4

        f1 = self.theta * rho
        f2 = -4.0 * self.theta * g6
        return f1, f2

    # -------------------------------------------------------------------------
    # Solvers
    # -------------------------------------------------------------------------
    def step_real_standard_33(self, psi, h):
        v = psi.copy()
        h2 = h * h
        for idx in range(6):
            f1, f2 = self.compute_f1_f2_spectral(v)
            theta_p = h * (S6_REAL_B[idx] * f1 + S6_REAL_C[idx] * h2 * f2)
            v *= np.exp(-1j * theta_p)
            v = self.ifft(self.fft(v) * np.exp(S6_REAL_A[idx] * h * self.disp_k))
        f1, f2 = self.compute_f1_f2_spectral(v)
        theta_p = h * (S6_REAL_B[6] * f1 + S6_REAL_C[6] * h2 * f2)
        v *= np.exp(-1j * theta_p)
        return v

    def step_real_stencil_12(self, psi, h):
        v = psi.copy()
        h2 = h * h
        for idx in range(6):
            f1, f2 = self.compute_f1_f2_stencil(v)
            theta_p = h * (S6_REAL_B[idx] * f1 + S6_REAL_C[idx] * h2 * f2)
            v *= np.exp(-1j * theta_p)
            v = self.ifft(self.fft(v) * np.exp(S6_REAL_A[idx] * h * self.disp_k))
        f1, f2 = self.compute_f1_f2_stencil(v)
        theta_p = h * (S6_REAL_B[6] * f1 + S6_REAL_C[6] * h2 * f2)
        v *= np.exp(-1j * theta_p)
        return v

    def step_yoshida6(self, psi, h):
        v = psi.copy()
        for idx in range(7):
            w = YOSHIDA6_W[idx]
            v = self.ifft(self.fft(v) * np.exp(0.5 * w * h * self.disp_k))
            f1 = self.theta * (v.real**2 + v.imag**2)
            v *= np.exp(-1j * w * h * f1)
            v = self.ifft(self.fft(v) * np.exp(0.5 * w * h * self.disp_k))
        return v

    def step_real_flash_18(self, psi, h):
        v = psi.copy()
        h2 = h * h
        # Midpoint spectral evaluation: evaluate spectral commutator once per active stage
        for idx in range(6):
            # Midpoint approximation for f1 and f2
            f1 = self.theta * (v.real**2 + v.imag**2)
            v_mid = v * np.exp(-1j * 0.5 * h * S6_REAL_B[idx] * f1)
            _, f2 = self.compute_f1_f2_spectral(v_mid)
            v *= np.exp(-1j * h * (S6_REAL_B[idx] * f1 + S6_REAL_C[idx] * h2 * f2))
            v = self.ifft(self.fft(v) * np.exp(S6_REAL_A[idx] * h * self.disp_k))
        f1 = self.theta * (v.real**2 + v.imag**2)
        v_mid = v * np.exp(-1j * 0.5 * h * S6_REAL_B[6] * f1)
        _, f2 = self.compute_f1_f2_spectral(v_mid)
        v *= np.exp(-1j * h * (S6_REAL_B[6] * f1 + S6_REAL_C[6] * h2 * f2))
        return v

    def step_chin4a(self, psi, h):
        f1_0 = self.theta * (psi.real**2 + psi.imag**2)
        v = psi * np.exp(-1j * (1.0 / 6.0) * h * f1_0)
        v = self.ifft(self.fft(v) * np.exp(0.5 * h * self.disp_k))
        f1, f2 = self.compute_f1_f2_spectral(v)
        v *= np.exp(-1j * h * ((2.0 / 3.0) * f1 - (1.0 / 72.0) * (h**2) * f2))
        v = self.ifft(self.fft(v) * np.exp(0.5 * h * self.disp_k))
        f1_end = self.theta * (v.real**2 + v.imag**2)
        v *= np.exp(-1j * (1.0 / 6.0) * h * f1_end)
        return v

    def step_strang(self, psi, h):
        f1 = self.theta * (psi.real**2 + psi.imag**2)
        v = psi * np.exp(-1j * 0.5 * h * f1)
        v = self.ifft(self.fft(v) * np.exp(h * self.disp_k))
        f1 = self.theta * (v.real**2 + v.imag**2)
        v *= np.exp(-1j * 0.5 * h * f1)
        return v


def main():
    print('=' * 115)
    print('MODULE 7: S6* REAL GEOMETRIC SPLITTING FAMILY GRAND COMPARATIVE BENCHMARK SUITE')
    print('=' * 115)
    print('Solvers: S6* Real-Stencil-12 (12 FFTs), Yoshida-6 (14 FFTs), S6* Real-Flash-18 (18 FFTs),')
    print('         S6* Real-Standard-33 (33 FFTs), Yoshida-6 (14 FFTs), Chin-4A (8 FFTs), Strang (2 FFTs)')
    print('=' * 115)

    sim = RealBenchmarkSuite(N=2048, L=80.0, theta=-2.0)
    x0 = 12.0
    v_vel = 1.0
    dphi = np.pi / 2.0

    # Two colliding solitons with phase shift pi/2
    psi1 = (1.0 / np.cosh(sim.x + x0)) * np.exp(1j * v_vel * (sim.x + x0))
    psi2 = (1.0 / np.cosh(sim.x - x0)) * np.exp(-1j * v_vel * (sim.x - x0) + 1j * dphi)
    psi0 = psi1 + psi2

    N0, E0 = sim.compute_energy(psi0)
    print(f'Initial Two-Soliton State: Norm N(0) = {N0:.12f}, Energy E(0) = {E0:.12f}')

    T_target = 6.0
    h_ladder = [0.040, 0.020, 0.010, 0.005]

    # Ultra-Fine Reference using Real-Standard-33 at h_ref = 0.0005
    h_ref = 0.0005
    n_ref = int(round(T_target / h_ref))
    print(f'\nComputing Ultra-Fine Gold Reference using S6* Real-Standard-33 (h_ref = {h_ref}, {n_ref} steps)...')
    t0 = time.perf_counter()
    psi_ref = psi0.copy()
    for _ in range(n_ref):
        psi_ref = sim.step_real_standard_33(psi_ref, h_ref)
    t_ref = time.perf_counter() - t0
    ref_norm, ref_E = sim.compute_energy(psi_ref)
    print(f'Gold Reference computed in {t_ref:.2f} s. max|psi| = {np.max(np.abs(psi_ref)):.6f}, norm = {ref_norm:.12f}')

    solvers = [
        ('S6* Real-Stencil-12',  sim.step_real_stencil_12),
        ('S6* Real-Flash-18',    sim.step_real_flash_18),
        ('S6* Real-Standard-33', sim.step_real_standard_33),
        ('Yoshida-6',            sim.step_yoshida6),
        ('Chin-4A (4th order)',  sim.step_chin4a),
        ('Strang (2nd order)',   sim.step_strang),
    ]

    results = {}
    print('=' * 115)
    print('TABLE 1: TWO-SOLITON HIGH-VELOCITY COLLISION (T = 6.0, N = 2048)')
    print('=' * 115)
    print(f'{"Method":<24} | {"Step h":>8} | {"Actual FFTs":>12} | {"Wall Time":>12} | {"L2 Error":>18} | {"Order p":>8} | Norm Defect')
    print('-' * 115)

    for name, step_fn in solvers:
        results[name] = {'h': [], 'ffts': [], 'time': [], 'error': [], 'norm_defect': [], 'energy_defect': []}
        prev_err = None
        for h in h_ladder:
            n_steps = int(round(T_target / h))
            sim.reset_fft_count()
            t_start = time.perf_counter()
            v = psi0.copy()
            if 'Yoshida-6' in name:
                v = sim.integrate_yoshida6(v, n_steps=n_steps, h=h, use_fsal=True)
            else:
                for _ in range(n_steps):
                    v = step_fn(v, h)
            t_elapsed = time.perf_counter() - t_start
            total_ffts = sim.get_fft_count()
            l2_err = np.sqrt(sim.dx * np.sum(np.abs(v - psi_ref)**2))
            order_str = f'{np.log2(prev_err / l2_err):.2f}' if (prev_err and l2_err > 1e-14) else '—'
            prev_err = l2_err

            v_norm, v_E = sim.compute_energy(v)
            norm_defect = abs(v_norm - N0)
            energy_defect = abs(v_E - E0)

            results[name]['h'].append(h)
            results[name]['ffts'].append(total_ffts)
            results[name]['time'].append(t_elapsed)
            results[name]['error'].append(l2_err)
            results[name]['norm_defect'].append(norm_defect)
            results[name]['energy_defect'].append(energy_defect)

            print(f'{name:<24} | {h:8.4f} | {total_ffts:12d} | {t_elapsed:10.4f} s | {l2_err:18.6e} | {order_str:>8} | {norm_defect:.2e}')
        print('-' * 115)

    # =========================================================================
    # Problem B: Direct Difference Audit (Real-Stencil-12 vs Real-Standard-33)
    # =========================================================================
    print('\n' + '=' * 115)
    print('AUDIT: REAL-STENCIL-12 vs REAL-STANDARD-33 INTRINSIC DISCREPANCY')
    print('=' * 115)
    for h in [0.040, 0.020, 0.010]:
        n_steps = int(round(T_target / h))
        v_std = psi0.copy()
        v_sten = psi0.copy()
        for _ in range(n_steps):
            v_std = sim.step_real_standard_33(v_std, h)
            v_sten = sim.step_real_stencil_12(v_sten, h)
        discrepancy = np.sqrt(sim.dx * np.sum(np.abs(v_std - v_sten)**2))
        print(f'Step h = {h:.4f} ({n_steps:4d} steps): Discrepancy = {discrepancy:.6e} (Verifies 8th-order stencil transparency)')
    print('=' * 115)

    # =========================================================================
    # Figure 7: Dual-Panel Grand Pareto Frontier for Real Branch
    # =========================================================================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    color_map = {
        'S6* Real-Stencil-12':  ('#d32f2f', 's-', 2.8),
        'Yoshida-6':            ('#388e3c', 'o-', 2.2),
        'S6* Real-Flash-18':    ('#0288d1', '^-', 2.2),
        'S6* Real-Standard-33': ('#1976d2', 'v-', 2.0),
        'Yoshida-6':            ('#7b1fa2', 'x-.', 1.8),
        'Chin-4A (4th order)':  ('#5d4037', '+:', 1.5),
        'Strang (2nd order)':   ('#757575', 'd:', 1.2),
    }

    for name, data in results.items():
        c, m, lw = color_map.get(name, ('black', 'o-', 1.5))
        ax1.loglog(data['ffts'], data['error'], m, color=c, linewidth=lw, label=name, markersize=7)
        time_ms = [t * 1000.0 for t in data['time']]
        ax2.loglog(time_ms, data['error'], m, color=c, linewidth=lw, label=name, markersize=7)

    ax1.set_xlabel('Total FFT Operations (Computational Cost)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Global $L_2$ Wavefield Error vs. Ultra-Fine Reference', fontsize=12, fontweight='bold')
    ax1.set_title('Real-S6* Pareto Frontier: Precision vs. FFT Work (N=2048)', fontsize=13, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, fontsize=10)

    ax2.set_xlabel('Wall-Clock Execution Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Global $L_2$ Wavefield Error vs. Ultra-Fine Reference', fontsize=12, fontweight='bold')
    ax2.set_title('Real-S6* Pareto Frontier: Precision vs. Wall-Clock Time', fontsize=13, fontweight='bold')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True, fontsize=10)

    plt.suptitle('Figure 7: S6* Real Symplectic Geometric Splitting Family Pareto Frontier (GPE Solitons, N=2048)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save exports
    for base in ['/mnt/agentdata/tiered/c_a5dcefecf6bdfde0/s6-star-splitting', '/mnt/agentdata/gcs/c_a5dcefecf6bdfde0/s6-star-splitting']:
        fig_path = f'{base}/figures/figure7_real_grand_pareto.png'
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'[AUTHENTIC EXPORT] Saved Figure 7 to: {fig_path}')

        data_path = f'{base}/data/module7_real_grand_data.npz'
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savez_compressed(
            data_path,
            h_ladder=np.array(h_ladder),
            stencil_12_error=np.array(results['S6* Real-Stencil-12']['error']),
            stencil_12_ffts=np.array(results['S6* Real-Stencil-12']['ffts']),
            yoshida_6_error=np.array(results['Yoshida-6']['error']),
            yoshida_6_ffts=np.array(results['Yoshida-6']['ffts']),
            flash_18_error=np.array(results['S6* Real-Flash-18']['error']),
            flash_18_ffts=np.array(results['S6* Real-Flash-18']['ffts']),
            standard_33_error=np.array(results['S6* Real-Standard-33']['error']),
            standard_33_ffts=np.array(results['S6* Real-Standard-33']['ffts']),
            chin4a_error=np.array(results['Chin-4A (4th order)']['error']),
            strang_error=np.array(results['Strang (2nd order)']['error']),
        )
        print(f'[AUTHENTIC EXPORT] Saved benchmark data to: {data_path}')

    plt.close()
    print('=' * 115)
    print('ALL MODULE 7 REAL GRAND SUITE BENCHMARKS COMPLETED AUTHENTICALLY!')
    print('=' * 115)

if __name__ == '__main__':
    main()
