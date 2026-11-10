"""Torch backend for the S6* Simplex-Commutator adjoint engine.

The mesh geometry and FE sparsity pattern are static.  The nodal potential is
allowed to be differentiable, so the double commutator values remain part of
the local autograd graph during a pullback stage.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only in torch-less installs
    torch = None
    nn = object

from ..core.coefficients import S6_REAL_A, S6_REAL_B, S6_REAL_C


if torch is not None:

    def _apply_C_edge_traversal(V: torch.Tensor, u: torch.Tensor,
                               row: torch.Tensor, col: torch.Tensor,
                               t_values: torch.Tensor) -> torch.Tensor:
        """Compliable core of the discrete double-commutator traversal.

        C_ij = -(V_i - V_j)^2 T_ij, applied matrix-free via one edge-wise
        scatter.  Kept as a module-level function with plain tensor arguments
        (no ``self``) so ``torch.compile`` can trace it cleanly.  Deliberately
        the out-of-place variant: under ``torch.compile`` the inductor backend
        fuses the elementwise ops around the scatter itself.
        """
        tv = t_values.to(device=u.device, dtype=u.real.dtype)
        values = -(V[row] - V[col]).square() * tv
        contrib = values.to(dtype=u.dtype) * u[col]
        out = torch.zeros_like(u)
        return out.index_add(0, row, contrib)

    class SimplexTorchBackend(nn.Module):
        """Static FE operator data plus matrix-free/Lanczos Simplex operators."""

        def __init__(self, mesh, m_krylov: int = 16, tol: float = 1e-13,
                     lanczos_check_every: int = 4, compile_kernels: bool = False):
            super().__init__()
            if getattr(mesh, "K", None) is None:
                raise ValueError("mesh must expose an assembled sparse stiffness matrix K")
            if mesh.N_nodes <= 0:
                raise ValueError("mesh must contain at least one node")
            if m_krylov < 2:
                raise ValueError("m_krylov must be >= 2")
            if tol <= 0:
                raise ValueError("tol must be positive")
            if lanczos_check_every < 1:
                raise ValueError("lanczos_check_every must be >= 1")

            K = mesh.K.tocoo()
            inv_m = np.asarray(mesh.inv_M_diag, dtype=np.float64)
            t_values = inv_m[K.row] * np.asarray(K.data, dtype=np.float64)
            t_indices = np.vstack([K.row, K.col]).astype(np.int64)
            coords = np.asarray(mesh.nodes, dtype=np.float64)

            self.register_buffer("row", torch.as_tensor(K.row, dtype=torch.long), persistent=True)
            self.register_buffer("col", torch.as_tensor(K.col, dtype=torch.long), persistent=True)
            self.register_buffer("t_values", torch.as_tensor(t_values, dtype=torch.float64), persistent=True)
            self.register_buffer("t_indices", torch.as_tensor(t_indices, dtype=torch.long), persistent=True)
            self.register_buffer("mass_diag", torch.as_tensor(np.asarray(mesh.M_diag, dtype=np.float64), dtype=torch.float64), persistent=True)
            self.register_buffer("inv_mass_diag", torch.as_tensor(inv_m, dtype=torch.float64), persistent=True)
            self.register_buffer("coords", torch.as_tensor(coords, dtype=torch.float64), persistent=True)

            self.N = int(mesh.N_nodes)
            self.m_krylov = int(m_krylov)
            self.tol = float(tol)
            # Host-sync cadence for the Lanczos residual check (enhancement #1):
            # the Krylov breakdown test is evaluated on-device and only
            # materialized to the host every `lanczos_check_every` iterations
            # (plus the final one), instead of once per iteration.
            self.lanczos_check_every = int(lanczos_check_every)
            # Enhancement #3: opt-in torch.compile for the hot edge-traversal
            # kernel.  Off by default: compilation happens lazily on first use
            # and any failure (e.g. no inductor toolchain) falls back to eager
            # exactly once, so behavior never depends on the compiler.
            self.compile_kernels = bool(compile_kernels)
            self._compiled_apply_C = None
            self._compile_failed = False
            self.register_buffer("A", torch.as_tensor(np.asarray(S6_REAL_A, dtype=np.float64)), persistent=True)
            self.register_buffer("B", torch.as_tensor(np.asarray(S6_REAL_B, dtype=np.float64)), persistent=True)
            self.register_buffer("C", torch.as_tensor(np.asarray(S6_REAL_C, dtype=np.float64)), persistent=True)

        def sparse_T(self, device=None, dtype=None):
            device = self.t_values.device if device is None else device
            dtype = self.t_values.dtype if dtype is None else dtype
            values = self.t_values.to(device=device, dtype=dtype)
            return torch.sparse_coo_tensor(
                self.t_indices.to(device=device), values,
                (self.N, self.N), device=device, dtype=dtype
            ).coalesce()

        def apply_T(self, u: torch.Tensor) -> torch.Tensor:
            # Sparse matmul requires matching scalar types; promote the fixed
            # real FE operator to the wavefunction dtype for complex fields.
            T = self.sparse_T(device=u.device, dtype=u.dtype)
            return torch.sparse.mm(T, u.unsqueeze(-1)).squeeze(-1)

        def apply_C_from_V(self, V: torch.Tensor, u: torch.Tensor,
                           out: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Apply C(V) with one differentiable traversal of the FE sparsity pattern.

            C_ij = -(V_i - V_j)^2 T_ij.  Using index_add/scatter instead of a
            differentiable sparse tensor avoids backend-specific sparse autograd
            limitations while preserving the exact discrete operator.

            Pass a preallocated ``out`` buffer (same shape/dtype/device as ``u``)
            to reuse it across repeated calls, e.g. across the iterates of one
            Lanczos run; it is zeroed in place before accumulation, saving one
            allocation per call.  The caller must only do this when no autograd
            graph is being built through these calls (each Lanczos iterate must
            stay alive for the backward), see ``potential_expm``.

            When ``compile_kernels=True`` and no graph is being built, the
            out-of-place compiled kernel is preferred over the buffer path and
            over eager, with a one-strike fallback to eager on any failure.
            """
            if out is None and self.compile_kernels and not self._compile_failed \
                    and not (u.requires_grad or V.requires_grad):
                try:
                    if self._compiled_apply_C is None:
                        self._compiled_apply_C = torch.compile(
                            _apply_C_edge_traversal, fullgraph=False, dynamic=False)
                    return self._compiled_apply_C(
                        V, u, self.row.to(device=u.device),
                        self.col.to(device=u.device), self.t_values)
                except Exception:
                    # Never let the compiler change numerics or break a run.
                    self._compile_failed = True
                    self._compiled_apply_C = None
            row = self.row.to(device=u.device)
            col = self.col.to(device=u.device)
            tv = self.t_values.to(device=u.device, dtype=u.real.dtype)
            values = -(V[row] - V[col]).square() * tv
            contrib = values.to(dtype=u.dtype) * u[col]
            if out is None:
                out = torch.zeros_like(u)
            else:
                out.zero_()
            # In-place accumulation into a non-grad buffer: safe exactly when
            # the caller guarantees no backward will need these intermediates.
            return out.index_add_(0, row, contrib)

        def m_inner(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            m = self.mass_diag.to(device=u.device, dtype=u.real.dtype)
            return torch.sum(torch.conj(u) * (m * v))

        def m_norm(self, u: torch.Tensor) -> torch.Tensor:
            val = torch.real(self.m_inner(u, u))
            return torch.sqrt(torch.clamp(val, min=0.0))

        def _lanczos(self, u: torch.Tensor, apply_op, tau: float, inverse: bool = False,
                     m: Optional[int] = None, tol: Optional[float] = None) -> torch.Tensor:
            """Apply exp(+-i tau A) to a vector using M-inner-product Lanczos.

            Enhancement #1: the Krylov residual breakdown test is evaluated
            on-device (``b_j < tol`` as a device scalar) and the host is
            synchronized only every ``lanczos_check_every`` iterations and on
            the final iteration, instead of once per iteration.  Checking less
            often can only enlarge the Krylov subspace by a few vectors, i.e.
            it can only make the exponential more accurate.
            """
            m = self.m_krylov if m is None else int(m)
            tol = self.tol if tol is None else float(tol)
            check_every = self.lanczos_check_every
            beta0 = self.m_norm(u)
            if float(beta0.detach().item()) < 1e-15 or abs(tau) < 1e-18:
                return u.clone()

            q = u / beta0
            q_prev = torch.zeros_like(q)
            beta_prev = torch.zeros((), device=u.device, dtype=u.real.dtype)
            basis = []
            alpha = []
            beta = []
            actual_m = m

            for j in range(m):
                basis.append(q)
                w = apply_op(q)
                a_j = torch.real(self.m_inner(q, w))
                alpha.append(a_j)
                r = w - a_j * q
                if j > 0:
                    r = r - beta_prev * q_prev
                b_j = self.m_norm(r)
                is_last = (j == m - 1)
                # Single host sync per `check_every` iterations: the comparison
                # itself stays on-device; only the boolean crosses to the host.
                if is_last or (j + 1) % check_every == 0:
                    if is_last or bool((b_j < tol).item()):
                        actual_m = j + 1
                        break
                beta.append(b_j)
                q_prev, q = q, r / b_j
                beta_prev = b_j

            alpha_t = torch.stack(alpha[:actual_m])
            tri = torch.diag(alpha_t)
            if actual_m > 1:
                beta_t = torch.stack(beta[:actual_m - 1])
                tri = tri + torch.diag(beta_t, diagonal=1) + torch.diag(beta_t, diagonal=-1)

            sign = 1.0 if inverse else -1.0
            phase = torch.complex(
                torch.zeros((), device=u.device, dtype=u.real.dtype),
                torch.as_tensor(sign * tau, device=u.device, dtype=u.real.dtype)
            )
            coeff = torch.linalg.matrix_exp(phase * tri)
            e1 = torch.zeros(actual_m, device=u.device, dtype=u.dtype)
            e1[0] = beta0
            cvec = coeff @ e1
            Q = torch.stack(basis[:actual_m], dim=0)
            return torch.sum(cvec.unsqueeze(-1) * Q, dim=0)

        def kinetic_expm(self, u: torch.Tensor, tau: float, inverse: bool = False) -> torch.Tensor:
            return self._lanczos(u, lambda q: self.apply_T(q), tau, inverse=inverse)

        def potential_expm(self, u: torch.Tensor, V: torch.Tensor, bh: float, ch3: float,
                           inverse: bool = False) -> torch.Tensor:
            if abs(ch3) < 1e-18:
                phase = (1.0j if inverse else -1.0j) * bh
                return u * torch.exp(phase * V)
            # Enhancement #2: reuse one accumulation buffer across the Lanczos
            # iterates of this stage.  This is only safe when no autograd graph
            # is built through apply_W: under grad, every Lanczos iterate must
            # stay alive for the backward, so in-place reuse would corrupt it.
            # When compile_kernels is on, the compiled out-of-place kernel
            # (dispatched inside apply_C_from_V with out=None) is used instead.
            building_graph = u.requires_grad or V.requires_grad
            use_compiled = self.compile_kernels and not self._compile_failed
            buf = None
            if not building_graph and not use_compiled:
                buf = torch.zeros_like(u)
            def apply_W(q):
                return bh * (V * q) + ch3 * self.apply_C_from_V(V, q, out=buf)
            return self._lanczos(u, apply_W, 1.0, inverse=inverse)


else:  # pragma: no cover
    class SimplexTorchBackend:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for S6* Simplex adjoint")
