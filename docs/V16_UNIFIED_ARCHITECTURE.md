# S6* Splitting Framework: Release v16 Unified Architecture

## 1. Architectural Taxonomy

```text
                               S6* GEOMETRIC SPLITTING FRAMEWORK (V16)
                                               │
       ┌───────────────────────────────────────┼───────────────────────────────────────┐
       ▼                                       ▼                                       ▼
  BRANCH I: REAL                         BRANCH II: COMPLEX                      BRANCH III: SIMPLEX
(Hamiltonian Waves)                    (Parabolic Diffusion)                 (2D Triangular Meshes)
  ├── Real-Standard-33 (33 FFT)          ├── Complex-Standard-32 (32 FFT)       ├── S6* Simplex-Commutator (157 SpMV)
  └── Real-Stencil-12  (12 FFT)          ├── Complex-Flash-22    (22 FFT)       └── Baselines:
                                         └── Complex-Stencil-12  (12 FFT)           ├── Simplex-Yoshida-14 (112 SpMV)
                                                                                    └── Simplex-Parabolic  (128 SpMV)
```

## 2. Active Family Members
1. **`S6* Real-Standard-33`**: 13-stage canonical reference (33 FFTs/step).
2. **`S6* Real-Stencil-12`**: 8th-order compact spatial difference stencil (12 FFTs/step).
3. **`S6* Complex-Standard-32`**: 13-stage forward-positive (32 FFTs/step).
4. **`S6* Complex-Flash-22`**: 13-stage midpoint subflow with $c_1=c_7=0$ (22 FFTs/step).
5. **`S6* Complex-Stencil-12`**: 6-stage forward-positive with 8th-order spatial stencils (12 FFTs/step).
6. **`S6* Simplex-Commutator`**: 13-stage unified Lanczos potential stage on simplicial FE meshes (157 SpMVs/step).

## 3. Classical Baselines (Explicit Academic Attribution)
- **`Yoshida-6 (1990 Baseline)`**: 7-stage recursive fractal composition (14 FFTs/step with FSAL).
- **`BCChM-6 P6S7 (2013 Baseline)`**: 7-stage forward-positive complex composition (14 FFTs/step with FSAL).
- **`Castella-Blanes (2009 Baseline)`**: 5-stage forward-positive composition (10 FFTs/step).
- **`Simplex-Yoshida-14`**: Classical Yoshida-6 composition on simplicial meshes (112 SpMVs/step).
- **`Simplex-Parabolic (BCChM-6 Baseline)`**: 7-stage forward-positive complex composition on simplicial meshes (128 SpMVs/step).
