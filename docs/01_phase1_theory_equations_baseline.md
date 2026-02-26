# Phase 1 Deliverable: Theory and Equations Baseline

## 1. Reproduction Scope

Primary target is the ringdown analysis in the paper for the dominant harmonic:

- signal mode: `h_22(t)`
- model family: overtone-expanded QNM model with `N = 0...7`
- key claim to reproduce:
  - including overtones enables accurate modeling from around peak strain,
  - fundamental-only model requires later start times to reduce bias.

## 2. Core Equations to Implement

## 2.1 Ringdown model

For `t >= t0`:

`h_22^N(t) = sum_{n=0}^{N} C_22n * exp(-i * omega_22n(M_f, chi_f) * (t - t0))`

where:

- `C_22n` are complex amplitudes (fit parameters in linear solve),
- `omega_22n` are complex QNM frequencies from perturbation theory.

## 2.2 Complex inner product

`<x, y> = integral_{t0}^{T} x(t) * conj(y(t)) dt`

## 2.3 Mismatch

`M = 1 - <h_NR, h_model> / sqrt(<h_NR,h_NR> * <h_model,h_model>)`

## 2.4 Remnant-parameter error metric

`epsilon = sqrt((delta M_f / M)^2 + (delta chi_f)^2)`

## 3. Two Analysis Branches

## 3.1 Fixed-remnant branch

- Inputs: NR-remnant `M_f, chi_f`.
- Use theory `omega_22n(M_f, chi_f)`.
- For each start time `t0` and overtone count `N`, solve linear least squares for `C_22n`.
- Output: mismatch curve `M(t0; N)`.

## 3.2 Free-remnant branch

- Treat `M_f, chi_f` as search variables.
- For each `(M_f, chi_f)` in grid or optimizer:
  - compute `omega_22n`,
  - solve for `C_22n`,
  - compute mismatch.
- Output:
  - best-fit `(M_f, chi_f)`,
  - error metric `epsilon`,
  - mismatch landscape in `(M_f, chi_f)` plane.

## 4. Implementation-Ready Algorithm Skeleton

1. Load NR `h_22(t)` and identify `t_peak` of strain amplitude.
2. Build start-time list (e.g., `t_peak - 25M` to `t_peak + 60M`).
3. For each overtone count `N`:
4. For each `t0`:
5. Build design matrix columns `exp(-i * omega_n * (t - t0))`.
6. Solve complex LS for `C_n`.
7. Build `h_model` and evaluate mismatch.
8. Store min mismatch and corresponding `t0`.
9. Repeat with free `(M_f, chi_f)` for inference tests.

## 5. Acceptance Criteria for Phase 1

- Scope is explicitly mapped to paper claims.
- Equations and symbols are unambiguous.
- Next phase can start coding without revisiting paper text.

Status: completed.
