# Reproduction Plan for arXiv:1903.08284v2

## Paper Work Decomposition (author-level tasks)

### A. Theory and model definition

- Define ringdown as linear superposition of QNMs for a chosen harmonic:
  - Dominant mode focus: `(l, m) = (2, 2)`
  - Overtone index: `n = 0 ... N`
- Use Kerr perturbation-theory frequencies:
  - `omega_lmn = omega_lmn(M_f, chi_f)`
- Parameterize complex amplitudes:
  - `C_lmn = |A_lmn| exp(-i phi_lmn)`

### B. Numerical fitting and objective metrics

- Time-domain unweighted linear least-squares fit for `C_22n`.
- Two analysis branches:
  - Fixed remnant (`M_f, chi_f`) from NR truth.
  - Free remnant (`M_f, chi_f`) inferred by minimizing mismatch.
- Compute mismatch:
  - `M = 1 - <h_NR, h_model> / sqrt(<h_NR,h_NR><h_model,h_model>)`
  - Inner product integral over `[t0, T]`.

### C. Physical inference checks

- Evaluate remnant parameter bias across start times `t0`.
- Compare `N=0` (fundamental only) vs multi-overtone models.
- Use error metric:
  - `epsilon = sqrt((delta M_f / M)^2 + (delta chi_f)^2)`

### D. Figure-level reproduction tasks

- Mismatch vs start-time curves for multiple `N`.
- Waveform and residual comparison at selected `t0` (e.g., peak strain).
- Mismatch landscape in `(M_f, chi_f)` plane.
- Error distributions across multiple simulations.
- Overtone amplitudes/relative contributions vs time.

### E. Robustness and caveats

- Detect late-time overfitting to numerical noise.
- Sensitivity to fitting window `[t0, T]`.
- Basis/systematics notes (spherical vs spheroidal mixing effects).

## Batch Plan

### Batch 1 (this batch): Theory + equation baseline

- Output:
  - exact reproduction scope
  - symbol table and equation list
  - algorithmic pseudocode for later implementation

### Batch 2: Data + preprocessing

- waveform ingestion API
- time alignment at peak strain
- common sampling and window definitions

### Batch 3: Core solver implementation

- linear LS solver for complex coefficients
- mismatch/inner-product utilities
- free-(M_f, chi_f) grid/optimizer wrapper

### Batch 4: Figure reproduction

- scripts to regenerate major paper figures
- plotting style and export

### Batch 5: Validation

- regression tests
- uncertainty and robustness checks
- discrepancy log vs paper values

### Batch 6: Optional extensions

- additional SXS systems
- noise injections and detectability studies
- no-hair-test-oriented diagnostics
