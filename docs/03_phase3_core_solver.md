# Phase 3 Deliverable: Core Fitting Solver

## Implemented Modules

- `src/ringdown/metrics.py`
  - `inner_product(x, y, t)`
  - `mismatch(h_nr, h_model, t)`
  - `remnant_error_epsilon(...)`
- `src/ringdown/fit.py`
  - `build_design_matrix(t, omegas, t0)`
  - `solve_complex_lstsq(t, h, omegas, t0)`
- `src/ringdown/scan.py`
  - `fit_at_start_time(wf, omegas, t0, t_end=None)`
  - `scan_start_times_fixed_omegas(wf, omegas, t0_grid, t_end=None)`
  - `grid_search_remnant(...)` with pluggable `omega_provider`

## What Is Solved in Phase 3

- Complex linear least-squares solve for QNM amplitudes `C_n`.
- Paper-style mismatch computation on fitting window `[t0, T]`.
- Start-time scanning for fixed mode frequencies.
- Generic `(M_f, chi_f)` grid-search wrapper (frequency mapping externalized).

## Demo

Run synthetic validation:

```powershell
$env:PYTHONPATH="src"
python scripts/phase3_demo_solver.py
```

Expected behavior:

- increasing overtone count (`N`) lowers best mismatch,
- best-fit start time for the full synthetic model is close to ringdown onset.

## Next Step

Integrate actual Kerr QNM frequency provider `omega_22n(M_f, chi_f)` and connect to real NR/SXS waveform data.
