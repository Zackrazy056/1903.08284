# Phase 5 (Part 1): Multi-Simulation Error Distribution

This batch adds a sweep script to reproduce the paper's distribution-style diagnostic for remnant-parameter error:

- fixed start time `t0 = t_peak`,
- compare `N={0,3,7}` overtone models,
- compute `epsilon = sqrt((delta Mf / M)^2 + (delta chi_f)^2` from free `(Mf, chi_f)` search.

## Script

- `scripts/phase5_sxs_error_distribution.py`

## Key Features

- SXS metadata-based filtering close to paper regime:
  - `SXS:BBH:*`
  - `q <= q_max`
  - near-aligned spins (`|chi_x|, |chi_y| <= aligned_tol`)
  - spin magnitudes `<= spin_abs_max`
- Per-simulation `(Mf, chi_f)` grid search for each `N`.
- Adaptive grid expansion when minimum lands on a boundary.
- Outputs:
  - per-fit CSV
  - failure CSV
  - epsilon histogram figure.

## Example runs

Small sanity run:

```powershell
$env:PYTHONPATH="src"
python scripts/phase5_sxs_error_distribution.py `
  --max-sims 6 --no-download `
  --mf-points 25 --chif-points 25 `
  --output-csv results/phase5_sweep_6_adaptive.csv `
  --output-fig results/phase5_sweep_6_adaptive.png
```

12-system run:

```powershell
$env:PYTHONPATH="src"
python scripts/phase5_sxs_error_distribution.py `
  --max-sims 12 --no-download `
  --mf-points 25 --chif-points 25 `
  --output-csv results/phase5_sweep_12_adaptive.csv `
  --output-fig results/phase5_sweep_12_adaptive.png
```

## Current snapshot

From `results/phase5_sweep_6_adaptive.csv`:

- `N=0`: median `epsilon ~ 1.24e-1`
- `N=3`: median `epsilon ~ 1.06e-2`
- `N=7`: median `epsilon ~ 2.80e-3`

This preserves the paper’s core trend: increasing overtones strongly reduces remnant-parameter bias at `t_peak`.
