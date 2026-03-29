# Phase 5 (Part 2): Fig. 3 Paper-Style Reproduction

This note separates the paper-faithful `Fig. 3` histogram from the broader `Phase5` publication summary plot.

## Goal

Reproduce the paper's `Fig. 3` analysis:

- free-remnant error metric `epsilon`,
- multiple SXS simulations near the paper regime,
- compare `N = 0, 3, 7` at `t0 = t_peak`,
- render as a single-panel logarithmic histogram with paper-like styling.

## Recommended Analysis Pass

The earlier `Phase5` sweep used odd grid sizes, which place the NR remnant exactly on the center grid point. That can artificially create many `epsilon = 0` hits for the best-performing overtone model.

For `Fig. 3`, use even grid sizes so the true remnant is not hard-coded onto the search grid:

```powershell
$env:PYTHONPATH="src"
python scripts/phase5_sxs_error_distribution.py `
  --max-sims 31 `
  --sample-mode stratified_q `
  --mf-points 40 --chif-points 40 `
  --output-csv results/fig3_repro_31_even40.csv `
  --output-fig results/fig3_repro_31_even40_raw.png
```

## Paper-Style Plot

```powershell
python scripts/phase5_fig3_paper_style.py `
  --input-csv results/fig3_repro_31_even40.csv `
  --output results/fig3_repro_31_even40_paper.png `
  --summary-md results/fig3_repro_31_even40_summary.md
```

If a few SXS downloads fail transiently, you can recover them individually and merge them back into the main CSV. In the current workspace, this recovery step increased the successful sample count from `26` to `30`.

## Styling Choices

The dedicated plotting script aims to stay close to the paper panel:

- single panel only,
- serif font stack,
- inward ticks on all sides,
- logarithmic `epsilon` axis,
- outline histograms instead of filled bars,
- legend in the upper-left,
- compact figure size suitable for a paper column.

## Interpretation

A successful reproduction should preserve the paper's qualitative trend:

- `N = 0` has the broadest, largest-error distribution,
- `N = 3` shifts strongly left,
- `N = 7` is the left-most and narrowest distribution.

Exact medians need not match the paper perfectly because they depend on:

- the chosen simulation subset,
- search-grid resolution,
- current SXS catalog snapshot and cached waveform levels.

## Current Workspace Snapshot

The current reproduction run used:

- attempted sample count: `31`,
- successful sample count after recovery: `30`,
- grid: `40 x 40`,
- start time: `t0 = t_peak`,
- models: `N = 0, 3, 7`.

Generated files:

- `results/fig3_repro_31_even40.csv`
- `results/fig3_repro_31_even40_failures.csv`
- `results/fig3_repro_31_even40_recovery.csv`
- `results/fig3_repro_31_even40_recovery_failures.csv`
- `results/fig3_repro_31_even40_merged.csv`
- `results/fig3_repro_30_success_even40_paper.png`
- `results/fig3_repro_30_success_even40_paper.pdf`
- `results/fig3_repro_30_success_even40_summary.md`

Current summary from the merged 30-simulation run:

- `N = 0`: median `epsilon = 3.425e-01`, `p90 = 5.534e-01`
- `N = 3`: median `epsilon = 1.203e-02`, `p90 = 4.958e-02`
- `N = 7`: median `epsilon = 1.720e-03`, `p90 = 8.088e-03`

This reproduces the paper's qualitative `Fig. 3` ordering cleanly:

- `N = 0` is broad and biased,
- `N = 3` shifts strongly left,
- `N = 7` is the narrowest and lowest-error distribution.
