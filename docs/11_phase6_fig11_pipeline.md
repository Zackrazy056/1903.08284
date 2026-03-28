# Phase 6 (Part 1): Fig.11 Shared-Injection Posterior Comparison

This batch adds a dedicated Fig.11-style script for the paper's Bayesian comparison:

- `N=0` analyzed at multiple positive `dt0` values after the peak strain,
- `N=3` analyzed at `dt0 = 0 ms`,
- one shared injection and one shared noise realization across all curves.

That last point is the key difference from a naive "run Fig.10 several times" workflow.
For Fig.11, later start times must naturally lose post-peak SNR instead of being rescaled back to the same target SNR.

## Script

- `scripts/phase6_fig11_emcee_compare.py`

## What The Script Does

1. Loads `SXS:BBH:0305` mode `(2,2)`.
2. Builds a detector-like real strain channel from the complex mode.
3. Uses the detector-strain peak as `t_h-peak`, then shifts that peak to `t=0`.
4. Creates one post-peak injection over `[0, t_end]`.
5. Rescales only that full `dt0=0` signal to the target post-peak SNR.
6. Draws one colored Gaussian noise realization in the time domain.
7. Reuses the same noisy data for all analysis windows:
   - `N=0, dt0 in {0, 3, 6, 10} ms`
   - `N=3, dt0 = 0 ms`
8. Runs emcee separately for each analysis window and overlays the 90% posterior contours.

## Paper-Parity Detail

The script is built to preserve the paper's Fig.11 logic:

- later `dt0` windows start from the same observed data stream,
- later windows therefore contain less ringdown power,
- the corresponding SNR decreases automatically,
- the broader late-time `N=0` contours are therefore a physical consequence of discarded signal power, not a plotting choice.

## Example Run

```powershell
$env:PYTHONPATH="src"
python scripts/phase6_fig11_emcee_compare.py `
  --no-download `
  --nwalkers 64 `
  --emcee-burnin 1000 `
  --emcee-steps 3000 `
  --output results/fig11_emcee_compare.png `
  --diag-csv results/fig11_emcee_compare_diag.csv `
  --samples-prefix results/fig11_emcee_compare_samples.npz
```

## Smoke Validation Run

The script was smoke-tested with a short chain:

```powershell
$env:PYTHONPATH="src"
python scripts/phase6_fig11_emcee_compare.py `
  --no-download `
  --nwalkers 24 `
  --emcee-burnin 40 `
  --emcee-steps 80 `
  --qnm-chi-grid-size 120 `
  --t-end 70 `
  --output results/fig11_smoke.png `
  --diag-csv results/fig11_smoke_diag.csv `
  --samples-prefix results/fig11_smoke_samples.npz
```

Observed window SNRs from `results/fig11_smoke_diag.csv`:

- `N=0, dt0=0 ms`: `42.30`
- `N=0, dt0=3 ms`: `25.95`
- `N=0, dt0=6 ms`: `14.22`
- `N=0, dt0=10 ms`: `5.41`
- `N=3, dt0=0 ms`: `42.30`

This confirms the intended shared-injection behavior: later `dt0` windows are not re-normalized and therefore lose SNR.

## Outputs

- figure: combined Fig.11-style contour plot
- diagnostics CSV:
  - requested and realized `dt0`,
  - acceptance,
  - autocorrelation estimates,
  - ESS,
  - per-window SNR,
  - posterior quantiles,
  - truth-in-90%-HPD flag
- optional per-run sample NPZ files

## Current Status

- pipeline implemented
- smoke test passed
- production-length chains not yet run

The next step is to run a longer production pass and check whether the paper statement
"truth enters the `N=0` 90% region only for sufficiently late start times" is reproduced quantitatively.
