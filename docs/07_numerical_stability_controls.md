# Numerical Stability Controls for Fig.1 Scans

This note documents stability guards added to the fitting pipeline.

## Implemented safeguards

- Explicit SVD truncation in least squares:
  - `np.linalg.lstsq(..., rcond=1e-12)` (configurable)
- Design-matrix conditioning filter:
  - reject windows if `condition_number > max_condition_number`
- Overtone-amplitude sanity filter:
  - reject windows if `max(|C_n|, n>=1) / |C_0| > max_overtone_ratio`
- Low-signal window filter:
  - reject windows if `integral |h|^2 dt <= min_signal_norm`

## Updated APIs

- `solve_complex_lstsq(..., lstsq_rcond=...)`
- `fit_at_start_time(..., lstsq_rcond, max_condition_number, max_overtone_to_fund_ratio, min_signal_norm)`
- `scan_start_times_fixed_omegas(...)` now skips invalid windows instead of forcing unstable fits.
- `grid_search_remnant(...)` now stores:
  - `mismatch_grid` with `NaN` on invalid points
  - `valid_mask` for valid-fit locations

## Fig.1 script controls

`scripts/phase4_figure1_mismatch_vs_t0.py` now exposes:

- `--lstsq-rcond`
- `--max-condition-number`
- `--max-overtone-ratio`
- `--min-signal-norm`

Current policy:

- `figure1` defaults to paper-style settings (no extra guards).
- `figure1 --use-stability-guards` restores the stricter guard defaults when needed for diagnostics.
- `figure3` keeps wider `--max-overtone-ratio=1e4` to avoid over-filtering parameter-space scans.

Example (stricter late-time filtering):

```powershell
$env:PYTHONPATH="src"
python scripts/phase4_figure1_mismatch_vs_t0.py `
  --input-format sxs --sxs-location SXS:BBH:0305v2.0/Lev6 --no-download `
  --n-max 7 --t-end 90 `
  --max-overtone-ratio 100 `
  --output results/figure1_sxs0305v2_n7_ratio100.png
```
