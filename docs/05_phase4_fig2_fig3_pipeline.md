# Phase 4 (Part 2): Figure 2 and Exploratory Landscape Pipelines

This batch adds:

- waveform + residual plot at `t0=t_peak` (paper Fig. 2 style),
- exploratory mismatch landscapes in `(M_f, chi_f)` related to paper Fig. 4/5/6.

## New Utilities

- `src/ringdown/compare.py`
  - `window_waveform(...)`
  - `interp_complex(...)`
  - `phase_align_to_reference_at_tref(...)`

## New Scripts

- `scripts/phase4_figure2_waveform_residual.py`
  - fits `N` overtones at chosen `t0`,
  - plots waveform (Re part) and residual `|h_NR - h_model|`,
  - optional NR resolution proxy curve using SXS `Lev6` vs `Lev5`,
  - default baseline fit includes a complex constant-offset term `b`.

- `scripts/phase4_figure3_mf_chif_landscape.py`
  - performs grid search in `(M_f, chi_f)`,
  - plots `log10(mismatch)` contour maps,
  - marks best-fit point and NR remnant point,
  - default baseline fit includes a complex constant-offset term `b`,
  - is best treated as an exploratory landscape tool rather than the final paper-faithful `Fig. 4/5/6` renderer.

## Reproduction Commands

### Fig. 2 style (SXS:BBH:0305, N=7, t0=0)

```powershell
$env:PYTHONPATH="src"
python scripts/phase4_figure2_waveform_residual.py `
  --input-format sxs `
  --sxs-location SXS:BBH:0305/Lev6 `
  --sxs-reference-location SXS:BBH:0305/Lev5 `
  --n-overtones 7 --t0 0 --t-end 90 `
  --output results/figure2_sxs0305_n7_t0peak.png
```

Pure-QNM fallback:

```powershell
python scripts/phase4_figure2_waveform_residual.py `
  --input-format sxs `
  --sxs-location SXS:BBH:0305/Lev6 `
  --sxs-reference-location SXS:BBH:0305/Lev5 `
  --n-overtones 7 --t0 0 --t-end 90 `
  --no-constant-offset `
  --output results/figure2_sxs0305_n7_t0peak_no_offset.png
```

### Exploratory landscape contrast (`N=0` vs `N=7`)

```powershell
$env:PYTHONPATH="src"
python scripts/phase4_figure3_mf_chif_landscape.py `
  --input-format sxs --sxs-location SXS:BBH:0305/Lev6 `
  --n-overtones-list 0,7 --t0 0 --t-end 90 `
  --mf-half-width 0.05 --chif-half-width 0.12 `
  --mf-points 81 --chif-points 81 `
  --output results/figure3_paper_aligned_n0_n7.png
```

Notes:

- `--n-overtones-list 0,7` draws shared-colorbar side-by-side panels for quick comparison.
- Default fit includes constant offset `b`; add `--no-constant-offset` for the pure-QNM branch.
- The dedicated paper-faithful `Fig. 4/5/6` reproduction now lives in `scripts/phase4_fig456_paper_style.py`.

## Current Numeric Snapshot (Exploratory)

- `N=0, t0=0`:
  - best `(M_f, chi_f) = (1.00203294, 0.572085187)`,
  - `epsilon ≈ 1.3e-1` (strong bias).
- `N=7, t0=0`:
  - best `(M_f, chi_f) = (0.95203294, 0.692085187)`,
  - `epsilon ≈ 1.1e-16` (recovers the NR remnant on the chosen grid).

This exploratory script reproduces the overtone diagnostic qualitatively, but the final paper-style `Fig. 4/5/6` outputs should be taken from the dedicated strict script documented separately.
