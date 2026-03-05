# Phase 4 (Part 1): Figure 1 Pipeline

This batch delivers the first real-data figure pipeline:

- real Kerr QNM frequencies from `qnm`
- real NR waveform from `sxs` (`SXS:BBH:0305` by default)
- mismatch-vs-start-time curves for `N=0...Nmax`

## New Modules

- `src/ringdown/frequencies.py`
  - `kerr_qnm_omega_lmn(...)`
  - `kerr_qnm_omegas_22n(...)`
  - `make_omega_provider_22(...)`
- `src/ringdown/sxs_io.py`
  - `load_sxs_waveform22(location="SXS:BBH:0305")`
  - extracts `(2,2)` strain mode and remnant metadata

## Figure Script

- `scripts/phase4_figure1_mismatch_vs_t0.py`

### Run with SXS data

```powershell
$env:PYTHONPATH="src"
python scripts/phase4_figure1_mismatch_vs_t0.py `
  --input-format sxs `
  --sxs-location SXS:BBH:0305v2.0/Lev6 `
  --no-download `
  --n-max 7 `
  --t-end 90 `
  --output results/figure1_sxs0305.png
```

### Run with local CSV/NPZ

```powershell
$env:PYTHONPATH="src"
python scripts/phase4_figure1_mismatch_vs_t0.py `
  --input-format csv `
  --input data/examples/synthetic_h22.csv `
  --mf 0.952 --chif 0.692 `
  --n-max 7 `
  --output results/figure1_from_csv.png
```

## Notes

- Script aligns waveform to peak strain (`t_peak -> 0`), then scans `t0` in `[-25M, 60M]` by default.
- For SXS input, `mf` and `chif` default to metadata remnant values.
- Default fitting now includes a complex constant-offset term `b` to absorb baseline drift.
- Use `--no-constant-offset` to recover the original pure-QNM basis.
- Defaults keep no extra numerical guards enabled.
- `--use-stability-guards` re-enables guard filters from stability audits if needed.
- With `--no-download`, loader first tries local `~/.sxs/cache` files to avoid catalog/network failures.
