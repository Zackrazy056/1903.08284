# Fig.2 Re-Audit (Paper Consistency Check)

## Goal

Re-check differences between our Fig.2-style result and the paper, identify any basic algorithm problems, and regenerate Fig.2 with corrected settings.

## Audit Findings

### 1) Ringdown fit core algorithm: no basic error found

- Model uses paper form:
  - `h_22^N(t) = sum_{n=0}^N C_n exp(-i omega_n (t-t0))`
- Frequencies use Kerr QNMs from `qnm`, converted by `omega = (M_f omega_bar)/M_f`.
- Fit is complex unweighted least squares on `[t0, T]`.
- For `SXS:BBH:0305`, `N=7`, `t0=0`, mismatch remains `~6.6e-7`, consistent with paper-level behavior.

### 2) A real issue was found in the NR error-proxy alignment

Previous implementation only did phase alignment at one time point (after peak alignment).  
This can inflate `|Lev6-Lev5|` if there is small residual time shift.

Fix applied:

- added joint time+phase alignment on a short window:
  - `align_time_and_phase_by_window(...)` in `src/ringdown/compare.py`
- `scripts/phase4_figure2_waveform_residual.py` now defaults to this mode:
  - `--ref-align-mode window_time_phase`

## Regenerated Fig.2 (paper-closer setup)

Use v2.0 levels to match paper epoch:

```powershell
$env:PYTHONPATH="src"
python scripts/phase4_figure2_waveform_residual.py `
  --input-format sxs `
  --sxs-location SXS:BBH:0305v2.0/Lev6 `
  --sxs-reference-location SXS:BBH:0305v2.0/Lev5 `
  --no-download `
  --n-overtones 7 --t0 0 --t-end 90 `
  --output results/figure2_sxs0305v2_n7_t0peak.png
```

Observed output summary:

- `fit_mismatch = 6.574337e-07`
- `model_residual_stats = (min=2.183719e-05, median=1.354828e-04, max=3.659620e-04)`
- `nr_error_proxy_stats = (min=1.292591e-06, median=6.037653e-06, max=7.569092e-05)`

Output figure:

- `results/figure2_sxs0305v2_n7_t0peak.png`
