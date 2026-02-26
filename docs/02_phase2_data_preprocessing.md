# Phase 2 Deliverable: Data and Preprocessing

## Implemented Modules

- `src/ringdown/types.py`
  - `Waveform22(t, h, source)` standard container.
- `src/ringdown/io.py`
  - `load_waveform_csv(path)` for `t,re_h22,im_h22`.
  - `load_waveform_npz(path)` for arrays `t,h_real,h_imag`.
  - input validation (shape, monotonic time, minimum length).
- `src/ringdown/preprocess.py`
  - `peak_time_from_strain(wf)`
  - `align_to_peak(wf)` -> waveform shifted to `t_peak=0`
  - `crop_time(wf, t_min, t_max)`
  - `resample_uniform(wf, dt)` (linear interpolation)
  - `build_start_time_grid(...)` with default range `[-25M, 60M]`

## Demo Script

- `scripts/phase2_demo_preprocess.py`
  - runs the full preprocessing chain and prints key diagnostics.

## Example

```powershell
$env:PYTHONPATH="src"
python scripts/phase2_demo_preprocess.py `
  --input data/examples/synthetic_h22.csv `
  --format csv `
  --dt 0.2 --crop-min -40 --crop-max 80
```

## Output Contract for Phase 3

After preprocessing, solver modules can assume:

- time array strictly increasing
- complex `h_22` available on chosen window
- optional uniform sampling available
- start-time grid ready for mismatch scan experiments
