# Fig.10 Fix1 Status (2026-03-27)

## What Was Fixed

- Root cause confirmed: several Phase 6 scripts used the wrong analytic frequency-domain sign convention.
- The fix is now centralized in `src/ringdown/fd_likelihood.py`:
  - `complex_ringdown_mode_tilde(...)`
  - `real_ringdown_mode_tilde(...)`
- Updated scripts:
  - `scripts/phase6_fig10_emcee_full_strict.py`
  - `scripts/phase6_fig10_kombine_full.py`
  - `scripts/phase6_n3_kombine_emcee_repro.py`
  - `scripts/phase6_fig10_minimal_dynesty.py`
  - `scripts/phase6_fig11_emcee_compare.py`
- Added regression tests:
  - `tests/test_fd_likelihood_analytic_models.py`

## Verification

- `python -m py_compile` passed for the patched Phase 6 scripts and `src/ringdown/fd_likelihood.py`.
- `python -m pytest tests/test_fd_likelihood_analytic_models.py -q` passed (`2 passed`).

## New Experiment Results

### 1. Strict emcee medium run

Command result files:
- `results/fig10_emcee_full_strict_mid_fix1.png`
- `results/fig10_emcee_full_strict_mid_fix1_diag.csv`
- `results/fig10_emcee_full_strict_mid_fix1_samples_N{0,1,2,3}_emcee.npz`

Posterior medians:

| N | Mf q50 | chif q50 | epsilon72(q50) |
|---|---:|---:|---:|
| 0 | 87.933 | 0.756 | 0.2768 |
| 1 | 68.459 | 0.555 | 0.1369 |
| 2 | 70.308 | 0.682 | 0.0265 |
| 3 | 73.605 | 0.203 | 0.4944 |

Interpretation:
- This is a real improvement over the pre-fix pathological boundary-seeking behavior.
- `N=2` is now the best of the four.
- `N=3` is still not paper-faithful.

### 2. Complex-channel dynesty candidate

Command result files:
- `results/fig10_dynesty_complex_fix1.png`
- `results/fig10_dynesty_complex_fix1_diag.csv`
- `results/fig10_dynesty_complex_fix1_samples_N{0,1,2,3}.npz`

Posterior medians:

| N | Mf q50 | chif q50 | Converged flag |
|---|---:|---:|---:|
| 0 | 78.766 | 0.506 | True |
| 1 | 78.696 | 0.695 | False |
| 2 | 77.531 | 0.248 | False |
| 3 | 87.517 | 0.960 | False |

Interpretation:
- The complex-channel dynesty path is not currently better than the strict real-channel emcee path.
- The current issue is therefore not only the old sign bug.

### 3. Dedicated N=3 long-run consistency check

Command result files:
- `results/fig10_n3_strict_long_fix1_samples_N3_emcee.npz`
- `results/fig10_n3_strict_long_fix1_samples_N3_emcee_alt.npz`

The run ended with the built-in consistency gate:

- `|mf_q50(primary-alt)| = 9.1804`
- `|chif_q50(primary-alt)| = 0.1953`

Posterior medians from saved samples:

| Chain | Mf q50 | chif q50 |
|---|---:|---:|
| primary emcee | 63.841 | 0.524 |
| alt emcee (independent prior init) | 73.021 | 0.720 |

Interpretation:
- `N=3` is still strongly multimodal / sampler-sensitive under the current setup.
- The second chain lands much closer to truth than the first one, but the two chains are not mutually consistent.
- So the old analytic-model bug was real and important, but **it was not the only blocker** preventing a successful Fig.10 reproduction.

## Current Best Reading Of The Situation

- Fixed:
  - wrong analytic FT sign
  - missing finite-duration real/complex reusable models
  - lack of regression tests for the analytic FT formulas
- Improved:
  - strict emcee no longer collapses to obviously unphysical boundary modes for all `N`
- Still unresolved:
  - `N=3` does not robustly dominate
  - `N=3` chains remain mode-sensitive
  - complex-channel dynesty is still not paper-faithful

## Best Next Step

The highest-value next task is not more plotting; it is **N=3 posterior-geometry diagnosis**:

1. cluster and compare the two saved `N=3` long-run chains;
2. compute truth HPD rank for each mode separately;
3. decide whether the remaining blocker is:
   - sampler mixing,
   - signal/likelihood channel definition,
   - or a remaining data-product / baseline inconsistency.
