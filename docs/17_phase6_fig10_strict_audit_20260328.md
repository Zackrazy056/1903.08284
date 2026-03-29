# Fig.10 Strict Audit (2026-03-28)

This audit compares the current Phase 6 implementation against the paper target for `1903.08284` Fig. 10.

## 1. Paper Target

Fig. 10 is not a generic Bayesian ringdown figure. The paper target is:

- injection from `SXS:BBH:0305`,
- face-on source,
- detector response specialized to `F_+` optimal and `F_x = 0`,
- detector-frame total mass `72 Msun`,
- source distance `400 Mpc`,
- single-detector Advanced LIGO design PSD,
- post-peak optimal SNR about `42.3`,
- start time defined by the detector strain peak `t_h-peak`,
- `Delta t0 = 0` for Fig. 10,
- inference models with `N = 0, 1, 2, 3`,
- priors:
  - `Mf in [10, 100] Msun`
  - `chif in [0, 1]`
  - phases in `[0, 2pi]`
  - amplitudes in `[0.01, 250] h_peak`
  - `h_peak = 2e-21`

## 2. Implementation Assumptions

### 2.1 Injection calibration does not match the paper

Current strict emcee path:

- uses `target_hpeak = 2e-21`,
- then independently rescales the same signal again to force `target_snr_postpeak = 42.3`.

Relevant code:

- `scripts/phase6_fig10_emcee_full_strict.py:227-228`
- `scripts/phase6_fig10_emcee_full_strict.py:280`
- `scripts/phase6_fig10_emcee_full_strict.py:300-309`

Current dynesty/publication path does the same two-step rescaling:

- `scripts/phase6_figure10_posterior.py:35-37`
- `scripts/phase6_figure10_posterior.py:240-243`
- `scripts/phase6_figure10_posterior.py:297-307`

This is a major paper mismatch. In the paper, `72 Msun` and `400 Mpc` determine the signal amplitude physically, and the resulting post-peak SNR is then reported. In the current implementation, `h_peak` and post-peak SNR are both imposed externally.

### 2.2 Detector geometry / polarization is only approximated

Current strict emcee path constructs a "detector-like" channel by:

- rotating the complex `h22` so the peak is real,
- then taking the real part.

Relevant code:

- `scripts/phase6_fig10_emcee_full_strict.py:57-65`

This is not a full detector projection with explicit `F_+`, `F_x`, inclination, and polarization-angle handling. It is a plus-like heuristic channel.

Current dynesty/publication path is even farther from the paper detector study in its default frequency-domain mode:

- it only supports the complex strain channel for frequency-domain inference.

Relevant code:

- `scripts/phase6_figure10_posterior.py:69-72`
- `scripts/phase6_figure10_posterior.py:216-217`
- `scripts/phase6_figure10_posterior.py:232-235`

So the repository currently has two inconsistent Fig. 10 channels:

- strict emcee: approximate real detector-like channel,
- dynesty/publication path: complex `h22` channel.

Neither one is an exact implementation of the paper's detector-response assumptions.

### 2.3 Peak-time reference is inconsistent across code paths

Strict emcee uses a detector-like peak:

- `scripts/phase6_fig10_emcee_full_strict.py:273-285`

This is closer to the paper target.

But the dynesty/publication path defaults to:

- `--t0-reference complex_peak`

and even labels that as the paper-style setting.

Relevant code:

- `scripts/phase6_figure10_posterior.py:74-79`
- `scripts/phase6_figure10_posterior.py:227-249`

This is a direct mismatch with the paper, which defines `Delta t0` using the detector strain peak `t_h-peak`, not the complex-mode peak.

### 2.4 Distance appears in the interface but is not actually used

The dynesty/publication path exposes:

- `--distance-mpc 400.0`

Relevant code:

- `scripts/phase6_figure10_posterior.py:35`

But the present implementation rescales by `target_hpeak` and `target_snr_postpeak`; there is no corresponding physical distance-scaling path in the shown inference setup. This means the CLI advertises a paper parameter that is not governing the injection.

## 3. Numerical Details

### 3.1 The mass prior is wrong and is visibly truncating the posterior

Paper prior:

- `Mf in [10, 100] Msun`

Current implementation:

- `Mf in [50, 100] Msun`

Relevant code:

- `scripts/phase6_fig10_emcee_full_strict.py:234-239`
- `scripts/phase6_figure10_posterior.py:82-85`

This is not a small stylistic change. The production diagnostics show the failed strict-emcee posteriors piling up near the lower bound:

- `results/fig10_emcee_full_strict_prod1_diag.csv`

Examples:

- `N=0`: `mf_q16 = 50.275`
- `N=1`: `mf_q16 = 50.183`
- `N=2`: `mf_q16 = 50.345`
- `N=3`: `mf_q16 = 50.715`

This strongly suggests the current posterior geometry is being distorted by the prior floor.

### 3.2 Some parts are actually aligned with the paper

The following pieces look broadly correct:

- remnant frequencies are locked to Kerr QNM frequencies through `(Mf, chif)`:
  - `scripts/phase6_fig10_emcee_full_strict.py:172-178`
  - `scripts/phase6_figure10_posterior.py:409-420`
- amplitudes and phases are sampled directly:
  - `scripts/phase6_fig10_emcee_full_strict.py:137-145`
  - `scripts/phase6_figure10_posterior.py:380-387`
- phase prior is full-range `[0, 2pi]`
- amplitude prior uses `[0.01, 250] h_peak`
- Advanced LIGO design PSD fit is implemented:
  - `src/ringdown/fd_likelihood.py:42-59`
- the current analytic FD sign convention bug was already fixed:
  - `docs/12_phase6_fig10_fix1_status.md:5-16`

So the main blockers no longer look like a missing Kerr-frequency coupling or a simple sign error.

### 3.3 SNR normalization is not paper-identical

The code forces `target_snr_postpeak = 42.3` over the chosen PSD-weighted band:

- `scripts/phase6_fig10_emcee_full_strict.py:296-305`
- `scripts/phase6_figure10_posterior.py:295-307`

This is close in spirit to the paper number, but not identical to "inject the physically scaled waveform, then report the resulting post-peak optimal SNR".

It also means the achieved `h_peak` after SNR rescaling is not fixed to `2e-21` anymore.

### 3.4 N=3 is still genuinely multimodal after the analytic-FT fix

The remaining `N=3` failure is real, but it likely sits downstream of the injection/channel/prior mismatches above.

Relevant summaries:

- `docs/12_phase6_fig10_fix1_status.md:66-100`
- `results/paper_reproduction_audit_20260327.md:18-57`

Current strict-emcee production diagnostics:

- `results/fig10_emcee_full_strict_prod1_diag.csv`

show all `N=0..3` posteriors missing the true remnant region, with `N=3` not becoming the best case.

## 4. Plotting / Interpretation

The current publication-facing path and the strict emcee path are not plotting the same underlying experiment:

- one path is complex-channel and uses `complex_peak` by default,
- the other path is real-channel and uses a detector-like peak.

That means some of the visual disagreement is not a pure sampler issue; the figure-generation layer is mixing different physical assumptions.

## Prioritized Blockers

1. Injection calibration is not paper-faithful: `h_peak` and SNR are both imposed rather than derived from `72 Msun` and `400 Mpc`.
2. Detector-response implementation is inconsistent with the paper detector setup and differs across code paths.
3. Peak-time definition is inconsistent across code paths, and the dynesty/publication default is wrong for the paper.
4. The `Mf` prior floor is wrong (`50` instead of `10`) and is visibly truncating current posteriors.
5. Only after fixing 1-4 does the remaining `N=3` multimodality become worth re-auditing as a sampler-only problem.

## Best Next Audit Sequence

1. Replace the current dual rescaling (`target_hpeak` plus `target_snr_postpeak`) with a single physical injection path derived from `72 Msun` and `400 Mpc`.
2. Implement one paper-faithful detector channel only: face-on, `F_+` only, `F_x = 0`.
3. Standardize `Delta t0` on the detector strain peak everywhere.
4. Restore the paper prior `Mf in [10, 100] Msun`.
5. Re-run the low-cost `(Mf, chif)` likelihood-grid sanity checks before touching long MCMC again.
