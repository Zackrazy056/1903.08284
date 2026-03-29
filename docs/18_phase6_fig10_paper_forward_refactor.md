# Fig.10 Paper-Forward Refactor

## What Changed

This refactor collapses the previous Fig.10 forward model variants into one paper-faithful path:

- `SXS:BBH:0305v2.0/Lev6`
- detector-frame total mass `72 Msun`
- source distance `400 Mpc`
- injected detector channel = face-on, plus-only projection of the injected `(l=m=2)` mode
- detector peak defines `t_h-peak`
- `Delta t0 = t0 - t_h-peak`
- paper priors:
  - `Mf in [10, 100] Msun`
  - `chif in [0, 1]`
  - amplitudes in `[0.01, 250] h_peak`
  - phases in `[0, 2pi]`

The shared implementation now lives in [paper_fig10.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/src/ringdown/paper_fig10.py).

## Key Fixes

- Removed the old double injection scaling:
  - no more manual `target_hpeak`
  - no more manual `target_snr_postpeak`
- The Fig.10 strict emcee script now derives both quantities from the physical forward model.
- The `Mf` prior lower bound is now `10`, not `50`.
- The detector channel is no longer the old peak-phase-rotated `Re[h22 e^{-i phi_peak}]` approximation.

## Important Numerical Finding

The previous analytic PSD surrogate was too optimistic for the detector-study normalization.

Using the shared paper-faithful forward model together with bilby's Advanced LIGO design PSD gives:

- `h_peak = 2.1106918323003677e-21`
- `post-peak optimal SNR = 41.33566174654596`
- `t_h-peak - t_peak = -1.2998300950075645 M = -0.46096569860381276 ms`

These are close to the paper expectations:

- `h_peak ~ 2e-21`
- `post-peak SNR ~ 42.3`
- `t_h-peak ~ t_peak - 1.3 M`

## New Scripts

- [phase6_fig10_emcee_full_strict.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_fig10_emcee_full_strict.py)
  - now uses the shared paper-faithful injection path
- [phase6_fig10_paper_forward_sanity.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_fig10_paper_forward_sanity.py)
  - low-cost `(Mf, chif)` grid sanity check using real-channel profile fits

## Sanity Result

Primary sanity output:

- [fig10_paper_forward_sanity.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_paper_forward_sanity.png)
- [fig10_paper_forward_sanity.md](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_paper_forward_sanity.md)
- [fig10_paper_forward_sanity.json](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_paper_forward_sanity.json)

Smoke sanity output:

- [fig10_paper_forward_sanity_smoke.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_paper_forward_sanity_smoke.png)
- [fig10_paper_forward_sanity_smoke.md](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_paper_forward_sanity_smoke.md)
- [fig10_paper_forward_sanity_smoke.json](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_paper_forward_sanity_smoke.json)

The qualitative mechanism is back:

- `N=0`: best `(Mf, chif) = (78.0000, 0.6805)`, `epsilon = 1.318e-01`
- `N=3`: best `(Mf, chif) = (68.8475, 0.6873)`, `epsilon = 6.364e-03`

So at `Delta t0 = 0`, the simplified grid proxy now shows the expected trend:

- `N=0` is clearly biased away from the true remnant
- `N=3` moves the best-fit region back near the truth

## Remaining Work

- Refactor the dynesty/publication Fig.10 script onto the same shared forward model.
- Replace the low-cost profile-grid proxy with a likelihood-grid check if we want a stricter pre-MCMC diagnostic.
- Only after those are stable should we restart production posterior runs.
