# Fig.10 Dynesty Shared-Forward Short Compare

## What Was Done

The publication-facing dynesty script [phase6_figure10_posterior.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_figure10_posterior.py) was refactored to use the shared paper-faithful forward model in [paper_fig10.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/src/ringdown/paper_fig10.py).

This removes the previous Fig.10-specific drift:

- no manual `target_hpeak`
- no manual `target_snr_postpeak`
- no `complex_peak` / `plus_peak` switch
- no `complex` vs `plus` observation-channel split inside the publication path
- `Mf` prior now defaults to `[10, 100] Msun`

The script now evaluates the real detector-like channel using the shared observation and the frequency-domain likelihood with `channel="real"`.

## Verification

- [phase6_figure10_posterior.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_figure10_posterior.py) compiles
- [fd_likelihood.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/src/ringdown/fd_likelihood.py) now supports `FrequencyDomainRingdownLikelihood(channel="real")`
- tests:
  - [test_fd_likelihood_analytic_models.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/tests/test_fd_likelihood_analytic_models.py)
  - [test_paper_fig10_helpers.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/tests/test_paper_fig10_helpers.py)
  - current result: `5 passed`

## Short Compare

Unified short comparison output:

- [fig10_dynesty_paper_short_compare.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_dynesty_paper_short_compare.png)
- [fig10_dynesty_paper_short_compare_diag.csv](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_dynesty_paper_short_compare_diag.csv)
- [fig10_dynesty_paper_short_compare_traces.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_dynesty_paper_short_compare_traces.png)

Configuration:

- static dynesty
- `nlive = 60`
- `bound = single`
- `sample = rwalk`
- `walks = 16`
- `dlogz = 4.0`
- same shared observation for `N=0` and `N=3`

Result:

- `N=0`:
  - `mf_q50 = 95.764`
  - `chif_q50 = 0.876`
  - `ess_kish = 31.5`
  - `converged = 0`
- `N=3`:
  - `mf_q50 = 13.669`
  - `chif_q50 = 0.028`
  - `ess_kish = 1.5`
  - `converged = 0`

Interpretation:

- the shared forward model is no longer the dominant issue
- the cheap dynesty run is still badly underconverged
- `N=3` is much more fragile than `N=0` under the same short-run settings
- this keeps the current diagnosis consistent:
  - forward problem: largely fixed
  - posterior problem: still unresolved

## Separate Short Runs

For extra context, separate light runs were also generated:

- [fig10_dynesty_paper_short_N0.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_dynesty_paper_short_N0.png)
- [fig10_dynesty_paper_short_N0_diag.csv](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_dynesty_paper_short_N0_diag.csv)
- [fig10_dynesty_paper_short_N3.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_dynesty_paper_short_N3.png)
- [fig10_dynesty_paper_short_N3_diag.csv](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_dynesty_paper_short_N3_diag.csv)

These are still diagnostic-only and should not be interpreted as production posteriors.

## Remaining Legacy Entrypoints

Some exploratory Fig.10-era scripts still retain the old dual-scaling logic and are not yet aligned to the shared helper:

- [phase6_fig10_minimal_dynesty.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_fig10_minimal_dynesty.py)
- [phase6_fig10_kombine_full.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_fig10_kombine_full.py)
- [phase6_n3_emcee_move_compare.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_n3_emcee_move_compare.py)
- [phase6_n3_kombine_emcee_repro.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_n3_kombine_emcee_repro.py)
- [phase6_n3_mode_logposterior_probe.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_n3_mode_logposterior_probe.py)

They should be treated as legacy diagnostics until they are migrated to the shared forward model or explicitly renamed as non-paper experiments.
