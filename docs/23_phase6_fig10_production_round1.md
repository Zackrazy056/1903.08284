# Phase6 Fig.10 Production Round 1

## Scope

This round was run entirely through the new registered production wrappers:

- `paper_fig10_n3_logrel_relphase_candidate`
- `paper_fig10_n0_anchor`

Both runs used the sealed shared-forward conventions:

- plus-only real detector channel
- `t_h-peak` start time
- bilby aLIGO design PSD
- finite-duration convention `duration_sec = (N-1)dt`
- `constant offset = False`
- `physics heuristics = False`

Because the SXS GitHub API was rate-limited, both runs used `--no-download` and relied on the local cache.

## N=3 Production Candidate

Output directory:
[paper_fig10_n3_logrel_relphase_candidate](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/production/paper_fig10_n3_logrel_relphase_candidate)

Key diagnostics from `fig10_n3_logrel_relphase_candidate.json`:

- `converged = 1`
- `ess_kish = 979.85`
- `gap_to_profile_max = 3.17`
- `map_(Mf, chif) = (75.61, 0.7309)`
- `q50_(Mf, chif) = (75.17, 0.7082)`
- `map_epsilon = 0.1055`
- `q50_epsilon = 0.0934`
- `truth_ball_frac = 0.0`

Interpretation:

- This is materially better than the earlier candidate with the same parameterization.
- The sampled posterior is now close to the shared profile reference in log-likelihood space.
- However, the main posterior mass is still not centered on the true remnant basin.

## N=0 Anchor

Output directory:
[paper_fig10_n0_anchor](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/production/paper_fig10_n0_anchor)

Key diagnostics from `fig10_n0_anchor_diag.csv`:

- `mf_q50 = 95.06`
- `chif_q50 = 0.8720`
- `ess_kish = 1455.5`
- `logzerr = 0.368`
- `converged = 0` under the script's strict criterion `logzerr <= 0.3`

Interpretation:

- Even under the shared production framework, `N=0` remains strongly biased away from the truth.
- The chain has high effective sample size, so this is not a "random wandering" result.
- The non-converged flag is driven by the conservative `logzerr` threshold, not by low ESS.

## Main Conclusion

Round 1 clears an important threshold:

- the paper-faithful posterior mechanism now appears in the production framework itself, not only in grid/profile studies.

Specifically:

- `N=0` stays strongly biased.
- `N=3` under `logrel_relphase` moves substantially closer to the truth-adjacent high-likelihood basin.

But the job is not finished:

- the `N=3` posterior is still offset from the true remnant values, so this is not yet a paper-level Fig.10 recovery.

## Best Next Move

Keep the platform fixed and only adjust the sampler campaign around the same `N=3 + logrel_relphase` production candidate.

Do not return to the old baseline parameterization.
