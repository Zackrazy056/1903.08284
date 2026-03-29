# Fig.10 N=3 Logrel Candidate Status

## Candidate Path

The current production-facing `N=3` candidate is now:

- shared paper-faithful forward model
- `N = 3`
- `logrel_relphase` parameterization

Dedicated script:

- [phase6_fig10_n3_logrel_relphase_candidate.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_fig10_n3_logrel_relphase_candidate.py)

## Main Candidate Run

Output files:

- [fig10_n3_logrel_relphase_candidate.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_logrel_relphase_candidate.png)
- [fig10_n3_logrel_relphase_candidate_traces.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_logrel_relphase_candidate_traces.png)
- [fig10_n3_logrel_relphase_candidate_slices.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_logrel_relphase_candidate_slices.png)
- [fig10_n3_logrel_relphase_candidate.md](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_logrel_relphase_candidate.md)
- [fig10_n3_logrel_relphase_candidate.json](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_logrel_relphase_candidate.json)

Key numbers:

- `profile_max_logl = 872.38`
- sampled `max_logl = 859.61`
- `gap_to_profile_max = 12.77`
- `ess_kish = 271.05`
- `map_(Mf, chif) = (80.25, 0.752)`
- `map_epsilon = 0.173`

Interpretation:

- this is much better than the old baseline coordinates
- the run is numerically healthy enough to be informative
- but it is still not sitting in the truth-adjacent basin

## Follow-up Sampler Check

A targeted sampler variant was also tested:

- `sample = rslice`
- `bound = multi`

Output files:

- [fig10_n3_logrel_relphase_candidate_rslice.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_logrel_relphase_candidate_rslice.png)
- [fig10_n3_logrel_relphase_candidate_rslice.md](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_logrel_relphase_candidate_rslice.md)
- [fig10_n3_logrel_relphase_candidate_rslice.json](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_logrel_relphase_candidate_rslice.json)

Key numbers:

- `gap_to_profile_max = 18.68`
- `ess_kish = 1.02`
- `map_(Mf, chif) = (75.54, 0.719)`
- `map_epsilon = 0.101`

Interpretation:

- `rslice` moved the MAP closer to the truth-adjacent region
- but the run was badly under-sampled and not a production candidate

## Current Read

At this stage:

- `logrel_relphase` is still the correct default `N=3` parameterization
- the main remaining problem is sampler behavior inside that improved coordinate system
- the next useful production attempt should stay in `logrel_relphase`, but likely adjust sampler strategy rather than reverting coordinates

## Most Useful Next Move

Keep:

- shared forward model
- `N = 3`
- `logrel_relphase`

Then run one focused sampler study around that fixed setup, rather than reopening the parameterization question.
