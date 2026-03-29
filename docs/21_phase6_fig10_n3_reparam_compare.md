# Fig.10 N=3 Reparameterization Compare

## Goal

This experiment compares several short-run `N=3` parameterizations under the same:

- shared paper-faithful forward model,
- injected observation,
- detector channel,
- likelihood,
- dynesty budget.

The purpose is not to produce a publication posterior.
It is to test which parameterization is least hostile to the sampler.

## Script

- [phase6_fig10_n3_reparam_compare.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_fig10_n3_reparam_compare.py)

Compared parameterizations:

- `baseline`
  - linear amplitudes
  - absolute phases
- `logamp`
  - log amplitudes
  - absolute phases
- `relphase`
  - linear amplitudes
  - relative phases
- `logrel_relphase`
  - log fundamental amplitude
  - log overtone-to-fundamental amplitude ratios
  - relative phases

## Outputs

- [fig10_n3_reparam_compare.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_reparam_compare.png)
- [fig10_n3_reparam_compare.json](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_reparam_compare.json)
- [fig10_n3_reparam_compare.md](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_n3_reparam_compare.md)

## Reference

The script also computes a profile-likelihood reference over a local `(Mf, chif)` grid for the same observation.

- `profile_max_logl = 872.3765994694783`

This lets us compare each short sampler run against a fixed geometry benchmark.

## Results

- `baseline`
  - `gap_to_profile = 717.97`
  - `map_epsilon = 0.8377`
  - clearly misses the relevant basin
- `logamp`
  - `gap_to_profile = 15.34`
  - `map_epsilon = 0.2460`
  - huge improvement over baseline
- `relphase`
  - `gap_to_profile = 714.88`
  - `map_epsilon = 0.7747`
  - relative phase alone does not help
- `logrel_relphase`
  - `gap_to_profile = 2.51`
  - `map_epsilon = 0.0625`
  - best result by a wide margin

## Main Takeaway

The dominant improvement is not "relative phase" by itself.
The real win is a combined reparameterization that compresses the amplitude sector:

- log amplitude scale
- overtone amplitudes expressed relative to the fundamental
- relative phases

This is the first controlled short-run result that gets very close to the profiled `N=3` high-likelihood basin under the shared forward model.

## Recommendation

The next production-facing `N=3` attempts should use `logrel_relphase` as the default experimental parameterization.

Baseline linear-amplitude coordinates are now weak enough that they should no longer be treated as the default reference for short `N=3` runs.
