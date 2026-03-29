# Experiment Registry

This file is the human-readable companion to [experiments.yaml](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/experiments.yaml).

## Rules

- Anything intended to support a paper claim must have a registry entry.
- Production entrypoints must go through the registered wrapper under `scripts/production/` or `scripts/exploratory/`.
- If a script bypasses the shared forward model, it is not a production-path experiment.
- Shared physical conventions for the paper-faithful Fig.10 study are sealed in [conventions.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/src/ringdown/conventions.py) and consumed through [paper_fig10.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/src/ringdown/paper_fig10.py).

## Registered Experiments

| experiment_id | paper_target | status | canonical entrypoint | outputs |
| --- | --- | --- | --- | --- |
| `paper_fig10_dynesty_short` | Fig.10 detector-study posterior comparison (`N=0..3`) | `production` | [paper_fig10_dynesty_short.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/production/paper_fig10_dynesty_short.py) | [results/production/paper_fig10_dynesty_short](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/production/paper_fig10_dynesty_short) |
| `paper_fig10_profile_geometry` | Fig.10 posterior-geometry audit under shared forward | `exploratory` | [paper_fig10_profile_geometry.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/exploratory/paper_fig10_profile_geometry.py) | [results/exploratory/paper_fig10_profile_geometry](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/exploratory/paper_fig10_profile_geometry) |
| `paper_fig10_n0_anchor` | Fig.10 `N=0` sanity-anchor posterior under shared forward | `production` | [paper_fig10_n0_anchor.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/production/paper_fig10_n0_anchor.py) | [results/production/paper_fig10_n0_anchor](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/production/paper_fig10_n0_anchor) |
| `paper_fig10_n3_logrel_relphase_candidate` | Fig.10 `N=3` production-candidate posterior with improved parameterization | `production_candidate` | [paper_fig10_n3_logrel_relphase_candidate.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/production/paper_fig10_n3_logrel_relphase_candidate.py) | [results/production/paper_fig10_n3_logrel_relphase_candidate](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/production/paper_fig10_n3_logrel_relphase_candidate) |

## Conventions

- Detector channel: plus-only real channel from the face-on `(l=m=2)` mode.
- Peak definition: `t_h-peak`.
- PSD source: bilby aLIGO design PSD.
- Finite-duration convention: `duration_sec = tau_window_sec[-1] = (N-1)dt` on the uniform post-`t0` grid.
- Mass/time units: NR in geometric total-mass units, likelihood in SI seconds.
- `epsilon` normalization: divide by the initial total binary mass `M`, matching the paper.
- `constant offset`: disabled for all registered Fig.10 production / production-candidate / geometry paths.
- physics heuristics: disabled for all registered Fig.10 production / production-candidate / geometry paths.

## Directory Split

- `src/ringdown/`: reusable numerical and physical core.
- `src/ringdown/experiments/`: experiment objects, registry-aware wrappers, output conventions.
- `scripts/production/`: canonical paper-claim entrypoints.
- `scripts/exploratory/`: geometry probes, parameterization studies, candidate runs.
- `results/production/`: production outputs.
- `results/exploratory/`: research and diagnosis outputs.
