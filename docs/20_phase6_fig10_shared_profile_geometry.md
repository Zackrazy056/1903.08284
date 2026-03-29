# Fig.10 Shared-Forward Profile Geometry

## Purpose

This diagnostic does not attempt a full posterior run. Instead, it asks a lower-level question:

- under the shared paper-faithful forward model,
- with the same injected observation,
- what does the profiled likelihood geometry look like for `N=0` and `N=3`?

The goal is to distinguish:

- a forward-model failure,
- a broad but well-behaved posterior,
- and a narrow / difficult / sampler-hostile posterior geometry.

## Outputs

- [fig10_profile_geometry_shared.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_profile_geometry_shared.png)
- [fig10_profile_geometry_shared_slices.png](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_profile_geometry_shared_slices.png)
- [fig10_profile_geometry_shared.json](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_profile_geometry_shared.json)
- [fig10_profile_geometry_shared.md](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/results/fig10_profile_geometry_shared.md)

Script:

- [phase6_fig10_profile_geometry_shared.py](/c:/Users/97747/GW%20QNM%20EMRI%20PROJECT/1903.08284%20overtones/scripts/phase6_fig10_profile_geometry_shared.py)

## Main Findings

Using a `40 x 40` `(Mf, chif)` grid:

- `N=0` best profile point:
  - `(Mf, chif) = (78.0000, 0.6756)`
  - truth relative drop: `Delta logL = -161.24`
- `N=3` best profile point:
  - `(Mf, chif) = (72.3590, 0.7167)`
  - truth relative drop: `Delta logL = -1.27`

This is the key result.

It says:

- for `N=0`, the truth is genuinely far outside the dominant profiled-likelihood region
- for `N=3`, the truth is already very close to the best profiled-likelihood region

So the current failure mode is no longer "the correct basin is absent".
It is much more consistent with "the correct basin exists, but the full posterior is hard to sample".

## Local Slice Reading

The local amplitude/phase slices for `N=3` were taken around the profiled best point.

Qualitatively, they show that:

- amplitude directions are steep
- phase directions are even steeper
- different overtones have very different characteristic scales

This supports the current diagnosis that the overtone amplitude/phase sector is a major source of sampler difficulty.

## Interpretation

At this stage, the evidence is internally consistent:

1. The shared forward model matches the paper target closely.
2. The low-cost profile geometry recovers the expected `N=0` vs `N=3` mechanism.
3. Cheap posterior runs still fail badly.

That combination strongly points to a sampling / parameterization bottleneck rather than a remaining leading-order forward mismatch.
