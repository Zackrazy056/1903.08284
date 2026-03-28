# N=3 Geometry Diagnosis

## Purpose

`scripts/phase6_n3_posterior_geometry_diagnose.py` reads two saved long `N=3` chains and answers:

- whether the disagreement is dominated by posterior multimodality or by within-chain drift;
- how much posterior weight each chain assigns to the truth-adjacent mode;
- whether the truth falls inside each chain's 90% HPD region;
- how different the two chains are in remnant-parameter geometry.

## Default Inputs

- `results/fig10_n3_strict_long_fix1_samples_N3_emcee.npz`
- `results/fig10_n3_strict_long_fix1_samples_N3_emcee_alt.npz`

These are the two independent long chains from the repaired `Fig.10 strict` pipeline.

## Default Outputs

- `results/fig10_n3_geometry_fix1.png`
- `results/fig10_n3_geometry_fix1.json`
- `results/fig10_n3_geometry_fix1.md`

## Rerun

```powershell
$env:PYTHONPATH='src'
python scripts\phase6_n3_posterior_geometry_diagnose.py
```

## Current Takeaway

The present diagnosis points more strongly to **posterior multimodality plus poor inter-mode mixing** than to a single remaining sign or convention bug:

- the two long chains are internally stable across time chunks;
- they place very different weight on the truth-adjacent component;
- their 2D remnant posteriors have high JS distance and low Bhattacharyya overlap;
- the truth-adjacent mode exists, but it is not yet dominant enough to reproduce the paper claim.
