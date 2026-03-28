# N=3 Follow-up Findings

## New Diagnostics Added

- `scripts/phase6_n3_posterior_geometry_diagnose.py`
- `scripts/phase6_n3_mode_logposterior_probe.py`
- `scripts/phase6_n3_emcee_move_compare.py`

## Strongest Current Conclusions

### 1. The remaining blocker is not well described as a single wrong peak

The repaired `N=3` posterior is strongly multimodal.

- Geometry diagnosis output:
  - `results/fig10_n3_geometry_fix1.md`
  - `results/fig10_n3_geometry_fix1.json`
  - `results/fig10_n3_geometry_fix1.png`
- Main findings:
  - chains are internally stable across time chunks;
  - chains assign very different mass to the truth-adjacent mode;
  - JS distance between chain posteriors is high;
  - overlap is low.

This points to **posterior multimodality plus poor inter-mode mixing**.

### 2. The truth-adjacent mode is real and competitive

The log-posterior probe output:

- `results/fig10_n3_mode_logpost_fix1.md`
- `results/fig10_n3_mode_logpost_fix1.json`

shows that the truth-adjacent component is not obviously disfavored by the repaired likelihood.

In this probe:

- the truth-adjacent mode achieved the best sampled `max log posterior`;
- the best false mode did **not** beat it by a meaningful log-posterior margin.

This weakens the hypothesis that the current repaired likelihood is still simply pushing to the wrong single solution.

### 3. A naive switch to `DEMove` does not help

Move-comparison outputs:

- `results/fig10_n3_move_compare_fix1.csv`
- `results/fig10_n3_move_compare_fix1.md`
- `results/fig10_n3_move_compare_fix1_stretch_extra.csv`
- `results/fig10_n3_move_compare_fix1_stretch_extra.md`

show:

- `stretch` can sometimes capture a substantial fraction of the truth-adjacent mode;
- `stretch` is still highly seed-sensitive;
- `DEMove` was worse in the tested runs, with much lower truth-mode fraction and much worse sampled max log-posterior.

Current small-sample basin-capture summary:

- `stretch`: 6 tested seeds, mean truth-mode fraction `0.1179`
- `stretch`: truth-mode fraction `> 0.2` in only `1 / 6` runs
- `stretch`: truth-mode fraction `> 0.1` in `3 / 6` runs
- `DEMove`: 2 tested seeds, mean truth-mode fraction `0.0192`

## Practical Reading

At this point, the evidence favors:

1. repaired likelihood physics is substantially better than before;
2. truth-adjacent solutions exist and can score competitively;
3. the present `N=3` failure is mainly about **mode weight allocation / mode capture / inter-mode exploration**.

## Best Next Step

The highest-value next task is now:

- build a **mode-aware `N=3` campaign** rather than one-chain-one-answer sampling.

Concretely:

1. run many independent `stretch` chains from prior;
2. classify each run into the learned mode set;
3. estimate the probability of landing in the truth-adjacent basin;
4. only then decide whether to:
   - adopt multi-start aggregation,
   - change parameterization,
   - or introduce a more serious sampler upgrade.
