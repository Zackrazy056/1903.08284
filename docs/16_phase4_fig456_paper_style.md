# Phase 4 (Part 3): Fig. 4, Fig. 5, and Fig. 6

This note separates the paper's remnant-landscape figures from the earlier exploratory landscape script.

## What Changed

The older `scripts/phase4_figure3_mf_chif_landscape.py` is useful for exploration, but it mixes together:

- paper and non-paper plotting styles,
- single-panel and comparison-panel workflows,
- the engineering baseline that includes a constant offset term `b`.

For the paper figures, a dedicated script is clearer:

- `scripts/phase4_fig456_paper_style.py`

## Paper Target

Reproduce the three mismatch maps shown for `SXS:BBH:0305`:

- `Fig. 4`: `N = 7`, `t0 = t_peak`
- `Fig. 5`: `N = 0`, `t0 = t_peak`
- `Fig. 6`: `N = 0`, `t0 = t_peak + 47M`

The dedicated plotting script fixes:

- panel-specific mass ranges,
- white NR crosshairs,
- paper-like serif styling,
- dark-to-light heat map with white low-mismatch valleys,
- annotations placed inside each panel,
- separate colorbar scaling for the early-time `N=0` panel.

## Recommended Reproduction Command

```powershell
$env:PYTHONPATH="src"
python scripts/phase4_fig456_paper_style.py `
  --sxs-location SXS:BBH:0305/Lev6 `
  --no-download `
  --output-prefix results/fig456_paper_repro_strict
```

By default this uses the pure-QNM paper model, not the engineering baseline with the extra constant-offset term.

## Current Workspace Outputs

- `results/fig456_paper_repro_strict.png`
- `results/fig456_paper_repro_strict.pdf`
- `results/fig456_paper_repro_strict_fig4.png`
- `results/fig456_paper_repro_strict_fig5.png`
- `results/fig456_paper_repro_strict_fig6.png`
- `results/fig456_paper_repro_strict.json`
- `results/fig456_paper_repro_strict.md`

## Current Numeric Snapshot

From the strict run in this workspace:

- `Fig. 4`: best fit near the NR remnant, `epsilon ~ 1.25e-3`
- `Fig. 5`: strongly biased early-time fundamental-only fit, `epsilon ~ 3.10e-1`
- `Fig. 6`: late-time fundamental-only fit returns close to the NR remnant, `epsilon ~ 6.80e-3`

This reproduces the paper's intended message:

- many overtones at the peak recover the true remnant,
- the fundamental mode alone at the peak is biased,
- the fundamental mode alone at late time recovers the remnant again.
