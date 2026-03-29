# 1903.08284

This repository now includes a **staged reproduction workflow scaffold** for Giesler et al. (2019), *Black Hole Spectroscopy with Overtones of the Gravitational-Wave Quadrupole Mode* (arXiv:1903.08284).

## Current status

The remaining work has been split into explicit stages so you can iterate/replace one block at a time:

1. `download` — pull event metadata/data and create raw-data workspace.
2. `preprocess` — transform strain/posterior files into fit-ready arrays.
3. `fit` — run overtone model comparison (`N=0..7`) and record mismatch metrics.
4. `validate` — regenerate key trends/figures and summarize comparison.

## Quick start

```bash
./scripts/bootstrap_repro.sh
source .venv/bin/activate
python scripts/run_repro.py --dry-run
python scripts/run_repro.py all
```

## Stage-by-stage usage

```bash
python scripts/run_repro.py download
python scripts/run_repro.py preprocess
python scripts/run_repro.py fit
python scripts/run_repro.py validate
```

## Output layout

By default, outputs are written to `artifacts/`:

- `artifacts/data/`
- `artifacts/posterior/`
- `artifacts/figures/`

Override with:

```bash
python scripts/run_repro.py all --outdir /path/to/output
```

## Notes

- The workflow intentionally starts as a conservative scaffold to make future substitutions low risk.
- Each stage currently creates concrete placeholder artifacts so downstream automation can be wired immediately.
- Next step is to replace placeholder stage commands with your exact analysis scripts/data sources.
