# 1903.08284 Overtones Reproduction

This repository reproduces the main results of:

Giesler et al., *Black hole ringdown: the importance of overtones* (arXiv:1903.08284v2).

## Goal

Reproduce the paper's theory pipeline, numerical fitting pipeline, and key figures, then extend to robust validation and uncertainty checks.

## Work Breakdown

Detailed plan: `docs/00_reproduction_plan.md`
Numerical stability controls: `docs/07_numerical_stability_controls.md`

Phases:

1. Theory and equation baseline (completed in this batch)
2. Data and preprocessing setup (completed in this batch)
3. Core fitting implementation (multi-overtone ringdown model, completed in this batch)
4. Figure reproduction (substantially completed: Fig.1 + Fig.2-style + Mf/chi_f landscape)
5. Validation and robustness checks
6. Optional extensions

## Current Status

- [x] Phase 1 complete: theory scope, symbols, equations, and implementation-ready specs documented.
- [x] Phase 2 complete: waveform IO, peak alignment, windowing, resampling, start-time grid.
- [x] Phase 3 complete: complex LS solver, mismatch utilities, `t0` scanning, remnant-grid wrapper.
- [~] Phase 4 in progress: figure scripts for mismatch-vs-`t0`, waveform/residual, and `(M_f, chi_f)` landscapes are ready; Fig.2 alignment re-audit completed.
- [x] Numerical stability controls added: SVD truncation (`rcond`), condition-number and overtone-amplitude guards.
- [~] Phase 5 in progress: multi-simulation epsilon distribution sweep script added.
- [ ] Phase 6

## Next Batch (Phase 4)

- Add multi-simulation sweep (dozens of SXS cases) to reproduce error distribution panel.
- Generate publication-style figure formatting to better match paper visuals.
- Add automated parity report against paper trends (`N=0` bias vs high-`N` recovery).

## GitHub Sync

To keep this repository synced to GitHub:

1. Configure remote once:
   - `git remote add origin <your-github-repo-url>`
2. Push current branch:
   - `git push -u origin master`
3. Run auto-sync loop (commit + push every 60s when changes exist):
   - `powershell -ExecutionPolicy Bypass -File scripts/github_autosync.ps1 -RepoPath . -Branch master -IntervalSeconds 60`

For one-shot sync:

- `powershell -ExecutionPolicy Bypass -File scripts/github_autosync.ps1 -RepoPath . -Branch master -RunOnce`
