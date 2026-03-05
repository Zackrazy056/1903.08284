# ringdown_fig10_snpe

SNPE reproduction project for arXiv:1903.08284 Fig.10.

## Phase 0 Done Criteria

- Immutable reproduction spec is frozen in `configs/fig10.yaml`.
- Execution templates are prepared in:
  - `configs/mcmc_smoke.yaml`
  - `configs/snpe.yaml`
- Spec includes:
  - Injection setup (SXS:BBH:0305, mode (2,2), aLIGO design Gaussian noise).
  - Geometry and scaling (face-on, `F_plus=1`, `F_cross=0`, `M=72 Msun`, `D=400 Mpc`).
  - Inference model constraints (`delta_t0=0`, `N=0..3`, Eq.(1) Kerr QNM overtone model).
  - Priors and truth point (`Mf=68.5`, `chi_f=0.69`).
  - Post-peak SNR definition (`f>154.68 Hz`, target around `42.3`).

## Next

Phase A starts from generating `d_obs` and producing:

- `outputs/diagnostics/peak_alignment.png`
- `outputs/diagnostics/snr_report.json`

## Run Commands

- Phase A:
  - `python src/phase_a_build_observation.py --config configs/fig10.yaml`
- Phase B (MCMC smoke):
  - `python src/mcmc_smoke.py --config configs/mcmc_smoke.yaml`
  - `python src/mcmc_smoke.py --config configs/mcmc_smoke.yaml --assessment-only`
- Phase C (SNPE train + plot):
  - `python src/snpe_train.py --config configs/snpe.yaml --N 0,1,2,3`
  - `python src/snpe_infer_plot.py --config configs/snpe.yaml`
- Cold-start diagnostics (Round-3):
  - Try1 (data standardization baseline):
    - `python src/npe_coldstart_round3_try1.py --config configs/snpe.yaml --N 1 --n-samples 100`
  - Try2 (dynamic noise injection + strong overfit tuning):
    - `python src/npe_coldstart_round3_try2_dynamic_noise.py --config configs/snpe.yaml --N 1 --n-samples 100 --epochs-dynamic 20 --epochs-overfit 500 --batch-size 20 --lr 8e-4 --noise-floor-scale 0.0 --hidden-features 192 --num-transforms 8 --renorm-on-overfit-stage --output-prefix npe_coldstart_try2_dynamic20_overfit`
