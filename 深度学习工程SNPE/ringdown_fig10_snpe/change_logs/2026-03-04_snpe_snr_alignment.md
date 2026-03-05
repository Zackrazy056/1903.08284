# 2026-03-04 SNPE SNR Alignment Fixes

## 1. Scope
- `src/summarize.py`
- `src/snpe_train.py`
- `src/eval/snpe_rescue_checks.py`
- `src/npe_coldstart_round3_try1.py`
- `configs/snpe.yaml`

## 2. Action
- `summarize.py`
  - Updated whitening factor from `1/sqrt(Sn)` to `sqrt(4*df)/sqrt(Sn)`.
  - Added inline comment tying scaling to discrete inner-product/SNR convention.
- `snpe_train.py`
  - Split frequency controls into `snr_fmin_hz` (from fig10 spec) and `feat_fmin_hz` (from SNPE features).
  - Forced simulator SNR calculation to use `optimal_snr(..., fmin_hz=snr_fmin_hz)`.
  - Added diagnostics fields: `snr_estimator`, `snr_fmin_hz`, `feature_fmin_hz`.
  - Extended round log print with `rho_before_q50` and `rho_after_q50`.
- `snpe_rescue_checks.py`
  - Same SNR/feature fmin decoupling as training path.
  - Added explicit SNR diagnostics metadata (`snr_estimator`, `snr_fmin_hz`, `feature_fmin_hz`).
  - Made amplitude draw mode follow config (`uniform` or `log_uniform`) for parity checks.
- `npe_coldstart_round3_try1.py`
  - Added hard non-production labeling in report and console output.
  - Added `non_production_reasons` list for audit clarity.
- `configs/snpe.yaml`
  - Tightened `snr_conditioning.scale_min/max` from `1e-4/1e4` to `1e-3/1e2`.

## 3. Rationale
- Align training/rescue SNR diagnostics with Phase-A `optimal_snr` definition and immutable fig10 SNR integration band.
- Prevent hidden mismatch between feature-preprocessing band and SNR band.
- Keep feature scaling stable across `n_fft` choices.
- Reduce confusion from legacy coldstart script being interpreted as production evidence.
- Reduce risk of extreme SNR scaling outliers destabilizing training.

## 4. Validation Plan (executed after code edit)
- Run `snpe_rescue_checks.py` with N=3 quick settings and verify:
  - `simulated_signal_snr_before` remains high due to broad amplitude prior.
  - `simulated_signal_snr_after` median remains near target (42.3).
  - Report records `snr_estimator=noise.optimal_snr` and both fmin fields.
- Run `snpe_train.py` smoke for N=3 and verify per-round `rho_before_q50` and `rho_after_q50` in logs/JSON.

## 5. Risk / Follow-up
- Production full run can still be long on CPU fallback.
- If future configs change `features.fmin_hz`, check that SNR diagnostics still stay anchored to fig10 SNR band.

## 6. Validation Results (2026-03-04)
- Command:
  - `python src/eval/snpe_rescue_checks.py --config configs/snpe.yaml --N 3 --batch-size 128 --snr-samples 32 --overfit-n 10 --overfit-epochs 80 --output-prefix snpe_rescue_N3_prod_aligned_quick`
- Result:
  - `pass_all=true`.
  - `simulated_signal_snr_before.q50 = 3297.75`.
  - `simulated_signal_snr_after.q50 = 42.30`.
  - Report now records `snr_estimator=noise.optimal_snr`, `snr_fmin_hz=154.68`, `feature_fmin_hz=154.68`.

- Command:
  - `python src/snpe_train.py --config configs/snpe.yaml --N 3` (production settings)
- Result:
  - Timed out in this session after ~3.7h (tool timeout) before completion; no fresh production summary was written.

- Command:
  - `python src/snpe_train.py --config configs/snpe.yaml --N 3 --smoke-test --smoke-sims 30,30,20 --posterior-samples 300 --device cpu`
- Result:
  - Completed successfully.
  - Round logs show `rho_before_q50` high (~3e3) and `rho_after_q50=42.30` for all rounds.
  - `outputs/diagnostics/snpe_N3_train.json` includes new SNR metadata fields and per-round before/after stats.

## 7. Operational Note
- For full production run, execute with an external long-running session (screen/tmux/CI runner) to avoid terminal timeout.

## 8. Live Production Run Status
- A full production command (`python src/snpe_train.py --config configs/snpe.yaml --N 3`) is currently active as a background Python process.
- Detected process command line:
  - `python.exe 深度学习工程SNPE/ringdown_fig10_snpe/src/snpe_train.py --config 深度学习工程SNPE/ringdown_fig10_snpe/configs/snpe.yaml --N 3`
- Process ID observed during this session: `40928`.
- Note: this process was initiated from a timed shell invocation; terminal timeout did not terminate it.

## 9. GPU Migration (2026-03-04)
- Problem:
  - Host had NVIDIA GPU, but Python environment was `torch+cpu`, so SNPE could not use CUDA.
- Actions:
  1. Installed CUDA build of PyTorch with version compatible to `sbi`:
     - `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`, `torchaudio==2.5.1+cu124`
  2. Added GPU compatibility patch in `src/snpe_train.py`:
     - Ensure prior tensors are created on runtime training device.
     - Move `x_obs` tensor to runtime device.
     - Keep proposal-truncation mask indexing on candidate tensor device.
     - Return simulator features on same device as sampled `theta`.
  3. Validated with GPU smoke run:
     - `python src/snpe_train.py --config configs/snpe.yaml --N 3 --smoke-test --smoke-sims 30,30,20 --posterior-samples 300 --device cuda`
     - Completed successfully; per-round `rho_after_q50=42.30`.
  4. Started full N=3 production run on CUDA in background:
     - `python src/snpe_train.py --config configs/snpe.yaml --N 3 --device cuda`
- Live status at record time:
  - Running PID: `40536`
  - `nvidia-smi` shows process type `C` on GPU 0.

## 10. OOM Recovery and Resume (2026-03-04 22:01)
- Incident:
  - Previous full GPU run crashed with `torch.OutOfMemoryError` during round 1.
- Mitigation applied:
  - Config change: `optimizer.batch_size` reduced `512 -> 128`.
  - Config change: `runtime.simulation_data_device: cpu`.
  - Code change in `src/snpe_train.py`:
    - Added runtime handling of `simulation_data_device`.
    - Forced sampled training tensors (`theta`, `x`) to be stored on simulation data device.
    - Passed `data_device=simulation_data_device` to `inference.append_simulations`.
    - Kept model device as `cuda` for training compute.
    - Fixed proposal truncation boolean mask indexing to stay on candidate tensor device.
- Validation:
  - GPU smoke run completed successfully after patch:
    - `python src/snpe_train.py --config configs/snpe.yaml --N 3 --smoke-test --smoke-sims 30,30,20 --posterior-samples 300 --device cuda`
- Resume action:
  - Restarted full GPU production run:
    - `python src/snpe_train.py --config configs/snpe.yaml --N 3 --device cuda`
  - Current run log files:
    - `outputs/diagnostics/snpe_N3_prod_gpu_resume_20260304_220136.log`
    - `outputs/diagnostics/snpe_N3_prod_gpu_resume_20260304_220136.err.log`
  - Current run PID at start: `48532`.

## 11. Paper-level High-Sample Campaign (2026-03-05)
- Goal:
  - Increase simulation budget substantially toward paper-grade reproduction.
- New config added:
  - `configs/snpe_paper_highsample.yaml`
- Key changes vs baseline:
  - Rounds simulation counts: `100000 / 100000 / 50000` (was `30000 / 30000 / 15000`).
  - Posterior samples: `120000` (was `40000`).
  - Proposal probe samples: `30000` (was `12000`).
  - Draw batch size: `8192` (was `4096`).
  - Optimizer epochs/patience: `260 / 30` (was `200 / 20`).
  - Runtime: `device=cuda`, `simulation_data_device=cpu` (to control VRAM).
- Run started:
  - Command:
    - `python src/snpe_train.py --config configs/snpe_paper_highsample.yaml --N 3 --device cuda`
  - PID at launch: `49792`
  - Logs:
    - `outputs/diagnostics/snpe_N3_paper_highsample_20260305_163434.log`
    - `outputs/diagnostics/snpe_N3_paper_highsample_20260305_163434.err.log`
