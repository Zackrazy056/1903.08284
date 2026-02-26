from __future__ import annotations

import numpy as np

from ringdown.preprocess import build_start_time_grid
from ringdown.scan import scan_start_times_fixed_omegas
from ringdown.types import Waveform22


def make_synthetic_waveform() -> tuple[Waveform22, np.ndarray]:
    """
    Build synthetic signal:
    - t < 0: merger-like transient not captured by ringdown model
    - t >= 0: exact 3-mode ringdown with known frequencies
    """
    t = np.arange(-15.0, 90.0, 0.1)
    omegas_true = np.array([0.55 - 0.08j, 0.52 - 0.24j, 0.48 - 0.40j], dtype=complex)
    coeffs_true = np.array([0.9 + 0.2j, 0.5 - 0.3j, 0.4 + 0.1j], dtype=complex)

    h = np.zeros_like(t, dtype=complex)
    pre_mask = t < 0.0
    h[pre_mask] = 0.3 * np.exp((t[pre_mask] + 10.0) / 7.0) * np.exp(
        1j * (0.2 * t[pre_mask] + 0.008 * t[pre_mask] ** 2)
    )

    post_mask = ~pre_mask
    tp = t[post_mask]
    for c, w in zip(coeffs_true, omegas_true):
        h[post_mask] += c * np.exp(-1j * w * tp)

    rng = np.random.default_rng(1234)
    noise = 1e-4 * (rng.normal(size=t.size) + 1j * rng.normal(size=t.size))
    h = h + noise
    return Waveform22(t=t, h=h, source="synthetic_phase3"), omegas_true


def summarize(n: int, results: list) -> None:
    best = min(results, key=lambda r: r.mismatch)
    print(
        f"N={n} best_t0={best.t0:.2f} mismatch={best.mismatch:.6e} "
        f"residual_norm={best.residual_norm:.6e} n_samples={best.n_samples}"
    )


def main() -> None:
    wf, omegas_true = make_synthetic_waveform()
    t0_grid = build_start_time_grid(
        t_peak=0.0, m_total=1.0, rel_start_m=-10.0, rel_end_m=45.0, step_m=1.0
    )

    for n in [0, 1, 2]:
        omegas = omegas_true[: n + 1]
        results = scan_start_times_fixed_omegas(wf=wf, omegas=omegas, t0_grid=t0_grid)
        summarize(n=n, results=results)


if __name__ == "__main__":
    main()

