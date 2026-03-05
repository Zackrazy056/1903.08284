from __future__ import annotations

from pathlib import Path

import numpy as np


def aligo_design_psd_hz(f_hz: np.ndarray) -> np.ndarray:
    f = np.asarray(f_hz, dtype=float)
    psd = np.full_like(f, np.inf, dtype=float)

    mask = f >= 10.0
    if not np.any(mask):
        return psd

    x = f[mask] / 215.0
    s0 = 1.0e-49
    val = s0 * (
        np.power(x, -4.14)
        - 5.0 * np.power(x, -2.0)
        + 111.0 * (1.0 - np.power(x, 2.0) + 0.5 * np.power(x, 4.0)) / (1.0 + 0.5 * np.power(x, 2.0))
    )
    val[val <= 0.0] = np.inf
    psd[mask] = val
    return psd


def build_psd_interpolator_from_asd_file(psd_asd_path: Path):
    data = np.loadtxt(psd_asd_path, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"PSD/ASD file {psd_asd_path} must have at least 2 columns")

    freq = data[:, 0]
    asd = data[:, 1]
    psd = np.square(asd)
    f_min = float(np.min(freq))
    f_max = float(np.max(freq))

    def _psd_fn(f_hz: np.ndarray) -> np.ndarray:
        f = np.asarray(f_hz, dtype=float)
        out = np.full_like(f, np.inf, dtype=float)
        mask = (f >= f_min) & (f <= f_max)
        if np.any(mask):
            out[mask] = np.interp(f[mask], freq, psd)
        out[out <= 0.0] = np.inf
        return out

    return _psd_fn


def generate_colored_gaussian_noise(
    n_samples: int,
    dt: float,
    rng: np.random.Generator,
    psd_fn=aligo_design_psd_hz,
) -> np.ndarray:
    if n_samples < 2:
        return np.zeros(n_samples, dtype=float)

    freqs = np.fft.rfftfreq(n_samples, d=dt)
    df = freqs[1] - freqs[0]
    psd = psd_fn(freqs)

    n_tilde = np.zeros_like(freqs, dtype=np.complex128)

    # Interior positive-frequency bins are complex Gaussian.
    interior = np.arange(1, len(freqs) - (1 if n_samples % 2 == 0 else 0))
    finite = np.isfinite(psd[interior])
    idx = interior[finite]
    if idx.size:
        sigma = np.sqrt(0.25 * psd[idx] * df)
        n_tilde[idx] = sigma * (rng.normal(size=idx.size) + 1j * rng.normal(size=idx.size))

    # Nyquist bin (if present) is purely real.
    if n_samples % 2 == 0 and len(freqs) > 1 and np.isfinite(psd[-1]):
        sigma_nyq = np.sqrt(0.5 * psd[-1] * df)
        n_tilde[-1] = rng.normal() * sigma_nyq

    # Continuous-FT-like convention: n_tilde ~= dt * rfft(n_t)
    noise = np.fft.irfft(n_tilde / dt, n=n_samples)
    return np.asarray(noise, dtype=float)


def optimal_snr(signal: np.ndarray, dt: float, fmin_hz: float, psd_fn=aligo_design_psd_hz) -> float:
    n = len(signal)
    if n < 2:
        return 0.0

    freqs = np.fft.rfftfreq(n, d=dt)
    df = freqs[1] - freqs[0]
    h_tilde = dt * np.fft.rfft(signal)
    psd = psd_fn(freqs)

    mask = (freqs >= fmin_hz) & (freqs > 0.0) & np.isfinite(psd)
    if not np.any(mask):
        return 0.0

    rho2 = 4.0 * np.sum((np.abs(h_tilde[mask]) ** 2) / psd[mask]) * df
    return float(np.sqrt(max(rho2, 0.0)))
