from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass
class FixedFFTFeatureExtractor:
    n_time: int
    n_fft: int
    dt: float
    selected_indices: np.ndarray
    selected_freqs_hz: np.ndarray
    whiten_factors: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(len(self.selected_indices))

    @property
    def feature_dim(self) -> int:
        return int(2 * self.n_bins)

    def transform(self, signal_t: np.ndarray) -> np.ndarray:
        x = np.asarray(signal_t, dtype=float)
        if len(x) != self.n_time:
            raise ValueError(f"Expected signal length {self.n_time}, got {len(x)}")
        h_tilde = self.dt * np.fft.rfft(x, n=self.n_fft)
        z = h_tilde[self.selected_indices] * self.whiten_factors
        feat = np.concatenate([np.real(z), np.imag(z)], axis=0)
        return feat.astype(np.float32, copy=False)


def build_fixed_fft_feature_extractor(
    n_time: int,
    dt: float,
    psd_fn,
    fmin_hz: float,
    n_bins: int | None = None,
    n_freq_points: int | None = None,
    fmax_hz: float | None = None,
    n_fft: int | None = None,
) -> FixedFFTFeatureExtractor:
    if n_bins is not None and n_freq_points is not None:
        raise ValueError("Use only one of n_bins or n_freq_points.")

    requested_points = n_bins if n_bins is not None else n_freq_points
    if requested_points is not None:
        requested_points = int(requested_points)
        if requested_points <= 1:
            # <=1 means "do not downsample, keep full FFT mask".
            requested_points = None

    n_fft_use = int(n_fft) if n_fft is not None else int(n_time)
    if n_fft_use < int(n_time):
        raise ValueError(f"n_fft must be >= n_time ({n_time}), got {n_fft_use}.")

    freqs = np.fft.rfftfreq(n_fft_use, d=dt)
    psd = psd_fn(freqs)

    mask = (freqs >= fmin_hz) & np.isfinite(psd)
    if fmax_hz is not None:
        mask &= freqs <= fmax_hz
    idx_pool = np.where(mask)[0]

    if len(idx_pool) < 2:
        raise RuntimeError("Too few usable FFT bins after PSD/frequency masking.")

    if requested_points is None:
        idx = idx_pool
    else:
        if len(idx_pool) < requested_points:
            warnings.warn(
                f"Only {len(idx_pool)} usable FFT bins for features; reducing requested points from {requested_points} to {len(idx_pool)}.",
                RuntimeWarning,
            )
            requested_points = len(idx_pool)
        if requested_points < 2:
            raise RuntimeError("Too few usable FFT bins after PSD/frequency masking.")
        chosen = np.linspace(0, len(idx_pool) - 1, requested_points, dtype=int)
        idx = idx_pool[chosen]

    if requested_points is not None and len(idx_pool) > len(idx):
        warnings.warn(
            f"Downsampled FFT feature bins from {len(idx_pool)} to {len(idx)}.",
            RuntimeWarning,
        )

    # Keep feature scaling consistent with the discrete inner-product convention:
    # rho^2 = 4 * sum(|h_tilde|^2 / S_n) * df
    # so whitened bins use sqrt(4*df) / sqrt(S_n).
    df = float(freqs[1] - freqs[0])
    whiten = np.sqrt(4.0 * df) / np.sqrt(psd[idx])
    return FixedFFTFeatureExtractor(
        n_time=int(n_time),
        n_fft=int(n_fft_use),
        dt=float(dt),
        selected_indices=idx.astype(int),
        selected_freqs_hz=freqs[idx].astype(float),
        whiten_factors=whiten.astype(float),
    )
