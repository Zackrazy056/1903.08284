from __future__ import annotations

import numpy as np

from ringdown.fd_likelihood import (
    complex_ringdown_mode_tilde,
    continuous_ft_from_time_series,
    real_ringdown_mode_tilde,
)


def test_complex_ringdown_mode_tilde_matches_direct_integral() -> None:
    freqs = np.linspace(20.0, 512.0, 80)
    duration_sec = 0.08
    t = np.linspace(0.0, duration_sec, 12000)
    omegas = np.array([350.0 + 40.0j, 520.0 + 90.0j], dtype=complex)
    amps = np.array([1.3, 0.45], dtype=float)
    phases = np.array([0.2, 1.1], dtype=float)

    c_n = amps * np.exp(-1j * phases)
    h_t = np.sum(c_n[None, :] * np.exp(-1j * t[:, None] * omegas[None, :]), axis=1)

    analytic = complex_ringdown_mode_tilde(
        freqs,
        omegas,
        amps,
        phases,
        duration_sec=duration_sec,
    )
    numeric = continuous_ft_from_time_series(t, h_t, freqs)

    rel_err = np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric)
    assert rel_err < 5e-5


def test_real_ringdown_mode_tilde_matches_direct_integral() -> None:
    freqs = np.linspace(20.0, 512.0, 80)
    duration_sec = 0.08
    t = np.linspace(0.0, duration_sec, 12000)
    omegas = np.array([350.0 + 40.0j, 520.0 + 90.0j], dtype=complex)
    amps = np.array([1.3, 0.45], dtype=float)
    phases = np.array([0.2, 1.1], dtype=float)

    c_n = amps * np.exp(-1j * phases)
    h_complex = np.sum(c_n[None, :] * np.exp(-1j * t[:, None] * omegas[None, :]), axis=1)
    h_t = np.real(h_complex)

    analytic = real_ringdown_mode_tilde(
        freqs,
        omegas,
        amps,
        phases,
        duration_sec=duration_sec,
    )
    numeric = continuous_ft_from_time_series(t, h_t, freqs)

    rel_err = np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric)
    assert rel_err < 5e-5
