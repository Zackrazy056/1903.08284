from __future__ import annotations

import numpy as np

from ringdown.conventions import finite_duration_from_uniform_time_samples
from ringdown.fd_likelihood import (
    FrequencyDomainRingdownLikelihood,
    complex_ringdown_mode_tilde,
    continuous_ft_from_time_series,
    draw_colored_noise_rfft,
    optimal_snr,
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


def test_frequency_domain_likelihood_real_channel_matches_explicit_expression() -> None:
    freqs = np.linspace(20.0, 512.0, 80)
    duration_sec = 0.08
    omegas = np.array([350.0 + 40.0j, 520.0 + 90.0j], dtype=complex)
    amps_true = np.array([1.2, 0.4], dtype=float)
    phases_true = np.array([0.3, 1.0], dtype=float)
    amps_test = np.array([1.1, 0.5], dtype=float)
    phases_test = np.array([0.4, 0.9], dtype=float)
    psd = np.linspace(1.0e-45, 3.0e-45, freqs.size)
    df = float(freqs[1] - freqs[0])

    d_tilde = real_ringdown_mode_tilde(
        freqs,
        omegas,
        amps_true,
        phases_true,
        duration_sec=duration_sec,
    )
    h_tilde = real_ringdown_mode_tilde(
        freqs,
        omegas,
        amps_test,
        phases_test,
        duration_sec=duration_sec,
    )

    like = FrequencyDomainRingdownLikelihood(
        freqs_hz=freqs,
        d_tilde=d_tilde,
        psd=psd,
        df=df,
        duration_sec=duration_sec,
        channel="real",
    )

    explicit = 4.0 * df * np.sum(np.real(d_tilde * np.conjugate(h_tilde) / psd)) - 2.0 * df * np.sum(
        (np.abs(h_tilde) ** 2) / psd
    )
    got = like.log_likelihood(omegas, amps_test, phases_test)
    assert np.isclose(got, explicit)


def test_optimal_snr_matches_numeric_ft_pipeline_under_current_duration_convention() -> None:
    dt = 2.0e-5
    n_time = 5000
    t = np.arange(n_time, dtype=float) * dt
    duration_sec = finite_duration_from_uniform_time_samples(t)
    freqs = np.linspace(20.0, 512.0, 220)
    omegas = np.array([410.0 + 65.0j], dtype=complex)
    amps = np.array([1.1], dtype=float)
    phases = np.array([0.35], dtype=float)
    psd = np.linspace(0.8e-45, 1.6e-45, freqs.size)
    df = float(freqs[1] - freqs[0])

    c_n = amps * np.exp(-1j * phases)
    h_t = np.real(np.sum(c_n[None, :] * np.exp(-1j * t[:, None] * omegas[None, :]), axis=1))

    analytic = real_ringdown_mode_tilde(
        freqs,
        omegas,
        amps,
        phases,
        duration_sec=duration_sec,
    )
    numeric = continuous_ft_from_time_series(t, h_t, freqs)

    snr_analytic = optimal_snr(analytic, psd, df)
    snr_numeric = optimal_snr(numeric, psd, df)
    assert np.isclose(snr_analytic, snr_numeric, rtol=3e-4)


def test_real_ringdown_mode_tilde_matches_numeric_pipeline_under_sealed_duration_convention() -> None:
    dt = 1.5e-5
    n_time = 6000
    t = np.arange(n_time, dtype=float) * dt
    duration_sec = finite_duration_from_uniform_time_samples(t)
    freqs = np.linspace(24.0, 640.0, 96)
    omegas = np.array([390.0 + 45.0j, 610.0 + 95.0j], dtype=complex)
    amps = np.array([1.15, 0.32], dtype=float)
    phases = np.array([0.1, 1.4], dtype=float)

    c_n = amps * np.exp(-1j * phases)
    h_t = np.real(np.sum(c_n[None, :] * np.exp(-1j * t[:, None] * omegas[None, :]), axis=1))

    analytic = real_ringdown_mode_tilde(
        freqs,
        omegas,
        amps,
        phases,
        duration_sec=duration_sec,
    )
    numeric = continuous_ft_from_time_series(t, h_t, freqs)
    np.testing.assert_allclose(analytic, numeric, rtol=3.2e-4, atol=1e-12)


def test_draw_colored_noise_rfft_matches_target_one_sided_psd_statistics() -> None:
    rng = np.random.default_rng(12345)
    n_freq = 33
    df = 0.5
    psd = np.linspace(1.0e-46, 3.0e-46, n_freq)
    draws = np.asarray(
        [draw_colored_noise_rfft(rng, n_freq=n_freq, psd=psd, df=df) for _ in range(6000)],
        dtype=complex,
    )

    interior_power = np.mean(np.abs(draws[:, 1:-1]) ** 2, axis=0)
    target_interior_power = 0.5 * psd[1:-1] * df
    assert np.allclose(interior_power, target_interior_power, rtol=0.08)

    interior_real_var = np.var(np.real(draws[:, 1:-1]), axis=0)
    interior_imag_var = np.var(np.imag(draws[:, 1:-1]), axis=0)
    target_component_var = 0.25 * psd[1:-1] * df
    assert np.allclose(interior_real_var, target_component_var, rtol=0.08)
    assert np.allclose(interior_imag_var, target_component_var, rtol=0.08)

    endpoint_target_var = 0.5 * psd[[0, -1]] * df
    endpoint_real_var = np.var(np.real(draws[:, [0, -1]]), axis=0)
    endpoint_imag_power = np.mean(np.abs(np.imag(draws[:, [0, -1]])) ** 2, axis=0)
    assert np.allclose(endpoint_real_var, endpoint_target_var, rtol=0.08)
    assert np.all(endpoint_imag_power < (1e-12 * endpoint_target_var + 1e-60))
