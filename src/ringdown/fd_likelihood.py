from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rfft_continuous(x_t: np.ndarray, dt: float) -> np.ndarray:
    """
    Continuous-convention FFT approximation.

    Real input -> one-sided rFFT.
    Complex input -> full FFT.
    """
    if np.iscomplexobj(x_t):
        return np.fft.fft(x_t) * dt
    return np.fft.rfft(x_t) * dt


def continuous_ft_from_time_series(t_sec: np.ndarray, x_t: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """
    Approximate continuous Fourier transform on an arbitrary positive-frequency grid:
        X(f) = integral x(t) exp(-i 2*pi*f*t) dt
    """
    if t_sec.ndim != 1 or x_t.ndim != 1 or freqs_hz.ndim != 1:
        raise ValueError("t_sec, x_t, freqs_hz must be 1D")
    if t_sec.size != x_t.size:
        raise ValueError("t_sec and x_t must have identical length")
    if t_sec.size < 3:
        raise ValueError("need at least 3 time samples")
    # 数值上直接按连续傅里叶定义做积分，避免离散 FFT 的网格耦合假设。
    phase = np.exp(-2j * np.pi * t_sec[:, None] * freqs_hz[None, :])
    integrand = x_t[:, None] * phase
    return np.trapezoid(integrand, t_sec, axis=0)


def irfft_continuous(x_f: np.ndarray, n_time: int, dt: float) -> np.ndarray:
    """Inverse of rfft_continuous under numpy's FFT normalization."""
    return np.fft.irfft(x_f / dt, n=n_time)


def aligo_zero_det_high_power_psd(f_hz: np.ndarray, *, f_low_hz: float = 10.0) -> np.ndarray:
    """
    Analytic approximation to Advanced LIGO Zero-Detuned High-Power design PSD.

    Uses the commonly used closed-form fit:
        S_n(f) = 1e-49 * [x^-4.14 - 5 x^-2 + 111*(1 - x^2 + x^4/2)/(1 + x^2/2)]
        x = f / 215 Hz
    """
    f = np.asarray(f_hz, dtype=float)
    psd = np.full_like(f, np.inf, dtype=float)
    # 低于 f_low 的频段直接视作无效（inf），不会进入后续内积。
    valid = f >= f_low_hz
    if np.any(valid):
        x = f[valid] / 215.0
        term = (x ** -4.14) - 5.0 * (x**-2.0) + 111.0 * (1.0 - x**2.0 + 0.5 * x**4.0) / (1.0 + 0.5 * x**2.0)
        psd_valid = 1e-49 * term
        psd[valid] = np.clip(psd_valid, 1e-60, np.inf)
    return psd


def one_sided_inner_product(
    a_tilde: np.ndarray,
    b_tilde: np.ndarray,
    psd: np.ndarray,
    df: float,
    *,
    valid_mask: np.ndarray | None = None,
) -> float:
    if a_tilde.shape != b_tilde.shape or a_tilde.shape != psd.shape:
        raise ValueError("a_tilde, b_tilde, psd must have identical shapes")
    if df <= 0:
        raise ValueError("df must be positive")
    if valid_mask is None:
        valid = np.isfinite(psd) & (psd > 0)
    else:
        if valid_mask.shape != psd.shape:
            raise ValueError("valid_mask shape mismatch")
        valid = valid_mask & np.isfinite(psd) & (psd > 0)
    if not np.any(valid):
        raise ValueError("no valid frequency bins for inner product")
    # 这里是论文口径核心：<a,b> = 4 * df * Re[ sum a(f) b*(f) / S_n(f) ]。
    num = a_tilde[valid] * np.conjugate(b_tilde[valid]) / psd[valid]
    return float(4.0 * df * np.sum(np.real(num)))


def optimal_snr(
    h_tilde: np.ndarray,
    psd: np.ndarray,
    df: float,
    *,
    valid_mask: np.ndarray | None = None,
) -> float:
    val = one_sided_inner_product(h_tilde, h_tilde, psd, df, valid_mask=valid_mask)
    return float(np.sqrt(max(val, 0.0)))


def draw_colored_noise_rfft(
    rng: np.random.Generator,
    n_freq: int,
    psd: np.ndarray,
    df: float,
    *,
    enforce_real_endpoints: bool = True,
) -> np.ndarray:
    """
    Draw frequency-domain complex Gaussian noise compatible with one-sided PSD.

    For positive-frequency interior bins:
        E[|n_k|^2] = 0.5 * S_n(f_k) * df
    so Re/Im each has variance 0.25 * S_n * df.
    """
    if n_freq <= 1:
        raise ValueError("n_freq must be > 1")
    if psd.shape != (n_freq,):
        raise ValueError("psd shape mismatch")
    if df <= 0:
        raise ValueError("df must be positive")

    valid = np.isfinite(psd) & (psd > 0)
    # 在每个频点按 PSD 设定复高斯噪声方差，保证噪声模型与 S_n(f) 一致。
    std = np.zeros_like(psd, dtype=float)
    std[valid] = np.sqrt(0.25 * psd[valid] * df)
    noise = rng.normal(0.0, std, size=n_freq) + 1j * rng.normal(0.0, std, size=n_freq)
    invalid = ~valid
    noise[invalid] = 0.0

    if enforce_real_endpoints:
        # DC (and Nyquist for even-N FFT grids) are purely real for real time series.
        std0 = float(np.sqrt(max(0.5 * psd[0] * df, 0.0))) if np.isfinite(psd[0]) and psd[0] > 0 else 0.0
        noise[0] = rng.normal(0.0, std0) + 0.0j
        stdn = float(np.sqrt(max(0.5 * psd[-1] * df, 0.0))) if np.isfinite(psd[-1]) and psd[-1] > 0 else 0.0
        noise[-1] = rng.normal(0.0, stdn) + 0.0j
    return noise


def complex_ringdown_mode_tilde(
    freqs_hz: np.ndarray,
    omegas_rad_s: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    *,
    duration_sec: float,
    t0_sec: float = 0.0,
    include_finite_duration: bool = True,
) -> np.ndarray:
    """Analytic positive-frequency FT for a complex damped ringdown sum."""
    if freqs_hz.ndim != 1 or omegas_rad_s.ndim != 1 or amplitudes.ndim != 1 or phases.ndim != 1:
        raise ValueError("freqs_hz, omegas_rad_s, amplitudes, phases must be 1D")
    if not (omegas_rad_s.size == amplitudes.size == phases.size):
        raise ValueError("mode arrays must have identical size")
    if include_finite_duration and duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive when finite-duration windowing is enabled")

    c_n = amplitudes * np.exp(-1j * phases)
    delta = (2.0 * np.pi * freqs_hz)[None, :] + omegas_rad_s[:, None]
    if include_finite_duration:
        win = 1.0 - np.exp(-1j * delta * duration_sec)
    else:
        win = 1.0
    phase_shift = np.exp(-2j * np.pi * freqs_hz * t0_sec)
    h_modes = (-1j * c_n[:, None] * win) / delta
    return phase_shift * np.sum(h_modes, axis=0)


def real_ringdown_mode_tilde(
    freqs_hz: np.ndarray,
    omegas_rad_s: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    *,
    duration_sec: float,
    t0_sec: float = 0.0,
    include_finite_duration: bool = True,
) -> np.ndarray:
    """Analytic positive-frequency FT for a real detector-like ringdown channel."""
    if freqs_hz.ndim != 1 or omegas_rad_s.ndim != 1 or amplitudes.ndim != 1 or phases.ndim != 1:
        raise ValueError("freqs_hz, omegas_rad_s, amplitudes, phases must be 1D")
    if not (omegas_rad_s.size == amplitudes.size == phases.size):
        raise ValueError("mode arrays must have identical size")
    if include_finite_duration and duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive when finite-duration windowing is enabled")

    c_n = amplitudes * np.exp(-1j * phases)
    delta_pos = (2.0 * np.pi * freqs_hz)[None, :] + omegas_rad_s[:, None]
    delta_neg = (2.0 * np.pi * freqs_hz)[None, :] - np.conjugate(omegas_rad_s)[:, None]
    if include_finite_duration:
        win_pos = 1.0 - np.exp(-1j * delta_pos * duration_sec)
        win_neg = 1.0 - np.exp(-1j * delta_neg * duration_sec)
    else:
        win_pos = 1.0
        win_neg = 1.0
    phase_shift = np.exp(-2j * np.pi * freqs_hz * t0_sec)
    positive_branch = (-0.5j * c_n[:, None] * win_pos) / delta_pos
    negative_branch = (-0.5j * np.conjugate(c_n)[:, None] * win_neg) / delta_neg
    return phase_shift * np.sum(positive_branch + negative_branch, axis=0)


@dataclass(frozen=True)
class FrequencyDomainRingdownLikelihood:
    freqs_hz: np.ndarray
    d_tilde: np.ndarray
    psd: np.ndarray
    df: float
    duration_sec: float
    t0_sec: float = 0.0
    f_min_hz: float = 20.0
    f_max_hz: float | None = None
    include_finite_duration: bool = True
    channel: str = "complex"

    def __post_init__(self) -> None:
        if self.freqs_hz.shape != self.d_tilde.shape or self.freqs_hz.shape != self.psd.shape:
            raise ValueError("freqs_hz, d_tilde, psd must have identical shapes")
        if self.df <= 0:
            raise ValueError("df must be positive")
        if self.duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        if self.channel not in {"complex", "real"}:
            raise ValueError("channel must be either 'complex' or 'real'")

        if self.f_max_hz is None:
            vmax = np.inf
        else:
            vmax = float(self.f_max_hz)
        valid = (
            (self.freqs_hz >= self.f_min_hz)
            & (self.freqs_hz <= vmax)
            & np.isfinite(self.psd)
            & (self.psd > 0)
            & (self.freqs_hz > 0.0)
        )
        if not np.any(valid):
            raise ValueError("no valid bins after f-range and PSD masking")

        object.__setattr__(self, "_valid_mask", valid)
        f = self.freqs_hz[valid]
        object.__setattr__(self, "_f_calc", f)
        object.__setattr__(self, "_d_calc", self.d_tilde[valid])
        object.__setattr__(self, "_psd_calc", self.psd[valid])
        # t0 通过频域相位因子进入模型；t0 定义变化会直接改变后验几何。
        object.__setattr__(self, "_phase_shift", np.exp(-2j * np.pi * f * self.t0_sec))
        object.__setattr__(self, "_d_weighted", self.d_tilde[valid] / self.psd[valid])
        object.__setattr__(
            self,
            "_dd_const",
            float(4.0 * self.df * np.sum(np.real((self.d_tilde[valid] * np.conjugate(self.d_tilde[valid])) / self.psd[valid]))),
        )

    @property
    def valid_mask(self) -> np.ndarray:
        return self._valid_mask

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self._valid_mask))

    @property
    def f_calc(self) -> np.ndarray:
        return self._f_calc

    @property
    def d_calc(self) -> np.ndarray:
        return self._d_calc

    @property
    def psd_calc(self) -> np.ndarray:
        return self._psd_calc

    @property
    def dd_const(self) -> float:
        return self._dd_const

    def model_tilde(self, omegas_rad_s: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray) -> np.ndarray:
        if self.channel == "complex":
            return complex_ringdown_mode_tilde(
                self._f_calc,
                omegas_rad_s,
                amplitudes,
                phases,
                duration_sec=self.duration_sec,
                t0_sec=self.t0_sec,
                include_finite_duration=self.include_finite_duration,
            )
        return real_ringdown_mode_tilde(
            self._f_calc,
            omegas_rad_s,
            amplitudes,
            phases,
            duration_sec=self.duration_sec,
            t0_sec=self.t0_sec,
            include_finite_duration=self.include_finite_duration,
        )

    def log_likelihood(self, omegas_rad_s: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray) -> float:
        h_tilde = self.model_tilde(omegas_rad_s, amplitudes, phases)
        # 高斯噪声下 lnL = <d,h> - 1/2 <h,h> (+const)。
        d_h = 4.0 * self.df * np.sum(np.real(self._d_weighted * np.conjugate(h_tilde)))
        h_h = 4.0 * self.df * np.sum((np.abs(h_tilde) ** 2) / self._psd_calc)
        return float(d_h - 0.5 * h_h)
