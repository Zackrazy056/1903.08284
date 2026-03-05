from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AssessmentThresholds:
    truth_in_90_hpd_required: bool = True
    acceptance_mean_min: float = 0.12
    acceptance_min_per_walker_min: float = 0.02
    mf_edge_1pct_max: float = 0.10
    chi_edge_1pct_max: float = 0.05
    mf_iqr_fraction_of_prior_max: float = 0.70
    chi_iqr_fraction_of_prior_max: float = 0.70


def _credible_threshold(hist2d: np.ndarray, prob: float = 0.9) -> float:
    flat = np.asarray(hist2d, dtype=float).ravel()
    total = float(np.sum(flat))
    if total <= 0:
        return 0.0
    order = np.argsort(flat)[::-1]
    sorted_vals = flat[order]
    cdf = np.cumsum(sorted_vals) / total
    idx = int(np.searchsorted(cdf, prob, side="left"))
    idx = min(max(idx, 0), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _truth_hpd_rank(hist2d: np.ndarray, truth_mf: float, truth_chi: float, mf_range: tuple[float, float], chi_range: tuple[float, float]) -> float:
    # Returns posterior mass with density >= truth bin density (lower is better).
    h = np.asarray(hist2d, dtype=float)
    total = float(np.sum(h))
    if total <= 0:
        return 1.0

    mf_min, mf_max = mf_range
    chi_min, chi_max = chi_range
    if not (mf_min <= truth_mf <= mf_max and chi_min <= truth_chi <= chi_max):
        return 1.0

    n_chi, n_mf = h.shape
    mf_bin = int(np.floor((truth_mf - mf_min) / (mf_max - mf_min) * n_mf))
    chi_bin = int(np.floor((truth_chi - chi_min) / (chi_max - chi_min) * n_chi))
    mf_bin = min(max(mf_bin, 0), n_mf - 1)
    chi_bin = min(max(chi_bin, 0), n_chi - 1)
    truth_density = float(h[chi_bin, mf_bin])
    mass = float(np.sum(h[h >= truth_density])) / total
    return mass


def assess_phase_b(
    flat_samples: np.ndarray,
    acceptance_fraction: np.ndarray,
    truth_mf: float,
    truth_chi: float,
    mf_range: tuple[float, float],
    chi_range: tuple[float, float],
    thresholds: AssessmentThresholds,
    hist_bins: int = 90,
) -> dict[str, Any]:
    samples = np.asarray(flat_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError("flat_samples must have shape (n, >=2)")

    mf = samples[:, 0]
    chi = samples[:, 1]
    accept = np.asarray(acceptance_fraction, dtype=float)

    mf_min, mf_max = map(float, mf_range)
    chi_min, chi_max = map(float, chi_range)
    mf_width = mf_max - mf_min
    chi_width = chi_max - chi_min

    h2d, _, _ = np.histogram2d(
        mf,
        chi,
        bins=hist_bins,
        range=[[mf_min, mf_max], [chi_min, chi_max]],
    )
    h2d = h2d.T
    th90 = _credible_threshold(h2d, prob=0.9)

    n_chi, n_mf = h2d.shape
    mf_bin = int(np.floor((truth_mf - mf_min) / mf_width * n_mf))
    chi_bin = int(np.floor((truth_chi - chi_min) / chi_width * n_chi))
    mf_bin = min(max(mf_bin, 0), n_mf - 1)
    chi_bin = min(max(chi_bin, 0), n_chi - 1)
    truth_density = float(h2d[chi_bin, mf_bin])
    truth_in_90_hpd = bool(truth_density >= th90 and th90 > 0.0)

    mf_edge_w = 0.01 * mf_width
    chi_edge_w = 0.01 * chi_width
    mf_edge_1pct = float(np.mean((mf <= mf_min + mf_edge_w) | (mf >= mf_max - mf_edge_w)))
    chi_edge_1pct = float(np.mean((chi <= chi_min + chi_edge_w) | (chi >= chi_max - chi_edge_w)))

    mf_iqr = float(np.quantile(mf, 0.75) - np.quantile(mf, 0.25))
    chi_iqr = float(np.quantile(chi, 0.75) - np.quantile(chi, 0.25))
    mf_iqr_frac = mf_iqr / mf_width if mf_width > 0 else np.inf
    chi_iqr_frac = chi_iqr / chi_width if chi_width > 0 else np.inf

    acceptance_mean = float(np.mean(accept)) if len(accept) else 0.0
    acceptance_min_per_walker = float(np.min(accept)) if len(accept) else 0.0
    acceptance_max_per_walker = float(np.max(accept)) if len(accept) else 0.0

    truth_hpd_rank = _truth_hpd_rank(
        hist2d=h2d,
        truth_mf=float(truth_mf),
        truth_chi=float(truth_chi),
        mf_range=(mf_min, mf_max),
        chi_range=(chi_min, chi_max),
    )

    checks = {
        "truth_in_90_hpd": truth_in_90_hpd if thresholds.truth_in_90_hpd_required else True,
        "acceptance_mean": acceptance_mean >= thresholds.acceptance_mean_min,
        "acceptance_min_per_walker": acceptance_min_per_walker >= thresholds.acceptance_min_per_walker_min,
        "mf_edge_1pct": mf_edge_1pct <= thresholds.mf_edge_1pct_max,
        "chi_edge_1pct": chi_edge_1pct <= thresholds.chi_edge_1pct_max,
        "mf_iqr_fraction_of_prior": mf_iqr_frac <= thresholds.mf_iqr_fraction_of_prior_max,
        "chi_iqr_fraction_of_prior": chi_iqr_frac <= thresholds.chi_iqr_fraction_of_prior_max,
    }
    failed = [k for k, ok in checks.items() if not ok]

    metrics = {
        "truth_in_90_hpd": truth_in_90_hpd,
        "truth_hpd_rank": float(truth_hpd_rank),
        "hpd90_threshold_count": float(th90),
        "truth_bin_density_count": float(truth_density),
        "acceptance_mean": acceptance_mean,
        "acceptance_min_per_walker": acceptance_min_per_walker,
        "acceptance_max_per_walker": acceptance_max_per_walker,
        "mf_edge_1pct": mf_edge_1pct,
        "chi_edge_1pct": chi_edge_1pct,
        "edge_occupancy": float(mf_edge_1pct + chi_edge_1pct),
        "mf_iqr_fraction_of_prior": float(mf_iqr_frac),
        "chi_iqr_fraction_of_prior": float(chi_iqr_frac),
    }

    return {
        "pass": len(failed) == 0,
        "failed_checks": failed,
        "checks": checks,
        "metrics": metrics,
    }

