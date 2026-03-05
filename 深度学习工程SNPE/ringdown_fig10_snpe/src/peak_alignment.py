from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PeakAlignmentResult:
    t_peak_sec: float
    t_h_peak_sec: float
    delta_t_h_minus_peak_ms: float
    idx_peak: int
    idx_h_peak: int


def compute_peak_alignment(t_sec: np.ndarray, h_complex: np.ndarray, h_detector: np.ndarray) -> PeakAlignmentResult:
    idx_peak = int(np.argmax(np.abs(h_complex)))
    idx_h_peak = int(np.argmax(np.abs(h_detector)))
    t_peak = float(t_sec[idx_peak])
    t_h_peak = float(t_sec[idx_h_peak])
    delta_ms = 1.0e3 * (t_h_peak - t_peak)
    return PeakAlignmentResult(
        t_peak_sec=t_peak,
        t_h_peak_sec=t_h_peak,
        delta_t_h_minus_peak_ms=delta_ms,
        idx_peak=idx_peak,
        idx_h_peak=idx_h_peak,
    )


def save_peak_alignment_plot(
    t_sec: np.ndarray,
    h_complex: np.ndarray,
    h_detector: np.ndarray,
    result: PeakAlignmentResult,
    out_path: Path,
    zoom_window_ms: float = 3.0,
) -> None:
    t_ms = 1.0e3 * t_sec
    center_ms = 1.0e3 * result.t_peak_sec
    mask = (t_ms >= center_ms - zoom_window_ms) & (t_ms <= center_ms + zoom_window_ms)
    if not np.any(mask):
        mask = np.ones_like(t_ms, dtype=bool)

    amp_complex = np.abs(h_complex)
    amp_detector = np.abs(h_detector)
    if np.max(amp_complex) > 0:
        amp_complex = amp_complex / np.max(amp_complex)
    if np.max(amp_detector) > 0:
        amp_detector = amp_detector / np.max(amp_detector)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t_ms[mask], amp_complex[mask], lw=1.6, label=r"$|h|$ peak definition")
    ax.plot(t_ms[mask], amp_detector[mask], lw=1.4, label=r"$|h_{det}|$ peak definition")
    ax.axvline(1.0e3 * result.t_peak_sec, ls="--", lw=1.2, label=r"$t_{peak}$")
    ax.axvline(1.0e3 * result.t_h_peak_sec, ls=":", lw=1.2, label=r"$t_{h-peak}$")

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(
        "Peak alignment (Phase A)\n"
        + f"delta(t_h-peak - t_peak) = {result.delta_t_h_minus_peak_ms:.3f} ms"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

