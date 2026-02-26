from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Waveform22


def _validate_arrays(t: np.ndarray, h: np.ndarray) -> None:
    if t.ndim != 1 or h.ndim != 1:
        raise ValueError("t and h must be 1D arrays")
    if t.size != h.size:
        raise ValueError("t and h must have same length")
    if t.size < 3:
        raise ValueError("waveform must contain at least 3 samples")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time array must be strictly increasing")


def load_waveform_csv(path: str | Path) -> Waveform22:
    """
    Load waveform from CSV with columns:
    t, re_h22, im_h22
    """
    path = Path(path)
    arr = np.loadtxt(path, delimiter=",", comments="#")
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("CSV must have at least 3 columns: t,re_h22,im_h22")
    t = arr[:, 0]
    h = arr[:, 1] + 1j * arr[:, 2]
    _validate_arrays(t, h)
    return Waveform22(t=t, h=h, source=str(path))


def load_waveform_npz(path: str | Path) -> Waveform22:
    """
    Load waveform from NPZ with arrays:
    - t
    - h_real
    - h_imag
    """
    path = Path(path)
    with np.load(path) as data:
        t = data["t"]
        h = data["h_real"] + 1j * data["h_imag"]
    _validate_arrays(t, h)
    return Waveform22(t=t, h=h, source=str(path))

