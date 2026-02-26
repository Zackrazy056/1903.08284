from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Waveform22:
    """Dominant harmonic waveform container."""

    t: np.ndarray
    h: np.ndarray
    source: str = ""

