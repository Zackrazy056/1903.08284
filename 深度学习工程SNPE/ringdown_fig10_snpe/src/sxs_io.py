from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import sxs


@dataclass
class SXSMode22Data:
    sim_id: str
    lev: int
    t_M: np.ndarray
    h22: np.ndarray
    metadata: dict[str, Any]


def _sxs_cache_dir_for_sim(sim_id: str) -> Path:
    cache_name = sim_id.replace(":", "_").replace("/", "_")
    return Path.home() / ".sxs" / "cache" / cache_name


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _ensure_sidecar_files(sim_id: str, lev: int) -> None:
    # Force download of sidecar metadata/json. Some sxs versions expect generic
    # names in cache for inertial-frame transformation.
    sxs.load(f"{sim_id}/Lev{lev}:Strain_N2.json", download=True)
    sxs.load(f"{sim_id}/Lev{lev}:metadata.json", download=True)

    cache_dir = _sxs_cache_dir_for_sim(sim_id)
    lev_json = cache_dir / f"Lev{lev}_Strain_N2.json"
    lev_meta = cache_dir / f"Lev{lev}_metadata.json"
    generic_json = cache_dir / "Strain_N2.json"
    generic_meta = cache_dir / "metadata.json"

    _copy_if_exists(lev_json, generic_json)
    _copy_if_exists(lev_meta, generic_meta)


def _pick_lev(sim_entry: dict[str, Any], lev: str | int) -> int:
    lev_numbers = sim_entry.get("lev_numbers", [])
    if not lev_numbers:
        raise ValueError("No lev_numbers available in simulation entry")

    if isinstance(lev, int):
        lev_num = lev
    else:
        lev_str = str(lev).strip()
        if lev_str.lower() in {"highest", "max"}:
            lev_num = int(max(lev_numbers))
        elif lev_str.lower().startswith("lev"):
            lev_num = int(lev_str[3:])
        else:
            lev_num = int(lev_str)

    if lev_num not in lev_numbers:
        raise ValueError(f"Requested Lev{lev_num} not in available {lev_numbers}")
    return lev_num


def load_sxs_mode22(sim_id: str, lev: str | int = "highest") -> SXSMode22Data:
    sims = sxs.load("simulations", download=True)
    if sim_id not in sims:
        raise KeyError(f"{sim_id} not found in SXS simulations")

    sim_entry = sims[sim_id]
    lev_num = _pick_lev(sim_entry, lev)
    _ensure_sidecar_files(sim_id, lev_num)

    waveform = sxs.load(
        f"{sim_id}/Lev{lev_num}:Strain_N2.h5",
        download=True,
        transform_to_inertial=True,
        spin_weight=-2,
    )
    mode_idx = waveform.index(2, 2)
    t_M = np.asarray(waveform.t, dtype=float)
    h22 = np.asarray(waveform.data[:, mode_idx], dtype=np.complex128)

    metadata = dict(sxs.load(f"{sim_id}/Lev{lev_num}:metadata.json", download=True))
    return SXSMode22Data(sim_id=sim_id, lev=lev_num, t_M=t_M, h22=h22, metadata=metadata)

