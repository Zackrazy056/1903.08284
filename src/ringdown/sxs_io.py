from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import Waveform22


@dataclass(frozen=True)
class SXSRemnantInfo:
    sim_id: str
    level: str
    remnant_mass: float | None
    remnant_chif_z: float | None
    initial_total_mass: float | None


def _extract_remnant_info_from_metadata(
    meta: dict[str, Any], *, sim_id: str, level: str
) -> SXSRemnantInfo:
    remnant_mass = meta.get("remnant_mass")
    remnant_spin = meta.get("remnant_dimensionless_spin")
    mass1 = meta.get("initial_mass1")
    mass2 = meta.get("initial_mass2")

    if isinstance(remnant_spin, (list, tuple, np.ndarray)) and len(remnant_spin) >= 3:
        chif_z = float(remnant_spin[2])
    elif remnant_spin is None:
        chif_z = None
    else:
        chif_z = float(remnant_spin)

    total_mass = None
    if mass1 is not None and mass2 is not None:
        total_mass = float(mass1) + float(mass2)

    return SXSRemnantInfo(
        sim_id=sim_id,
        level=level,
        remnant_mass=None if remnant_mass is None else float(remnant_mass),
        remnant_chif_z=chif_z,
        initial_total_mass=total_mass,
    )


def _extract_remnant_info(sim: Any) -> SXSRemnantInfo:
    return _extract_remnant_info_from_metadata(
        sim.metadata,
        sim_id=str(getattr(sim, "sxs_id", "unknown")),
        level=str(getattr(sim, "Lev", "unknown")),
    )


def _extract_level_from_filename(path: Path) -> str:
    m = re.match(r"^(Lev[^_]+)_Strain_N2\.h5$", path.name)
    if m:
        return m.group(1)
    return "unknown"


def _find_best_cached_strain_file(cache_dir: Path) -> Path | None:
    candidates = list(cache_dir.glob("Lev*_Strain_N2.h5"))
    if not candidates:
        return None

    def level_key(path: Path) -> tuple[int, str]:
        m = re.match(r"^Lev(\d+)_Strain_N2\.h5$", path.name)
        if m:
            return int(m.group(1)), path.name
        return -1, path.name

    return sorted(candidates, key=level_key, reverse=True)[0]


def _load_waveform22_from_h5(
    strain_h5: Path, *, source: str, remnant_info: SXSRemnantInfo
) -> tuple[Waveform22, SXSRemnantInfo]:
    import sxs

    h = sxs.load(str(strain_h5), download=False)
    idx22 = h.index(2, 2)
    t = np.asarray(h.t, dtype=float)
    mode22 = np.asarray(h[:, idx22], dtype=complex)
    return Waveform22(t=t, h=mode22, source=source), remnant_info


def _load_metadata_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_from_cache_location(location: str) -> tuple[Waveform22, SXSRemnantInfo] | None:
    """
    Try to load from ~/.sxs/cache without touching remote catalog.
    """
    location_path = Path(location).expanduser()
    if location_path.exists():
        if location_path.is_dir():
            strain_h5 = _find_best_cached_strain_file(location_path)
            if strain_h5 is None:
                return None
            level = _extract_level_from_filename(strain_h5)
            meta = _load_metadata_json(location_path / f"{level}_metadata.json")
            info = _extract_remnant_info_from_metadata(
                {} if meta is None else meta,
                sim_id=location_path.name,
                level=level,
            )
            return _load_waveform22_from_h5(strain_h5, source=str(location_path), remnant_info=info)

        if location_path.is_file() and location_path.suffix.lower() == ".h5":
            level = _extract_level_from_filename(location_path)
            meta = _load_metadata_json(location_path.with_name(f"{level}_metadata.json"))
            info = _extract_remnant_info_from_metadata(
                {} if meta is None else meta,
                sim_id=location_path.parent.name,
                level=level,
            )
            return _load_waveform22_from_h5(location_path, source=str(location_path), remnant_info=info)

    if not location.startswith("SXS:"):
        return None

    sim_id, _, level = location.partition("/")
    cache_dir = Path.home() / ".sxs" / "cache" / sim_id.replace(":", "_")
    if not cache_dir.exists():
        return None

    if level:
        strain_h5 = cache_dir / f"{level}_Strain_N2.h5"
        if not strain_h5.exists():
            return None
        level_name = level
    else:
        strain_h5 = _find_best_cached_strain_file(cache_dir)
        if strain_h5 is None:
            return None
        level_name = _extract_level_from_filename(strain_h5)

    meta = _load_metadata_json(cache_dir / f"{level_name}_metadata.json")
    info = _extract_remnant_info_from_metadata(
        {} if meta is None else meta,
        sim_id=sim_id,
        level=level_name,
    )
    return _load_waveform22_from_h5(strain_h5, source=location, remnant_info=info)


def load_sxs_waveform22(
    location: str = "SXS:BBH:0305",
    *,
    download: bool = True,
) -> tuple[Waveform22, SXSRemnantInfo]:
    """
    Load dominant strain mode (2,2) from SXS simulation.
    """
    try:
        import sxs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sxs package is required for SXS loading. Install with: python -m pip install sxs"
        ) from exc

    if not download:
        cached = _load_from_cache_location(location)
        if cached is not None:
            return cached

    try:
        sim = sxs.load(location, download=download)
    except Exception:
        if download:
            raise
        cached = _load_from_cache_location(location)
        if cached is not None:
            return cached
        raise

    h = sim.h
    idx22 = h.index(2, 2)
    t = np.asarray(h.t, dtype=float)
    mode22 = np.asarray(h[:, idx22], dtype=complex)
    return Waveform22(t=t, h=mode22, source=str(location)), _extract_remnant_info(sim)
