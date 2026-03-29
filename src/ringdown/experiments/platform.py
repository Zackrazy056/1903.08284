from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    paper_target: str
    status: str
    forward_model: str
    likelihood_model: str
    parameterization: str
    output_tier: str
    entry_script: str
    include_constant_offset: bool | None = None
    physics_heuristics_enabled: bool | None = None
    notes: str = ""


@dataclass(frozen=True)
class SamplerConfig:
    sampler_name: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentPaths:
    repo_root: Path
    output_tier: str
    experiment_id: str

    @classmethod
    def for_repo(
        cls,
        experiment_id: str,
        output_tier: str,
        *,
        repo_root: Path | None = None,
    ) -> "ExperimentPaths":
        root = repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
        return cls(repo_root=root, output_tier=output_tier, experiment_id=experiment_id)

    @property
    def results_dir(self) -> Path:
        return self.repo_root / "results" / self.output_tier / self.experiment_id

    @property
    def docs_dir(self) -> Path:
        return self.repo_root / "docs"

    def ensure(self) -> "ExperimentPaths":
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self


class DiagnosticsWriter:
    def __init__(self, paths: ExperimentPaths) -> None:
        self.paths = paths.ensure()

    def path(self, filename: str) -> Path:
        return self.paths.results_dir / filename

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.path(filename)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_markdown(self, filename: str, text: str) -> Path:
        path = self.path(filename)
        path.write_text(text, encoding="utf-8")
        return path


def merge_missing_cli_args(argv: list[str], defaults: list[tuple[str, str]]) -> list[str]:
    merged = list(argv)
    for flag, value in defaults:
        if flag not in merged:
            merged.extend([flag, value])
    return merged


def invoke_legacy_script(script_path: Path, argv: list[str]) -> None:
    spec = importlib.util.spec_from_file_location(f"_legacy_{script_path.stem}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *argv]
        if hasattr(module, "main"):
            module.main()
        else:
            raise RuntimeError(f"legacy script {script_path} has no main()")
    finally:
        sys.argv = old_argv


def registry_row(
    config: ExperimentConfig,
    *,
    sampler: SamplerConfig | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    row = asdict(config)
    row["output_dir"] = output_dir
    if sampler is not None:
        row["sampler"] = asdict(sampler)
    return row
