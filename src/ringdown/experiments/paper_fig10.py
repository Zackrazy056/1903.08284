from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..conventions import paper_fig10_convention_summary
from ..paper_fig10 import (
    PaperFigure10Config,
    PaperFigure10Observation,
    PaperFigure10Signal,
    build_paper_fig10_signal,
    inject_paper_fig10_noise,
    paper_fig10_signal_diagnostics,
)
from .platform import (
    DiagnosticsWriter,
    ExperimentConfig,
    ExperimentPaths,
    SamplerConfig,
    invoke_legacy_script,
    merge_missing_cli_args,
    registry_row,
)


@dataclass(frozen=True)
class PaperFigure10ExperimentSpec:
    experiment: ExperimentConfig
    forward: PaperFigure10Config = field(default_factory=PaperFigure10Config)
    sampler: SamplerConfig | None = None
    default_outputs: tuple[tuple[str, str], ...] = ()
    extra_cli_args: tuple[str, ...] = ()

    def paths(self, *, repo_root: Path | None = None) -> ExperimentPaths:
        return ExperimentPaths.for_repo(
            self.experiment.experiment_id,
            self.experiment.output_tier,
            repo_root=repo_root,
        )

    def build_signal(self) -> PaperFigure10Signal:
        return build_paper_fig10_signal(self.forward)

    def build_observation(self, *, seed: int) -> PaperFigure10Observation:
        signal = self.build_signal()
        return inject_paper_fig10_noise(signal, np.random.default_rng(seed))

    def default_cli(self, *, repo_root: Path | None = None) -> list[tuple[str, str]]:
        paths = self.paths(repo_root=repo_root).ensure()
        return [(flag, str(paths.results_dir / filename)) for flag, filename in self.default_outputs]

    def registry_record(self, *, repo_root: Path | None = None) -> dict[str, Any]:
        row = registry_row(
            self.experiment,
            sampler=self.sampler,
            output_dir=str(self.paths(repo_root=repo_root).results_dir),
        )
        row["forward_config"] = asdict(self.forward)
        row["conventions"] = paper_fig10_convention_summary()
        return row


def run_registered_paper_fig10_script(
    spec: PaperFigure10ExperimentSpec,
    argv: list[str],
    *,
    repo_root: Path | None = None,
) -> None:
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
    merged = merge_missing_cli_args(list(argv), spec.default_cli(repo_root=root))
    if spec.extra_cli_args:
        merged.extend(spec.extra_cli_args)
    invoke_legacy_script(root / spec.experiment.entry_script, merged)


def write_paper_fig10_registry_snapshot(
    spec: PaperFigure10ExperimentSpec,
    *,
    repo_root: Path | None = None,
) -> Path:
    paths = spec.paths(repo_root=repo_root)
    writer = DiagnosticsWriter(paths)
    return writer.write_json("registry_snapshot.json", spec.registry_record(repo_root=repo_root))


PAPER_FIG10_DYNESTY_SHORT = PaperFigure10ExperimentSpec(
    experiment=ExperimentConfig(
        experiment_id="paper_fig10_dynesty_short",
        paper_target="Fig.10 detector-study posterior comparison (N=0..3)",
        status="production",
        forward_model="shared paper_fig10 forward",
        likelihood_model="real-channel frequency-domain Gaussian likelihood",
        parameterization="paper linear amplitudes + absolute phases",
        output_tier="production",
        entry_script="scripts/phase6_figure10_posterior.py",
        include_constant_offset=False,
        physics_heuristics_enabled=False,
        notes="Canonical production entrypoint for the shared-forward dynesty Fig.10 study.",
    ),
    sampler=SamplerConfig(
        sampler_name="dynesty",
        settings={"mode": "static", "channel": "real"},
    ),
    default_outputs=(
        ("--output", "fig10_dynesty_short.png"),
        ("--trace-output", "fig10_dynesty_short_traces.png"),
        ("--trace-all-prefix", "fig10_dynesty_short_trace_all.png"),
        ("--diag-csv", "fig10_dynesty_short_diag.csv"),
        ("--samples-prefix", "fig10_dynesty_short_samples.png"),
    ),
)


PAPER_FIG10_PROFILE_GEOMETRY = PaperFigure10ExperimentSpec(
    experiment=ExperimentConfig(
        experiment_id="paper_fig10_profile_geometry",
        paper_target="Fig.10 posterior-geometry diagnosis under shared forward",
        status="exploratory",
        forward_model="shared paper_fig10 forward",
        likelihood_model="real-channel frequency-domain Gaussian likelihood",
        parameterization="profiled amplitudes/phases at fixed (Mf, chif)",
        output_tier="exploratory",
        entry_script="scripts/phase6_fig10_profile_geometry_shared.py",
        include_constant_offset=False,
        physics_heuristics_enabled=False,
        notes="Exploratory geometry audit, not a publication figure generator.",
    ),
    default_outputs=(
        ("--output-heatmap", "fig10_profile_geometry_heatmap.png"),
        ("--output-slices", "fig10_profile_geometry_slices.png"),
        ("--summary-json", "fig10_profile_geometry.json"),
        ("--summary-md", "fig10_profile_geometry.md"),
    ),
)


PAPER_FIG10_N3_LOGREL_RELPHASE = PaperFigure10ExperimentSpec(
    experiment=ExperimentConfig(
        experiment_id="paper_fig10_n3_logrel_relphase_candidate",
        paper_target="Fig.10 N=3 production-candidate posterior under improved parameterization",
        status="production_candidate",
        forward_model="shared paper_fig10 forward",
        likelihood_model="real-channel frequency-domain Gaussian likelihood",
        parameterization="log amplitude + relative-to-fundamental amplitudes + relative phases",
        output_tier="production",
        entry_script="scripts/phase6_fig10_n3_logrel_relphase_candidate.py",
        include_constant_offset=False,
        physics_heuristics_enabled=False,
        notes="Current best N=3 parameterization candidate; still exploratory until stable contour recovery.",
    ),
    sampler=SamplerConfig(
        sampler_name="dynesty",
        settings={"mode": "static", "channel": "real", "parameterization": "logrel_relphase"},
    ),
    default_outputs=(
        ("--output", "fig10_n3_logrel_relphase_candidate.png"),
        ("--trace-output", "fig10_n3_logrel_relphase_candidate_traces.png"),
        ("--local-slices-output", "fig10_n3_logrel_relphase_candidate_slices.png"),
        ("--summary-json", "fig10_n3_logrel_relphase_candidate.json"),
        ("--summary-md", "fig10_n3_logrel_relphase_candidate.md"),
        ("--samples-output", "fig10_n3_logrel_relphase_candidate_samples.npz"),
    ),
)


PAPER_FIG10_N0_ANCHOR = PaperFigure10ExperimentSpec(
    experiment=ExperimentConfig(
        experiment_id="paper_fig10_n0_anchor",
        paper_target="Fig.10 N=0 sanity-anchor posterior under the shared forward",
        status="production",
        forward_model="shared paper_fig10 forward",
        likelihood_model="real-channel frequency-domain Gaussian likelihood",
        parameterization="paper linear amplitudes + absolute phases",
        output_tier="production",
        entry_script="scripts/phase6_figure10_posterior.py",
        include_constant_offset=False,
        physics_heuristics_enabled=False,
        notes="Anchor run used only to confirm that the production framework still recovers the expected N=0 bias.",
    ),
    sampler=SamplerConfig(
        sampler_name="dynesty",
        settings={"mode": "static", "channel": "real", "n_values": [0]},
    ),
    default_outputs=(
        ("--output", "fig10_n0_anchor.png"),
        ("--trace-output", "fig10_n0_anchor_traces.png"),
        ("--trace-all-prefix", "fig10_n0_anchor_trace_all.png"),
        ("--diag-csv", "fig10_n0_anchor_diag.csv"),
        ("--samples-prefix", "fig10_n0_anchor_samples.png"),
    ),
    extra_cli_args=("--n-values", "0"),
)


PAPER_FIG10_REGISTRY = (
    PAPER_FIG10_DYNESTY_SHORT,
    PAPER_FIG10_PROFILE_GEOMETRY,
    PAPER_FIG10_N0_ANCHOR,
    PAPER_FIG10_N3_LOGREL_RELPHASE,
)


def paper_fig10_registry_markdown(*, repo_root: Path | None = None) -> str:
    lines = [
        "# Experiment Registry",
        "",
        "| experiment_id | paper_target | status | forward_model | likelihood_model | parameterization | output_dir |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for spec in PAPER_FIG10_REGISTRY:
        row = spec.registry_record(repo_root=repo_root)
        lines.append(
            "| {experiment_id} | {paper_target} | {status} | {forward_model} | {likelihood_model} | {parameterization} | {output_dir} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Production entrypoints should use these registered wrappers instead of calling legacy phase scripts directly.",
            "Shared physical conventions are inherited from `ringdown.conventions.PAPER_FIG10_CONVENTIONS` and `ringdown.paper_fig10`.",
        ]
    )
    return "\n".join(lines) + "\n"


def paper_fig10_forward_snapshot(*, seed: int = 12345) -> dict[str, Any]:
    spec = PAPER_FIG10_DYNESTY_SHORT
    observation = spec.build_observation(seed=seed)
    return {
        "registry": spec.registry_record(),
        "signal_diagnostics": paper_fig10_signal_diagnostics(observation.signal),
    }
