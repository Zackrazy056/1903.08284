from __future__ import annotations

from pathlib import Path

from ringdown.experiments import (
    PAPER_FIG10_DYNESTY_SHORT,
    PAPER_FIG10_N0_ANCHOR,
    ExperimentPaths,
    paper_fig10_registry_markdown,
)
from ringdown.experiments.platform import merge_missing_cli_args


def test_experiment_paths_place_outputs_under_tier_and_id(tmp_path: Path) -> None:
    paths = ExperimentPaths.for_repo(
        "paper_fig10_dynesty_short",
        "production",
        repo_root=tmp_path,
    ).ensure()
    assert paths.results_dir == tmp_path / "results" / "production" / "paper_fig10_dynesty_short"
    assert paths.results_dir.exists()


def test_registered_paper_fig10_default_outputs_are_scoped_to_experiment_dir(tmp_path: Path) -> None:
    defaults = PAPER_FIG10_DYNESTY_SHORT.default_cli(repo_root=tmp_path)
    for _, value in defaults:
        assert str(tmp_path / "results" / "production" / "paper_fig10_dynesty_short") in value


def test_merge_missing_cli_args_preserves_user_overrides() -> None:
    merged = merge_missing_cli_args(
        ["--output", "custom.png"],
        [("--output", "default.png"), ("--diag-csv", "default.csv")],
    )
    assert merged == ["--output", "custom.png", "--diag-csv", "default.csv"]


def test_paper_fig10_registry_markdown_lists_registered_ids() -> None:
    md = paper_fig10_registry_markdown()
    assert "paper_fig10_dynesty_short" in md
    assert "paper_fig10_n0_anchor" in md
    assert "paper_fig10_n3_logrel_relphase_candidate" in md


def test_registered_specs_record_constant_offset_and_heuristics_policy() -> None:
    row = PAPER_FIG10_DYNESTY_SHORT.registry_record()
    assert row["include_constant_offset"] is False
    assert row["physics_heuristics_enabled"] is False

    anchor = PAPER_FIG10_N0_ANCHOR.registry_record()
    assert anchor["include_constant_offset"] is False
    assert anchor["physics_heuristics_enabled"] is False
