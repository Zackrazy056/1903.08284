#!/usr/bin/env python3
"""Orchestrate the remaining reproduction steps for arXiv:1903.08284.

This script keeps commands explicit and modular so each stage can be run
independently or as one full pipeline.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: str


def default_stages(outdir: Path) -> list[Stage]:
    """Return the default pipeline stages.

    NOTE: The commands are intentionally conservative placeholders that are
    easy to replace with project-specific implementations.
    """
    data_dir = outdir / "data"
    posterior_dir = outdir / "posterior"
    fig_dir = outdir / "figures"

    return [
        Stage(
            "download",
            "Download GW150914 strain + posterior samples from public sources",
            (
                "python -m pip install --quiet gwosc h5py numpy "
                "&& python - <<'EOF'\n"
                "from pathlib import Path\n"
                "from gwosc.datasets import event_gps\n"
                "print('GW150914 GPS:', event_gps('GW150914'))\n"
                f"Path('{data_dir}').mkdir(parents=True, exist_ok=True)\n"
                "print('Data directory prepared')\n"
                "EOF"
            ),
        ),
        Stage(
            "preprocess",
            "Convert downloaded files into ringdown-ready arrays",
            (
                "python - <<'EOF'\n"
                "from pathlib import Path\n"
                f"out = Path('{data_dir}') / 'preprocessed'\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'README.txt').write_text('Replace with preprocessing pipeline.\\n')\n"
                "print('Preprocess scaffold generated:', out)\n"
                "EOF"
            ),
        ),
        Stage(
            "fit",
            "Run overtone ringdown fit for N=0..7 and compare mismatch",
            (
                "python - <<'EOF'\n"
                "from pathlib import Path\n"
                f"out = Path('{posterior_dir}')\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'fit_results.csv').write_text('mode,mismatch\\nN0,TODO\\n')\n"
                "print('Fit scaffold generated:', out / 'fit_results.csv')\n"
                "EOF"
            ),
        ),
        Stage(
            "validate",
            "Regenerate key paper figure and compare with target trend",
            (
                "python - <<'EOF'\n"
                "from pathlib import Path\n"
                f"fig = Path('{fig_dir}')\n"
                "fig.mkdir(parents=True, exist_ok=True)\n"
                "(fig / 'validation_report.md').write_text('\\n'.join([\n"
                "  '# Validation Report',\n"
                "  '- Recreate Fig. 3/4 trend (pending).',\n"
                "  '- Compare peak-time consistency (pending).',\n"
                "]))\n"
                "print('Validation scaffold generated:', fig / 'validation_report.md')\n"
                "EOF"
            ),
        ),
    ]


def run_command(command: str, dry_run: bool) -> int:
    print(f"\\n$ {command}")
    if dry_run:
        return 0
    return subprocess.run(command, shell=True, check=False).returncode


def run_pipeline(stages: list[Stage], target: str, dry_run: bool) -> int:
    names = [stage.name for stage in stages]
    if target != "all" and target not in names:
        raise ValueError(f"Unknown target '{target}', choose from {names + ['all']}")

    selected = stages if target == "all" else [s for s in stages if s.name == target]

    for stage in selected:
        print(f"\\n==> [{stage.name}] {stage.description}")
        rc = run_command(stage.command, dry_run=dry_run)
        if rc != 0:
            print(f"Stage '{stage.name}' failed with exit code {rc}")
            return rc

    print("\\nPipeline completed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the staged reproduction workflow for arXiv:1903.08284"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="Which stage to run: download, preprocess, fit, validate, all",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts",
        help="Output directory to store intermediate and final artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands without executing them",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print shell-safe command snippets and exit",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stages = default_stages(Path(args.outdir))

    if args.print_commands:
        for stage in stages:
            print(f"[{stage.name}] {stage.description}")
            print(shlex.quote(stage.command))
        return 0

    return run_pipeline(stages, target=args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
