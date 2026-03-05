from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


TRUE_MF = 68.546372
TRUE_CHIF = 0.692085


@dataclass(frozen=True)
class Decision:
    path: Path
    kind: str  # main_png, trace_png, csv, npz
    action: str  # KEEP, DELETE
    reason: str
    family: str
    stem: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean fig10 assets in results/ using best+final retention policy."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="CSV report path (default: results/fig10_cleanup_report_YYYYMMDD.csv)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete files.")
    return parser.parse_args()


def is_trace_png(path: Path) -> bool:
    name = path.name
    if name.endswith("_traces.png"):
        return True
    return re.search(r"_trace_all_N\d+\.png$", name) is not None


def group_key(name: str) -> str:
    key = name
    key = re.sub(r"_revision\d+(_nob|_plus)?\.png$", "_revisionX.png", key)
    key = re.sub(r"_samplingfix\d+\.png$", "_samplingfixX.png", key)
    key = re.sub(r"_psd_revision\d+\.png$", "_psd_revisionX.png", key)
    key = re.sub(r"_psd_n3_long\d*\.png$", "_psd_n3_longX.png", key)
    key = re.sub(r"_minimal_.*\.png$", "_minimal_variant.png", key)
    key = re.sub(r"_n3_(mid|long)chain\.png$", "_n3_chainX.png", key)
    key = re.sub(r"_reproduction(_no_offset)?\.png$", "_reproductionX.png", key)
    return key


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def choose_diag_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    if "N" in rows[0]:
        n3_rows = [r for r in rows if str(r.get("N", "")).strip() == "3"]
        if n3_rows:
            return n3_rows[0]
    return rows[-1]


def parse_float_field(row: dict[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        val = row.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except ValueError:
            continue
    return None


def score_main_png(main_png: Path, csv_index: dict[str, Path]) -> float | None:
    stem = main_png.stem
    diag = csv_index.get(f"{stem}_diagnostics.csv") or csv_index.get(f"{stem}_diag.csv")
    if diag is None:
        return None
    rows = read_csv_rows(diag)
    row = choose_diag_row(rows)
    if row is None:
        return None
    mf_q50 = parse_float_field(row, ("mf_q50", "mf_q50_msun"))
    chif_q50 = parse_float_field(row, ("chif_q50", "chi_q50", "ch_q50"))
    if mf_q50 is None or chif_q50 is None:
        return None
    return abs(mf_q50 - TRUE_MF) / 50.0 + abs(chif_q50 - TRUE_CHIF)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if not results_dir.exists() or not results_dir.is_dir():
        raise ValueError(f"invalid results dir: {results_dir}")

    if args.report is None:
        today = datetime.now().strftime("%Y%m%d")
        report_path = results_dir / f"fig10_cleanup_report_{today}.csv"
    else:
        report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Scope: only fig10* assets in results.
    all_png = sorted(results_dir.glob("fig10*.png"))
    all_csv = sorted(results_dir.glob("fig10*.csv"))
    all_npz = sorted(results_dir.glob("fig10*.npz"))

    # Exclude cleanup reports from cleanup targets.
    all_csv = [
        p
        for p in all_csv
        if p.resolve() != report_path.resolve() and not re.match(r"fig10_cleanup_report_.*\.csv$", p.name)
    ]

    main_pngs = [p for p in all_png if not is_trace_png(p)]
    trace_pngs = [p for p in all_png if is_trace_png(p)]
    csv_index = {p.name: p for p in all_csv}

    families: dict[str, list[Path]] = {}
    for path in main_pngs:
        families.setdefault(group_key(path.name), []).append(path)

    keep_main: set[Path] = set()
    family_final: dict[str, Path] = {}
    family_best: dict[str, Path] = {}
    for fam, items in families.items():
        ordered = sorted(items, key=lambda p: (p.stat().st_mtime, p.name))
        final = ordered[-1]
        family_final[fam] = final

        scored: list[tuple[Path, float]] = []
        for item in ordered:
            score = score_main_png(item, csv_index)
            if score is not None:
                scored.append((item, score))
        best = min(scored, key=lambda x: x[1])[0] if scored else final
        family_best[fam] = best

        keep_main.add(final)
        keep_main.add(best)

    keep_trace: set[Path] = set()
    for fam, final in family_final.items():
        stem = final.stem
        traces = results_dir / f"{stem}_traces.png"
        if traces.exists():
            keep_trace.add(traces)
        for p in results_dir.glob(f"{stem}_trace_all_N*.png"):
            keep_trace.add(p)

    keep_csv: set[Path] = set()
    for main_png in keep_main:
        stem = main_png.stem
        diag = csv_index.get(f"{stem}_diagnostics.csv") or csv_index.get(f"{stem}_diag.csv")
        if diag is not None:
            keep_csv.add(diag)

    keep_npz: set[Path] = set()
    for main_png in keep_main:
        stem = main_png.stem
        for p in results_dir.glob(f"{stem}_samples_N*.npz"):
            keep_npz.add(p)

    decisions: list[Decision] = []
    for p in main_pngs:
        fam = group_key(p.name)
        action = "KEEP" if p in keep_main else "DELETE"
        if action == "KEEP":
            if p == family_best[fam] and p == family_final[fam]:
                reason = "best_and_final_main"
            elif p == family_best[fam]:
                reason = "best_main"
            else:
                reason = "final_main"
        else:
            reason = "main_not_best_or_final"
        decisions.append(
            Decision(
                path=p,
                kind="main_png",
                action=action,
                reason=reason,
                family=fam,
                stem=p.stem,
                size_bytes=p.stat().st_size,
            )
        )

    for p in trace_pngs:
        fam = group_key(p.name)
        action = "KEEP" if p in keep_trace else "DELETE"
        reason = "final_trace" if action == "KEEP" else "trace_not_for_final"
        decisions.append(
            Decision(
                path=p,
                kind="trace_png",
                action=action,
                reason=reason,
                family=fam,
                stem=p.stem,
                size_bytes=p.stat().st_size,
            )
        )

    for p in all_csv:
        action = "KEEP" if p in keep_csv else "DELETE"
        reason = "diag_for_kept_main" if action == "KEEP" else "csv_not_linked_to_kept_main"
        decisions.append(
            Decision(
                path=p,
                kind="csv",
                action=action,
                reason=reason,
                family=group_key(p.name),
                stem=p.stem,
                size_bytes=p.stat().st_size,
            )
        )

    for p in all_npz:
        action = "KEEP" if p in keep_npz else "DELETE"
        reason = "npz_for_kept_main" if action == "KEEP" else "npz_not_linked_to_kept_main"
        decisions.append(
            Decision(
                path=p,
                kind="npz",
                action=action,
                reason=reason,
                family=group_key(p.name),
                stem=p.stem,
                size_bytes=p.stat().st_size,
            )
        )

    delete_paths = [d.path for d in decisions if d.action == "DELETE"]
    bytes_to_delete = sum(d.size_bytes for d in decisions if d.action == "DELETE")

    deleted_count = 0
    deleted_bytes = 0
    if args.apply:
        for path in delete_paths:
            if path.exists():
                sz = path.stat().st_size
                path.unlink()
                deleted_count += 1
                deleted_bytes += sz

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "action",
                "kind",
                "reason",
                "family",
                "path",
                "size_bytes",
                "apply_mode",
            ]
        )
        for d in sorted(decisions, key=lambda x: (x.kind, x.action, x.path.name)):
            writer.writerow(
                [
                    d.action,
                    d.kind,
                    d.reason,
                    d.family,
                    str(d.path),
                    d.size_bytes,
                    int(args.apply),
                ]
            )

    print(f"mode={'apply' if args.apply else 'dry-run'}")
    print(f"report={report_path}")
    print(f"families={len(families)}")
    print("kept_main_images:")
    for fam in sorted(family_final.keys()):
        b = family_best[fam].name
        fimg = family_final[fam].name
        print(f"  {fam}: best={b}, final={fimg}")

    keep_count = sum(1 for d in decisions if d.action == "KEEP")
    del_count = sum(1 for d in decisions if d.action == "DELETE")
    print(f"keep_count={keep_count}")
    print(f"delete_count={del_count}")
    print(f"delete_bytes={bytes_to_delete}")
    if args.apply:
        print(f"deleted_count={deleted_count}")
        print(f"deleted_bytes={deleted_bytes}")


if __name__ == "__main__":
    main()
