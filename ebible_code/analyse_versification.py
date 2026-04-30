"""
analyse_versification.py — Standalone script for versification analysis.

Scans all project folders in EBIBLE_DATA_DIR (or TEST_EBIBLE_DATA_DIR with --test),
calls describe_versification_match() for each project that has a .vrs file, and outputs:
  1. {metadata}/analyse_versification.csv   — per-project scores and match details
  2. {metadata}/versification_scores_histogram.png — score distribution with threshold line
  3. Stdout summary

Run:
  poetry run python ebible_code/analyse_versification.py
  poetry run python ebible_code/analyse_versification.py --test
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from machine.scripture import Versification, VersificationType

# Allow running from repo root or from ebible_code/
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from settings_file import (
    compute_versification_scores,
    get_verse_data_from_vrs_obj,
)


@dataclass
class VersificationMatchReport:
    project_name: str
    best_match: VersificationType
    best_score: float
    scores: dict            # Dict[VersificationType, float]
    mismatch_counts: dict   # Dict[VersificationType, int]
    total_differentiating_chapters: int
    total_project_chapters: int
    status: str             # "matched" | "indistinguishable" | "unknown"
    notes: str


def _threshold() -> float:
    try:
        return float(os.environ.get("VERSIFICATION_UNKNOWN_THRESHOLD", "0.0"))
    except ValueError:
        return 0.0


def describe_versification_match(project_path: Path) -> VersificationMatchReport:
    """Full scoring and explainability report for one project, consistent with estimate_versification()."""
    project_name = project_path.name
    project_vrs_path = project_path / f"{project_name}.vrs"

    _empty_scores = {vt: 0.0 for vt in VersificationType if vt != VersificationType.UNKNOWN}
    _empty_mismatches = {vt: 0 for vt in VersificationType if vt != VersificationType.UNKNOWN}

    if not project_vrs_path.is_file():
        return VersificationMatchReport(
            project_name=project_name,
            best_match=VersificationType.ENGLISH,
            best_score=0.0,
            scores=_empty_scores,
            mismatch_counts=_empty_mismatches,
            total_differentiating_chapters=0,
            total_project_chapters=0,
            status="indistinguishable",
            notes="No .vrs file found. Defaulting to English.",
        )

    try:
        project_vrs_obj = Versification.load(project_vrs_path, fallback_name=project_name)
    except Exception as exc:
        return VersificationMatchReport(
            project_name=project_name,
            best_match=VersificationType.ENGLISH,
            best_score=0.0,
            scores=_empty_scores,
            mismatch_counts=_empty_mismatches,
            total_differentiating_chapters=0,
            total_project_chapters=0,
            status="indistinguishable",
            notes=f"Could not load .vrs file: {exc}. Defaulting to English.",
        )

    project_verse_data = get_verse_data_from_vrs_obj(project_vrs_obj)
    result = compute_versification_scores(project_verse_data)

    diff_chapters = result["project_differentiating_chapters"]
    scores: dict = result["scores"]
    mismatch_counts: dict = result["mismatch_counts"]
    total_project_chapters: int = result["total_project_chapters"]
    total_differentiating_chapters = len(diff_chapters)
    threshold = _threshold()

    if total_differentiating_chapters == 0:
        return VersificationMatchReport(
            project_name=project_name,
            best_match=VersificationType.ENGLISH,
            best_score=0.0,
            scores=scores,
            mismatch_counts=mismatch_counts,
            total_differentiating_chapters=0,
            total_project_chapters=total_project_chapters,
            status="indistinguishable",
            notes=(
                f"All {total_project_chapters} project chapters are invariant across versifications. "
                "Cannot distinguish; defaulting to English. "
                "'Indistinguishable' means the Bible only contains chapters and verses whose "
                "verse counts are identical in all versification systems."
            ),
        )

    best_score = max(scores.values()) if scores else 0.0
    best_types = [vt for vt, s in scores.items() if s == best_score]
    best_match = (
        VersificationType.ENGLISH if VersificationType.ENGLISH in best_types else best_types[0]
    )

    if best_score < threshold:
        per_std = ", ".join(
            f"{vt.name}={mismatch_counts[vt]}"
            for vt in sorted(mismatch_counts, key=lambda v: v.value)
        )
        return VersificationMatchReport(
            project_name=project_name,
            best_match=VersificationType.UNKNOWN,
            best_score=best_score,
            scores=scores,
            mismatch_counts=mismatch_counts,
            total_differentiating_chapters=total_differentiating_chapters,
            total_project_chapters=total_project_chapters,
            status="unknown",
            notes=(
                f"Best score {best_score:.1%} is below threshold {threshold:.1%}. "
                "No versification matched well. "
                "Settings.xml will use English (4) as a fallback. "
                f"Mismatch counts per standard: {per_std}"
            ),
        )

    return VersificationMatchReport(
        project_name=project_name,
        best_match=best_match,
        best_score=best_score,
        scores=scores,
        mismatch_counts=mismatch_counts,
        total_differentiating_chapters=total_differentiating_chapters,
        total_project_chapters=total_project_chapters,
        status="matched",
        notes=f"Best match: {best_match.name} ({best_score:.1%} of differentiating chapters match)",
    )


def _scan_projects(data_dir: Path) -> list:
    """Return project directories that have a matching .vrs file."""
    projects = []
    for folder_name in ("projects", "private_projects"):
        base = data_dir / folder_name
        if not base.is_dir():
            continue
        for proj_dir in sorted(base.iterdir()):
            if proj_dir.is_dir() and (proj_dir / f"{proj_dir.name}.vrs").is_file():
                projects.append(proj_dir)
    return projects


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyse versification match scores for all eBible projects.")
    parser.add_argument("--test", action="store_true", help="Use TEST_EBIBLE_DATA_DIR instead of EBIBLE_DATA_DIR")
    args = parser.parse_args()

    env_key = "TEST_EBIBLE_DATA_DIR" if args.test else "EBIBLE_DATA_DIR"
    data_dir_str = os.environ.get(env_key)
    if not data_dir_str:
        print(f"ERROR: {env_key} not set in .env", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(data_dir_str)
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    threshold = _threshold()

    project_dirs = _scan_projects(data_dir)
    if not project_dirs:
        print("No projects with .vrs files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing {len(project_dirs)} projects ...", flush=True)
    reports: list[VersificationMatchReport] = []
    for proj_dir in project_dirs:
        reports.append(describe_versification_match(proj_dir))

    # ---- CSV output ----
    std_types = [vt for vt in VersificationType if vt != VersificationType.UNKNOWN]
    csv_path = metadata_dir / "analyse_versification.csv"
    fieldnames = (
        ["project_name", "status", "best_match", "best_score",
         "total_differentiating_chapters", "total_project_chapters"]
        + [f"score_{vt.name}" for vt in std_types]
        + [f"mismatches_{vt.name}" for vt in std_types]
        + ["notes"]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            row: dict = {
                "project_name": r.project_name,
                "status": r.status,
                "best_match": r.best_match.name,
                "best_score": r.best_score,
                "total_differentiating_chapters": r.total_differentiating_chapters,
                "total_project_chapters": r.total_project_chapters,
                "notes": r.notes,
            }
            for vt in std_types:
                row[f"score_{vt.name}"] = r.scores.get(vt, 0.0)
                row[f"mismatches_{vt.name}"] = r.mismatch_counts.get(vt, 0)
            writer.writerow(row)

    # ---- Histogram ----
    best_scores = [r.best_score for r in reports]
    png_path = metadata_dir / "versification_scores_histogram.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [i / 10 for i in range(11)]
    ax.hist(best_scores, bins=bins, edgecolor="black", color="steelblue", alpha=0.8)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Best versification match score")
    ax.set_ylabel("Number of projects")
    ax.set_title("Versification best-match score distribution")
    ax.legend()
    fig.savefig(png_path, dpi=100)
    plt.close(fig)

    # ---- Stdout summary ----
    total = len(reports)
    indistinguishable = sum(1 for r in reports if r.status == "indistinguishable")
    below_threshold = sum(1 for r in reports if r.status == "unknown")

    print("\nVersification analysis complete.")
    print(f"Total projects analysed: {total}")
    print("\nScore distribution (best_score bands):")
    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        count = sum(
            1 for r in reports
            if lo <= r.best_score < hi or (i == 9 and r.best_score == 1.0)
        )
        print(f"  {lo:.1f}–{hi:.1f}: {count}")
    print(f"\nIndistinguishable projects (all chapters invariant): {indistinguishable}")
    print(f"Projects currently below threshold ({threshold:.2f}): {below_threshold}")
    print(f"\nCSV:       {csv_path}")
    print(f"Histogram: {png_path}")


if __name__ == "__main__":
    main()
