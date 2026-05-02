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
    _VT_PREFERENCE_ORDER,
    compute_versification_scores,
    get_verse_data_from_vrs_obj,
)


@dataclass
class VersificationMatchReport:
    project_name: str
    best_match: VersificationType
    matching_chapters: int          # total_project_chapters − mismatch_counts[best_match]
    scores: dict                    # Dict[VersificationType, float]  percentage 0.0–100.0
    mismatch_counts: dict           # Dict[VersificationType, int]  all-chapter mismatches
    total_differentiating_chapters: int
    total_project_chapters: int
    status: str                     # "matched" | "tied" | "unknown"
    notes: str


def _threshold() -> float:
    try:
        return float(os.environ.get("VERSIFICATION_MATCH_THRESHOLD", "0.0"))
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
            matching_chapters=0,
            scores=_empty_scores,
            mismatch_counts=_empty_mismatches,
            total_differentiating_chapters=0,
            total_project_chapters=0,
            status="tied",
            notes="No .vrs file found. Defaulting to English.",
        )

    try:
        project_vrs_obj = Versification.load(project_vrs_path, fallback_name=project_name)
    except Exception as exc:
        return VersificationMatchReport(
            project_name=project_name,
            best_match=VersificationType.ENGLISH,
            matching_chapters=0,
            scores=_empty_scores,
            mismatch_counts=_empty_mismatches,
            total_differentiating_chapters=0,
            total_project_chapters=0,
            status="tied",
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

    best_score = max(scores.values()) if scores else 0.0
    best_types = {vt for vt, s in scores.items() if s == best_score}

    # Apply preference order to break ties.
    best_match = VersificationType.ENGLISH
    for vt in _VT_PREFERENCE_ORDER:
        if vt in best_types:
            best_match = vt
            break

    matching_chapters = total_project_chapters - mismatch_counts.get(best_match, 0)

    if best_score < threshold:
        per_std = ", ".join(
            f"{vt.name}={mismatch_counts[vt]}"
            for vt in sorted(mismatch_counts, key=lambda v: v.value)
        )
        return VersificationMatchReport(
            project_name=project_name,
            best_match=VersificationType.ENGLISH,
            matching_chapters=total_project_chapters - mismatch_counts.get(VersificationType.ENGLISH, 0),
            scores=scores,
            mismatch_counts=mismatch_counts,
            total_differentiating_chapters=total_differentiating_chapters,
            total_project_chapters=total_project_chapters,
            status="unknown",
            notes=(
                f"Best score {best_score:.1f}% is below threshold {threshold:.1f}%. "
                "No versification matched well enough. "
                "Settings.xml will use English (4) as a fallback. "
                f"Mismatch counts per standard: {per_std}"
            ),
        )

    is_tied = len(best_types) > 1

    if not is_tied:
        return VersificationMatchReport(
            project_name=project_name,
            best_match=best_match,
            matching_chapters=matching_chapters,
            scores=scores,
            mismatch_counts=mismatch_counts,
            total_differentiating_chapters=total_differentiating_chapters,
            total_project_chapters=total_project_chapters,
            status="matched",
            notes=f"Best match: {best_match.name} ({best_score:.1f}% of project chapters match)",
        )

    # Tied case.
    tied_names = ", ".join(vt.name for vt in _VT_PREFERENCE_ORDER if vt in best_types)
    if total_differentiating_chapters == 0:
        notes = (
            f"All {total_project_chapters} project chapters are invariant; "
            f"all versifications match equally ({best_score:.1f}%). "
            "ENGLISH chosen by preference."
        )
    else:
        notes = f"Tied at {best_score:.1f}%: {tied_names}. {best_match.name} chosen as most common."

    return VersificationMatchReport(
        project_name=project_name,
        best_match=best_match,
        matching_chapters=matching_chapters,
        scores=scores,
        mismatch_counts=mismatch_counts,
        total_differentiating_chapters=total_differentiating_chapters,
        total_project_chapters=total_project_chapters,
        status="tied",
        notes=notes,
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
        ["project_name", "best_match", "status", "matching_chapters",
         "total_project_chapters", "total_differentiating_chapters"]
        + [f"mismatch_{vt.name}" for vt in std_types]
        + [f"score_{vt.name}" for vt in std_types]
        + ["notes"]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            row: dict = {
                "project_name": r.project_name,
                "best_match": r.best_match.name,
                "status": r.status,
                "matching_chapters": r.matching_chapters,
                "total_project_chapters": r.total_project_chapters,
                "total_differentiating_chapters": r.total_differentiating_chapters,
                "notes": r.notes,
            }
            for vt in std_types:
                row[f"mismatch_{vt.name}"] = r.mismatch_counts.get(vt, 0)
                row[f"score_{vt.name}"] = round(r.scores.get(vt, 0.0), 1)
            writer.writerow(row)

    # ---- Histogram ----
    # best score for each project = score of the winning versification (percentage 0–100)
    best_scores = [max(r.scores.values()) if r.scores else 0.0 for r in reports]
    png_path = metadata_dir / "versification_scores_histogram.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [i * 10 for i in range(11)]  # 0, 10, 20, ..., 100
    ax.hist(best_scores, bins=bins, edgecolor="black", color="steelblue", alpha=0.8)
    if threshold > 0.0:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"Threshold = {threshold:.1f}%")
        ax.legend()
    ax.set_xlabel("Best versification match score (%)")
    ax.set_ylabel("Number of projects")
    ax.set_title("Versification best-match score distribution")
    fig.savefig(png_path, dpi=100)
    plt.close(fig)

    # ---- Ranked score plot (one point per project) ----
    sorted_scores = sorted(best_scores, reverse=True)
    ranked_png_path = metadata_dir / "versification_scores_ranked.png"
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ranks = range(1, len(sorted_scores) + 1)
    ax2.scatter(ranks, sorted_scores, s=4, alpha=0.5, color="steelblue")
    if threshold > 0.0:
        ax2.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
                    label=f"Threshold = {threshold:.1f}%")
        ax2.legend()
    ax2.set_xlabel("Project rank (highest score first)")
    ax2.set_ylabel("Best versification match score (%)")
    ax2.set_title("Versification match scores — one point per project (sorted)")
    ax2.set_xlim(0, len(sorted_scores) + 1)
    ax2.set_ylim(-2, 102)
    fig2.savefig(ranked_png_path, dpi=100)
    plt.close(fig2)

    # ---- Stdout summary ----
    total = len(reports)
    tied_count = sum(1 for r in reports if r.status == "tied")
    below_threshold = sum(1 for r in reports if r.status == "unknown")

    print("\nVersification analysis complete.")
    print(f"Total projects analysed: {total}")
    print("\nScore distribution (best score bands, %):")
    for i in range(10):
        lo = i * 10
        hi = (i + 1) * 10
        count = sum(
            1 for bs in best_scores
            if lo <= bs < hi or (i == 9 and bs == 100.0)
        )
        print(f"  {lo:3d}-{hi:3d}: {count}")
    print(f"\nTied projects (decided by preference order): {tied_count}")
    print(f"Projects currently below threshold ({threshold:.1f}%): {below_threshold}")
    print(f"\nCSV:          {csv_path}")
    print(f"Histogram:    {png_path}")
    print(f"Ranked plot:  {ranked_png_path}")


if __name__ == "__main__":
    main()
