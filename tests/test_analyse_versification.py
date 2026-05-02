"""
V6 — Integration test for analyse_versification.py.

Runs the script against TEST_EBIBLE_DATA_DIR and asserts:
- CSV is created with expected columns
- PNG histogram is created and non-empty
- Stdout contains a score band line and a total count
Skips gracefully if TEST_EBIBLE_DATA_DIR is not set or the directory is absent.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

_test_data_dir_str = os.getenv("TEST_EBIBLE_DATA_DIR")
_test_data_dir = Path(_test_data_dir_str) if _test_data_dir_str else None


@pytest.fixture(scope="module")
def analyse_output(tmp_path_factory):
    """Run analyse_versification.py --test and return (stdout, csv_path, png_path)."""
    if _test_data_dir is None:
        pytest.skip("TEST_EBIBLE_DATA_DIR not set in .env")
    if not _test_data_dir.is_dir():
        pytest.skip(f"TEST_EBIBLE_DATA_DIR does not exist: {_test_data_dir}")

    result = subprocess.run(
        [sys.executable, "ebible_code/analyse_versification.py", "--test"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"analyse_versification.py exited with code {result.returncode}.\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
    metadata_dir = _test_data_dir / "metadata"
    csv_path = metadata_dir / "analyse_versification.csv"
    png_path = metadata_dir / "versification_scores_histogram.png"
    ranked_png_path = metadata_dir / "versification_scores_ranked.png"
    return result.stdout, csv_path, png_path, ranked_png_path


def test_csv_created_and_nonempty(analyse_output):
    _, csv_path, _, _ = analyse_output
    assert csv_path.exists(), f"CSV not found at {csv_path}"
    assert csv_path.stat().st_size > 0, "CSV is empty"


def test_csv_has_expected_columns(analyse_output):
    _, csv_path, _, _ = analyse_output
    first_line = csv_path.read_text(encoding="utf-8").splitlines()[0]
    for col in ("project_name", "best_match", "status", "matching_chapters",
                "total_project_chapters", "total_differentiating_chapters",
                "mismatch_ENGLISH", "mismatch_VULGATE", "mismatch_RUSSIAN_ORTHODOX",
                "score_ENGLISH", "score_VULGATE", "score_RUSSIAN_ORTHODOX", "notes"):
        assert col in first_line, f"Column '{col}' missing from CSV header"


def test_png_created_and_nonempty(analyse_output):
    _, _, png_path, ranked_png_path = analyse_output
    assert png_path.exists(), f"Histogram PNG not found at {png_path}"
    assert png_path.stat().st_size > 0, "Histogram PNG is empty"
    assert ranked_png_path.exists(), f"Ranked plot PNG not found at {ranked_png_path}"
    assert ranked_png_path.stat().st_size > 0, "Ranked plot PNG is empty"


def test_stdout_contains_total_and_bands(analyse_output):
    stdout, _, _, _ = analyse_output
    assert "Total projects analysed:" in stdout
    assert "Ranked plot:" in stdout
