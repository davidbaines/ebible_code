"""
Smoke tests for checking the extract files using pytest.
These are intended to be run when the extract files are updated.
The `CORPUS_PATH` environment variable must be set correctly.
See https://github.com/BibleNLP/ebible/issues/53 for more context.

Tests are intended to be run using pytest, e.g.
  pytest test_smoke.py
    or
  poetry run pytest test_smoke.py
    or, to run all tests
  poetry run pytest

Note these tests are quite CPU intensive so some thought is needed if we decide
to integrate them into the build pipeline.
"""
import os
import warnings
import pytest
import regex
from pathlib import Path
from typing import List
from dotenv import load_dotenv


# --- Fixtures ---

@pytest.fixture(scope="module")
def corpus_path() -> Path:
    """Provides the path to the corpus directory."""
    load_dotenv()
    corpus_path_lookup = os.getenv("CORPUS_PATH")
    if corpus_path_lookup:
        path = Path(corpus_path_lookup)

    if not path.is_dir():
         pytest.skip(f"Corpus directory not found at {path}, skipping smoke tests.")
    return path

# --- Helper Function ---

def _report_possible_failures(failures: List[str], condition: str) -> None:
    """Helper function to fail a test if failures are found."""
    if failures:
        fail_message = (
            f"{len(failures)} found not matching condition: {condition}.\n"
            f"Examples: {', '.join(failures[:20])}{'...' if len(failures) > 20 else ''}"
        )
        pytest.fail(fail_message, pytrace=False)

# --- Test Functions ---

def test_number_of_files(corpus_path: Path) -> None:
    """Tests if there are a sufficient number of corpus files."""
    all_extract_files = list(corpus_path.glob("*.txt"))
    assert len(all_extract_files) > 1000, "There should be at least 1000 corpus files"

def test_all_files_have_at_least_400_verses(corpus_path: Path) -> None:
    """Tests if all corpus files have a minimum number of non-empty lines (verses)."""
    def verse_count(path: Path) -> int:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                return sum(1 for line in fp if line.strip())
        except Exception as e:
            warnings.warn(f"Could not read or count lines in {path.name}: {e}")
            return 0

    paths_with_less_than_400_verses = [
        p.name for p in corpus_path.iterdir()
        if p.is_file() and p.suffix == '.txt' and verse_count(p) < 400
    ]
    _report_possible_failures(
        paths_with_less_than_400_verses, "file has at least 400 non-empty verses"
    )

def test_filename_format_correct(corpus_path: Path) -> None:
    """Tests if filenames adhere to the expected format: <translationId>.txt
    where translationId starts with a letter and contains only alphanumeric
    characters, hyphens, or underscores."""
    def is_valid(filename: str) -> bool:
        if not filename.endswith(".txt"):
            return False
        stem = filename[:-4]
        return bool(regex.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", stem))

    invalid_filenames = [
        p.name for p in corpus_path.iterdir() if p.is_file() and not is_valid(p.name)
    ]
    _report_possible_failures(
        invalid_filenames,
        "filename adheres to format `<translationId>.txt`",
    )

def test_certain_bibles_exist_in_corpus(corpus_path: Path):
    """Tests if specific, expected Bible extract files exist."""
    filenames = [
        "eng-kjv.txt",
    ]
    missing_bibles = [
        filename
        for filename in filenames
        if not (corpus_path / filename).exists()
    ]
    _report_possible_failures(
        missing_bibles, "extract file exists for the Bible"
    )

def test_certain_bibles_have_complete_OT_NT(corpus_path: Path):
    """Tests if specific Bibles have a high percentage of expected OT/NT verses."""
    filenames = [
        "engylt.txt",
        "engULB.txt",
        "fraLSG.txt",
    ]

    # Note that the extract file is structured:
    # - ot
    # - nt
    # - deuterocanonical
    # So we're just checking the front of the extract file has verses
    num_ot_nt_verses = 31170

    def is_99_percent_complete(path: Path) -> bool:
        """Returns True if the extract has >= 99% of expected OT and NT verses."""
        if not path.exists():
            warnings.warn(f"File not found for completeness check: {path.name}")
            return False
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                lines = fp.readlines()
                if len(lines) < num_ot_nt_verses:
                    return False
                non_empty_verse_count = sum(
                    1 for i, line in enumerate(lines)
                    if i < num_ot_nt_verses and line.strip()
                )
                return (non_empty_verse_count / num_ot_nt_verses) >= 0.99
        except Exception as e:
            warnings.warn(f"Could not read or check completeness of {path.name}: {e}")
            return False

    incomplete_bibles = [
        filename
        for filename in filenames
        if not is_99_percent_complete(corpus_path / filename)
    ]
    _report_possible_failures(
        incomplete_bibles,
        "has at least 99% of the verses corresponding to the OT and NT",
    )

def test_all_files_have_correct_number_of_lines(corpus_path: Path):
    """Tests if all corpus files have the exact expected number of lines."""
    expected_num_lines = 41899

    def has_correct_number_lines(path: Path) -> bool:
        try:
            with open(path, "rb") as fp:
                num_lines = sum(1 for _ in fp)
            return num_lines == expected_num_lines
        except Exception as e:
            warnings.warn(f"Could not read or count lines in {path.name}: {e}")
            return False

    bibles_with_incorrect_num_lines = [
        p.name for p in corpus_path.iterdir()
        if p.is_file() and p.suffix == '.txt' and not has_correct_number_lines(p)
    ]
    _report_possible_failures(
        bibles_with_incorrect_num_lines,
        f"has expected number of lines ({expected_num_lines})",
    )
