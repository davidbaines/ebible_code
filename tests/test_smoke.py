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
import pytest
import regex
from pathlib import Path
from typing import List
from dotenv import load_dotenv # Import the function


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
        # Consider limiting the number of failures printed for long lists
        fail_message = (
            f"{len(failures)} found not matching condition: {condition}.\n"
            f"Examples: {', '.join(failures[:20])}{'...' if len(failures) > 20 else ''}"
        )
        pytest.fail(fail_message, pytrace=False) # pytrace=False for cleaner output

# --- Test Functions ---

def test_number_of_files(corpus_path: Path) -> None:
    """Tests if there are a sufficient number of corpus files."""
    all_extract_files = list(corpus_path.glob("*.txt")) # More specific glob
    assert len(all_extract_files) > 1000, "There should be at least 1000 corpus files"

def test_all_files_have_at_least_400_verses(corpus_path: Path) -> None:
    """Tests if all corpus files have a minimum number of non-empty lines (verses)."""
    def verse_count(path: Path) -> int:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                # Count non-empty lines more directly
                return sum(1 for line in fp if line.strip())
        except Exception as e:
            pytest.warn(f"Could not read or count lines in {path.name}: {e}")
            return 0 # Treat unreadable files as having 0 verses for this check

    paths_with_less_than_400_verses = [
        p.name for p in corpus_path.iterdir()
        if p.is_file() and p.suffix == '.txt' and verse_count(p) < 400
    ]
    _report_possible_failures(
        paths_with_less_than_400_verses, "file has at least 400 non-empty verses"
    )

def test_filename_format_correct(corpus_path: Path) -> None:
    """Tests if filenames adhere to the expected format."""
    def is_valid(filename: str) -> bool:
        if not filename.endswith(".txt"):
            return False
        parts = filename[:-4].split("-") # Remove .txt before splitting
        if len(parts) < 2:
            return False
        language_code = parts[0]
        # Anchor the regex for safety
        return bool(regex.match("^[a-z]{2,3}$", language_code))

    invalid_filenames = [
        p.name for p in corpus_path.iterdir() if p.is_file() and not is_valid(p.name)
    ]
    _report_possible_failures(
        invalid_filenames,
        "filename adheres to format `{language code}-{project}.txt`",
    )

def test_certain_bibles_exist_in_corpus(corpus_path: Path):
    """Tests if specific, expected Bible extract files exist."""
    filenames = [
        "eng-eng-kjv.txt",
        # TODO - add more on advice of someone familiar with stable versions
        # We want versions that are unlikely to be removed from eBible.org over time
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
        "eng-engylt.txt",
        "eng-engULB.txt",
        "fra-fraLSG.txt",
        # TODO - add more on advice of someone familiar with stable versions
        # We want versions that are unlikely to be removed from eBible.org over time
        # and have complete OT and NT (and no ranges)
    ]

    # Note that the extract file is structured:
    # - ot
    # - nt
    # - deuterocanonical
    # So we're just checking the front of the extract file has verses
    num_ot_nt_verses = 31170 # Expected lines for OT+NT

    def is_99_percent_complete(path: Path) -> bool:
        """
        Returns if the extract at the path has >= 99% OT and NT verses.
        We don't require 100% to cater for random glitches like missing verses,
        slightly different versifications and other weird little differences.
        """
        if not path.exists():
             pytest.warn(f"File not found for completeness check: {path.name}")
             return False # Cannot be complete if it doesn't exist
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                lines = fp.readlines() # Read all lines for simplicity here
                if len(lines) < num_ot_nt_verses:
                     # Optimization: If total lines < expected, it can't be complete
                     return False
                # Count non-empty lines within the expected range
                non_empty_verse_count = sum(
                    1 for i, line in enumerate(lines)
                    if i < num_ot_nt_verses and line.strip()
                )
                return (non_empty_verse_count / num_ot_nt_verses) >= 0.99
        except Exception as e:
            pytest.warn(f"Could not read or check completeness of {path.name}: {e}")
            return False # Treat errors as incomplete

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
            # Iterate directly over the file handle for line counting (efficient)
            # Use binary mode 'rb' as line endings don't matter for counting
            with open(path, "rb") as fp:
                num_lines = sum(1 for _ in fp)
            return num_lines == expected_num_lines
        except Exception as e:
             pytest.warn(f"Could not read or count lines in {path.name}: {e}")
             return False # Treat errors as incorrect line count

    bibles_with_incorrect_num_lines = [
        p.name for p in corpus_path.iterdir()
        if p.is_file() and p.suffix == '.txt' and not has_correct_number_lines(p)
    ]
    _report_possible_failures(
        bibles_with_incorrect_num_lines,
        f"has expected number of lines ({expected_num_lines})",
    )

# """
# Smoke tests for checking the extract files.
# These are intended to be run when the extract files are updated.
# By default it will use the `corpus` dir in the base of the project,
# but you can set the `CORPUS_PATH` environment variable to point it somewhere else -
# this might be useful during development.
# See https://github.com/BibleNLP/ebible/issues/53 for more context.

# Tests are intended to be run from this directory, e.g.
#   python smoke_tests.py
#     or
#   poetry run python smoke_tests.py

# Currently it doesn't have any special dependencies so should not need poetry or any particular
# python version.

# Note these tests are quite CPU intensive so some thought is needed if we decide
# to integrate them into the build pipeline.
# """

# import unittest
# import os
# from pathlib import Path
# from typing import List
# import regex


# class SmokeTest(unittest.TestCase):
#     def setUp(self):
#         corpus_path_lookup = os.getenv("CORPUS_PATH")
#         if corpus_path_lookup:
#             self.corpus_path = Path(corpus_path_lookup)
#         else:
#             # Assumes script is run from ebible_code/ebible_code/
#             self.corpus_path = Path(__file__).parent.parent.parent / "corpus"

#     def test_number_of_files(self) -> None:
#         all_extract_files = list(self.corpus_path.iterdir()) # Use pathlib
#         self.assertGreater(
#             len(all_extract_files),
#             1000,
#             "There should be at least 1000 corpus files",
#         )

#     def test_all_files_have_at_least_400_verses(self) -> None:

#         def verse_count(path: Path) -> int:
#             with open(path, "r") as fp:
#                 # Count non-empty lines more directly
#                 return sum(1 for line in fp if line.strip())

#         paths_with_less_than_400_verses = [
#             p.name for p in self.corpus_path.iterdir() if p.is_file() and verse_count(p) < 400
#         ]


#         self._report_possible_failures(
#             paths_with_less_than_400_verses, "file has at least 400 non-empty verses"
#         )

#     def test_filename_format_correct(self) -> None:

#         def is_valid(filename: str) -> bool:
#             parts = filename.split("-")
#             if len(parts) < 2:
#                 return False
#             language_code = parts[0]
#             # Anchor the regex for safety
#             if not regex.match("^[a-z]{2,3}$", language_code):
#                 return False
#             if not filename.endswith(".txt"):
#                 return False
#             return True

#         invalid_filenames = [
#             p.name for p in self.corpus_path.iterdir() if p.is_file()
#             if not is_valid(filename)
#         ]

#         self._report_possible_failures(
#             invalid_filenames,
#             "filename adheres to format `{language code}-{project}.txt`",
#         )

#     def test_certain_bibles_exist_in_corpus(self):
#         filenames = [
#             "eng-eng-kjv.txt",
#             # TODO - add more on advice of someone familiar with stable versions
#             # We want versions that are unlikely to be removed from eBible.org over time
#         ]

#         missing_bibles = [
#             filename
#             for filename in filenames
#             if not (self.corpus_path / filename).exists()
#         ]

#         self._report_possible_failures(
#             missing_bibles, "extract file exists for the Bible"
#         )

#     def test_certain_bibles_have_complete_OT_NT(self):
#         filenames = [
#             "eng-engylt.txt",
#             "eng-engULB.txt",
#             "fra-fraLSG.txt",
#             # TODO - add more on advice of someone familiar with stable versions
#             # We want versions that are unlikely to be removed from eBible.org over time
#             # and have complete OT and NT (and no ranges)
#         ]

#         # Note that the extract file is structured:
#         # - ot
#         # - nt
#         # - deuterocanonical
#         # So we're just checking the front of the extract file has verses
#         num_ot_nt_verses = 31170

#         def is_99_percent_complete(path: Path) -> int:
#             """
#             Returns if the extract at the path has 99% OT and NT.
#             We don't require 100% to cater for random glitches like missing verses,
#             slightly different versifications and other weird little differences
#             """
#             with open(path, "r") as fp:
#                 lines = fp.readlines()
#                 # Use generator expression and sum
#                 non_empty_verse_count = sum(1 for i, line in enumerate(lines) if i < num_ot_nt_verses and line.strip())
#                 return ( non_empty_verse_count / num_ot_nt_verses
#                     > 0.99
#                 )

#         incomplete_bibles = [
#             filename
#             for filename in filenames
#             if not is_99_percent_complete(self.corpus_path / filename)
#         ]

#         self._report_possible_failures(
#             incomplete_bibles,
#             "has at least 99% of the verses corresponding to the OT and NT",
#         )

#     def test_all_files_have_correct_number_of_lines(self):
#         expected_num_lines = 41899

#         def has_correct_number_lines(path: Path) -> int:
#             # Iterate directly over the file handle for line counting
#             with open(path, "rb") as fp:
#                 num_lines = sum(1 for _ in fp)
#                 return num_lines == expected_num_lines

#         bibles_with_incorrect_num_lines = [
#             p.name for p in self.corpus_path.iterdir() if p.is_file()
#             if not has_correct_number_lines(p)
#         ]

#         self._report_possible_failures(
#             bibles_with_incorrect_num_lines,
#             f"has expected number of lines ({expected_num_lines})",
#         )

#     def _report_possible_failures(self, failures: List, condition: str) -> None:
#         if failures:
#             self.fail(
#                 f"{len(failures)} found not matching condition: {condition}.\n"
#                 + f"{','.join(failures)}" # Consider limiting the number of failures printed
#             )


# if __name__ == "__main__":
#     unittest.main()
