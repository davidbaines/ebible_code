import os

import pytest
from dotenv import load_dotenv
from pathlib import Path

from ebible_code.settings_file import get_versification_with_scoring
from ebible_code.rename_usfm import get_destination_file_from_book

load_dotenv()

_ebible_data_dir_env = os.getenv("EBIBLE_DATA_DIR")
EBIBLE_DATA_PROJECTS = Path(_ebible_data_dir_env).resolve() / "projects" if _ebible_data_dir_env else None


# --- Test get_versification_with_scoring ---

versification_test_cases = [
    ("eng-web-c",                        "English"),
    ("eng-vul",                          "Vulgate"),
    ("rus-synod-usfm-from-textus-rec",   "Russian Orthodox"),
    ("rus-vrt",                          "Russian Protestant"),
    ("abt-maprik",                       "English"),
    ("eng-uk-lxx2012",                   "Septuagint"),
]

@pytest.mark.parametrize("project_folder_name, expected_versification", versification_test_cases)
def test_get_versification(project_folder_name, expected_versification):
    """Tests get_versification_with_scoring against projects with known versifications."""
    if EBIBLE_DATA_PROJECTS is None:
        pytest.skip("EBIBLE_DATA_DIR not set in .env")
    project_path = EBIBLE_DATA_PROJECTS / project_folder_name
    if not project_path.is_dir():
        pytest.skip(f"Test project directory not found: {project_path}")

    actual = get_versification_with_scoring(project_path)
    assert actual == expected_versification, (
        f"For '{project_folder_name}': expected '{expected_versification}', got '{actual}'"
    )


# --- Test get_destination_file_from_book ---

rename_test_cases = [
    # (parent_folder,      input_filename,  expected_output_filename)
    ("abt-maprik",         "MAT.usfm",      "41MATabt.SFM"),   # NT book: 40 + 1 = 41
    ("abt-maprik",         "ROM.usfm",      "46ROMabt.SFM"),   # NT book: 45 + 1 = 46
    ("abt-maprik",         "REV.usfm",      "67REVabt.SFM"),   # NT book: 66 + 1 = 67
    ("eng-web-c",          "GEN.usfm",      "01GENeng.SFM"),   # OT book: 1 (unchanged)
    ("eng-web-c",          "EXO.usfm",      "02EXOeng.SFM"),   # OT book: 2 (unchanged)
    ("eng-web-c",          "RUT.usfm",      "08RUTeng.SFM"),   # OT book: 8 (unchanged)
    ("eng-web-c",          "1MA.usfm",      "781MAeng.SFM"),   # DC book: 77 + 1 = 78
    ("eng-web-c",          "2MA.usfm",      "792MAeng.SFM"),   # DC book: 78 + 1 = 79
    ("eng-uk-lxx2012",     "FRT.usfm",      "101FRTeng.SFM"),  # non-OT book: 100 + 1 = 101
]

@pytest.mark.parametrize("folder, input_name, expected_name", rename_test_cases)
def test_get_destination_file_from_book(folder, input_name, expected_name):
    """Tests that USFM files are renamed to the Paratext NNBBBISO.SFM convention."""
    result = get_destination_file_from_book(Path(folder) / input_name)
    assert result is not None, f"Expected a result for {input_name} in {folder}"
    assert result.name == expected_name, (
        f"For {input_name} in {folder}: expected {expected_name}, got {result.name}"
    )
