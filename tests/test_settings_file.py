import os

import pytest
from dotenv import load_dotenv
from pathlib import Path

from ebible_code.settings_file import get_book_names, get_versification, get_vrs_diffs

load_dotenv()

# Define base paths relative to the repository root and relative to the EBIBLE_DATA_DIR

ASSETS_PATH = Path("ebible_code/assets") # Path to where assets like vrs_diffs.yaml live
EBIBLE_DATA_DIR = Path(os.getenv("EBIBLE_DATA_DIR")).resolve()
EBIBLE_DATA_PROJECTS = EBIBLE_DATA_DIR / "projects"

# Load versification differences once for all tests
vrs_diffs = get_vrs_diffs()

# --- Test get_versification ---
# Define test cases: list of tuples (project_subfolder_name, expected_versification)
versification_test_cases = [
    ("eng-web-c", "English"),      # Standard English versification, also the default
    ("eng-vul", "Vulgate"),
    ("rus-synod-usfm-from-textus-rec", "Russian Orthodox"),
    ("rus-vrt", "Russian Protestant"),
    # Add a case that likely doesn't have specific diff markers and should use the default
    ("abt-maprik", "English"),
    # Add more test cases as needed
    ("eng-uk-lxx2012", "Septuagint"),
]

@pytest.mark.parametrize("project_folder_name, expected_versification", versification_test_cases)
def test_get_versification(project_folder_name, expected_versification):
    """Tests get_versification against various projects with known versifications."""
    project_path = EBIBLE_DATA_PROJECTS / project_folder_name
    
    # Ensure the project directory exists before running the test
    if not project_path.is_dir():
        pytest.skip(f"Test project directory not found: {project_path}")

    actual_versification = get_versification(project_path, vrs_diffs)
    assert actual_versification == expected_versification, \
        f"For project '{project_folder_name}': Expected versification '{expected_versification}', but found '{actual_versification}'"


# --- Test get_book_names ---
def test_get_book_names():
    test_cases = [
        ("abt-maprik", ["41MATabt.SFM", "46ROMabt.SFM", "67REVabt.SFM"]),
        ("ahr", ["41MATahr.SFM","42MRKahr.SFM"]), 
        ("eng-web-c", ["01GENeng.SFM", "02EXOeng.SFM", "08RUTeng.SFM", "781MAeng.SFM", "792MAeng.SFM"]),
        ("eng-uk-lxx2012", ["101FRTeng.SFM"])
    ]

    for folder_name, expected_filenames in test_cases:
        project_folder = EBIBLE_DATA_PROJECTS / folder_name
        if not project_folder.is_dir():
            pytest.skip(f"Test project directory not found: {project_folder}")

        filenames = set(file.name for file in project_folder.glob("*.SFM"))

        for expected_filename in expected_filenames:
            assert expected_filename in filenames, f"Missing {expected_filename} in {folder_name}"