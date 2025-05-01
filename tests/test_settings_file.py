from pathlib import Path
from ebible_code.settings_file import get_book_names, get_versification, get_vrs_diffs

vrs_diffs = get_vrs_diffs()

# Define base paths relative to the repository root
# Note: The original code seemed to assume these paths were relative to the
# *changed* working directory. If get_versification/get_book_names expect
# paths relative to the ebible_code dir itself, adjustments might be needed
# inside those functions or how these paths are constructed here.
# Assuming they should be relative to the repo root for now.
ASSETS_PATH = Path("ebible_code/assets") # Path to where assets like vrs_diffs.yaml live
TEST_DATA_PATH = Path("tests/test_data") # Example: Assuming test data might live here

# --- Test get_versification ---

# Example versification path (adjust if needed based on actual project structure)
VERSIFICATIONS_PATH = TEST_DATA_PATH / "versifications"

def test_get_versification_lxx():
    # Assuming 'eng-lxx2012' is a directory within VERSIFICATIONS_PATH
    versification = get_versification(VERSIFICATIONS_PATH / "eng-lxx2012", vrs_diffs)
    assert versification == "Septuagint"

# --- Test get_book_names ---

# Example book names path (adjust if needed)
BOOK_NAMES_PATH = TEST_DATA_PATH / "book_names"

def test_get_book_names():
    # Assuming 'aai', 'ahr-NTAii20-Ahirani-Devanagari', etc. are directories within BOOK_NAMES_PATH
    aai_books = get_book_names(BOOK_NAMES_PATH / "aai")
    assert aai_books[0] == ("MAT", "46-MATaai.usfm")

    ahr_books = get_book_names(BOOK_NAMES_PATH / "ahr-NTAii20-Ahirani-Devanagari")
    assert ahr_books[0] == ("MAT", "MAT.usfm")

    am_books = get_book_names(BOOK_NAMES_PATH / "am_ulb")
    assert am_books[0] == ("GEN", "01-GEN.usfm")
