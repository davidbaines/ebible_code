import argparse
import csv
import os
import re
import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

try:
    from machine.corpora import ParatextTextCorpus
    # from machine.scripture import VerseRef # For type hinting if needed
except ImportError:
    print("Error: The 'machine' library (SIL NLP toolkit) is not installed or accessible.", file=sys.stderr)
    print("Please install it: pip install sil-machine", file=sys.stderr)
    sys.exit(1)

# --- Logging Setup ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Constants for VRS file generation ---
# Copied from count_verses.py for consistent book ordering
BOOK_ORDER = [
    "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT", "1SA", "2SA",
    "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST", "JOB", "PSA", "PRO",
    "ECC", "SNG", "ISA", "JER", "LAM", "EZK", "DAN", "HOS", "JOL", "AMO",
    "OBA", "JON", "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL",
    # NT
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
    # Deuterocanon / Apocrypha
    "TOB", "JDT", "ESG", "WIS", "SIR", "BAR", "LJE", "S3Y", "SUS", "BEL",
    "1MA", "2MA", "3MA", "4MA", "1ES", "2ES", "MAN", "PS2", "ODA", "PSS",
    "EZA", "5EZ", "6EZ", "DAG", "LAO", "FRT", "BAK", "OTH", "CNC", "GLO",
    "TDX", "NDX", "XXA", "XXB", "XXC", "XXD", "XXE", "XXF", "XXG"
]
BOOK_SORT_KEY = {book: i for i, book in enumerate(BOOK_ORDER)}

# Regex for parsing verse strings like "1", "1a", "1-2" from corpus VerseRef.verse
VERSE_STR_PATTERN = re.compile(r"(\d+)([a-zA-Z]?)")

def parse_verse_string(verse_str: str) -> tuple[int, str]:
    """Parses a verse string (e.g., '1', '1a') into number and subdivision."""
    if not verse_str: # Handle empty verse strings if they occur
        return 0, ""
    match = VERSE_STR_PATTERN.match(verse_str)
    if match:
        return int(match.group(1)), match.group(2)
    try:
        return int(verse_str), ""
    except ValueError:
        logger.warning(f"Could not parse verse string '{verse_str}' into an integer.")
        return 0, ""

def get_project_name(project_path: Path) -> str:
    """Derives a project name from the project folder name."""
    return project_path.name

def is_paratext_folder(candidate_path: Path) -> bool:
    """
    Checks if a folder 'looks like' a Paratext project.
    A Paratext project folder contains .SFM or .usfm files (case-insensitive)
    and also a Settings.xml file.
    """
    if not candidate_path.is_dir():
        return False

    has_settings_xml = (candidate_path / "Settings.xml").is_file()
    if not has_settings_xml:
        return False

    has_sfm_files = any(candidate_path.glob("*.[sS][fF][mM]"))
    has_usfm_files = any(candidate_path.glob("*.[uU][sS][fF][mM]"))

    return has_sfm_files or has_usfm_files

def generate_vrs_from_project(project_path: Path):
    """
    Generates a .vrs file from the actual content of a Paratext project.
    """
    project_name = get_project_name(project_path)
    logger.info(f"Processing project: {project_name} at {project_path}")

    try:
        corpus = ParatextTextCorpus(str(project_path))
    except Exception as e:
        logger.error(f"Could not initialize ParatextTextCorpus for {project_name}: {e}")
        return

    # Structure: {book_id: {chapter_num: max_verse_num}}
    verse_data = defaultdict(lambda: defaultdict(int))
    processed_verse_refs = 0

    for text_row in corpus: # Iterate directly over the corpus
        processed_verse_refs += 1
        verse_ref = text_row.ref # Get the VerseRef from the TextRow
        book_id = verse_ref.book.upper() 
        chapter_num = int(verse_ref.chapter) # Convert chapter to int
        # verse_ref.verse can be like '1', '1a', '1-2', '1a-2b'
        # We need the primary numeric part of the first verse in a range or segment.
        # For '1-2', parse_verse_string('1-2') might fail if not handled.
        # Let's assume verse_ref.verse for single verses or start of range is parsable.
        # If verse_ref.verse is '1-2', we care about '1'.
        # The ParatextTextCorpus usually yields individual verse refs, so '1-2' as verse_ref.verse is less common.
        # It's more likely verse_ref.verse_num_str or similar would give '1' or '1a'.
        # Let's assume verse_ref.verse is the string for the specific verse number (e.g., "1", "1a")
        
        # If verse_ref.verse can be a range string like "1-3", we need to parse it.
        # For simplicity, we'll assume verse_ref.verse is the verse identifier string like "1" or "1a".
        # If it can be "1-3", the logic in parse_verse_string needs to be more robust or we use verse_ref.verse_num
        
        verse_num_str_to_parse = verse_ref.verse.split('-')[0] # Take the start of a range if present
        verse_num, _ = parse_verse_string(verse_num_str_to_parse)


        if chapter_num > 0 and verse_num > 0: # Ensure valid chapter and verse numbers
            verse_data[book_id][chapter_num] = max(
                verse_data[book_id][chapter_num], verse_num
            )
        elif chapter_num <= 0:
            logger.warning(f"Skipping verse_ref with invalid chapter {chapter_num} in {book_id} for {project_name}")
        # verse_num could be 0 if parse_verse_string failed, already logged by it.

    if processed_verse_refs == 0:
        logger.warning(f"No verse references found in project {project_name}. Skipping VRS generation.")
        return
    if not verse_data:
        logger.warning(f"No valid verse data collected for project {project_name} after processing {processed_verse_refs} refs. Skipping VRS generation.")
        return

    # --- Generate .vrs file content ---
    vrs_lines = []
    vrs_lines.append(f"# Versification for project: {project_name}")
    vrs_lines.append(f"# Generated on: {datetime.now().isoformat()}")
    vrs_lines.append("#")
    vrs_lines.append("# List of books, chapters, verses")
    vrs_lines.append("# One line per book.")
    vrs_lines.append("# One entry for each chapter.")
    vrs_lines.append("# Verse number is the maximum verse number for that chapter.")
    vrs_lines.append("#-----------------------------------------------------------")

    sorted_books = sorted(verse_data.keys(), key=lambda b: (BOOK_SORT_KEY.get(b, float('inf')), b))

    for book_id in sorted_books:
        chapters = verse_data[book_id]
        if not chapters: # Skip book if no chapter data was collected
            logger.debug(f"Skipping book {book_id} for {project_name} as it has no chapter data.")
            continue

        chapter_strings = []
        # Sort chapters numerically
        for chapter_num in sorted(chapters.keys()):
            max_verse = chapters[chapter_num]
            if max_verse > 0: # Only include chapters with actual verses
                chapter_strings.append(f"{chapter_num}:{max_verse}")
        
        if chapter_strings: # Only add line if there's valid chapter data
            vrs_lines.append(f"{book_id} {' '.join(chapter_strings)}")
        else:
            logger.debug(f"Skipping book {book_id} for {project_name} as no valid chapter:verse entries were generated.")


    # --- Write .vrs file ---
    vrs_filename = project_path / f"{project_name}.vrs"
    try:
        with open(vrs_filename, "w", encoding="utf-8") as f:
            for line in vrs_lines:
                f.write(line + "\n")
        logger.info(f"Successfully generated VRS file: {vrs_filename}")
    except IOError as e:
        logger.error(f"Could not write VRS file {vrs_filename}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing {vrs_filename}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generates a .vrs file from the content of Paratext projects."
    )
    parser.add_argument(
        "folder",
        nargs="?", # Makes the argument optional
        type=Path,
        help="Root directory to scan for Paratext projects. Defaults to EBIBLE_DATA_DIR environment variable.",
    )
    
    
    args = parser.parse_args()
    load_dotenv()

    if args.folder:
        folder = Path(args.folder)
        logger.info(f"Using folder from command line argument: {folder}")
    else:
        folder = Path(os.getenv("EBIBLE_DATA_DIR")) / "projects"
        if folder:
            logger.info(f"Using project folder within the EBIBLE_DATA_DIR environment variable: {folder}")
        else:
            logger.error(
                "No folder specified and EBIBLE_DATA_DIR environment variable is not set."
            )
            parser.print_help()
            sys.exit(1)

    if not folder.is_dir():
        logger.error(f"Directory does not exist: {folder}")
        sys.exit(1)

    if is_paratext_folder(folder):
        logger.info(f"Directory: {folder} is a single paratext project. Generating vrs file for it.")
        generate_vrs_from_project(folder)
    
    else:
        logger.info(f"Scanning for Paratext projects in: {folder}")
        project_count = 0
        found_projects = 0
        for item in folder.rglob("*"): # rglob walks recursively
            if item.is_dir():
                project_count +=1
                if is_paratext_folder(item):
                    found_projects +=1
                    generate_vrs_from_project(item)
                # Limit recursion depth if necessary, or rglob might go too deep in some structures.
                # For now, rglob is fine.

        logger.info(f"Scan complete. Checked {project_count} directories, found and processed {found_projects} Paratext projects.")

if __name__ == "__main__":
    main()
