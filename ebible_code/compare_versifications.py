# filename: compare_versifications.py

import argparse
import csv
import os
import re
import sys
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from dotenv import load_dotenv
load_dotenv()


import pandas as pd

try:
    from machine.scripture.verse_ref import VersificationType
    print(f"Successfully imported VersificationType: {VersificationType.ENGLISH}") # Test print
    CAN_IMPORT_VERSE_REF = True
except ImportError as e:
    print(f"Warning: Could not import from verse_ref.py: {e}. Versification column ordering will be alphabetical.", file=sys.stderr)
    print(f"ERROR: Could not import 'VersificationType' from 'machine.scripture.verse_ref': {e}", file=sys.stderr)
    print("Please ensure the 'sil-machine' library is installed and accessible in your PYTHONPATH.", file=sys.stderr)
    CAN_IMPORT_VERSE_REF = False
    # sys.exit(1) # Optionally exit if the import is critical


# --- Logging Setup ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Constants for VRS file generation ---
# Copied from generate_project_vrs.py for consistent book ordering
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

# Regex for parsing chapter:verse entries in VRS files
# Example line: GEN 1:31 2:25 3:24
VRS_LINE_PATTERN = re.compile(r"^([A-Z0-9]{3})\s+(.*)")
CHAPTER_VERSE_PATTERN = re.compile(r"(\d+):(\d+)")


def parse_vrs_file(vrs_path: Path) -> Dict[str, Dict[int, int]]:
    """
    Parses a .vrs file into a dictionary: {book_id: {chapter_num: max_verse}}.
    """
    parsed_data = defaultdict(lambda: defaultdict(int))
    if not vrs_path.is_file():
        logger.warning(f"VRS file not found: {vrs_path}")
        return parsed_data

    try:
        with open(vrs_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                match = VRS_LINE_PATTERN.match(line)
                if match:
                    book_id = match.group(1).upper()
                    chapters_part = match.group(2)
                    
                    for cv_match in CHAPTER_VERSE_PATTERN.finditer(chapters_part):
                        try:
                            chapter = int(cv_match.group(1))
                            verses = int(cv_match.group(2))
                            parsed_data[book_id][chapter] = verses
                        except ValueError:
                            logger.warning(f"Malformed chapter:verse '{cv_match.group(0)}' in {vrs_path.name} for book {book_id}")
                else:
                    # Could be a mapping line or other format we are ignoring for now
                    pass # logger.debug(f"Skipping non-data line in {vrs_path.name}: {line[:50]}")
    except Exception as e:
        logger.error(f"Error parsing VRS file {vrs_path}: {e}")
    return parsed_data


def find_vrs_files(directory: Path, pattern: str = "*.vrs") -> List[Path]:
    """Finds VRS files in a given directory."""
    if not directory.is_dir():
        logger.warning(f"Directory not found: {directory}")
        return []
    return sorted(list(directory.glob(pattern)))

def find_project_vrs_files(projects_root_dirs: List[Path]) -> Dict[str, Path]:
    """
    Finds <project_name>.vrs files in project subdirectories.
    Returns a dictionary: {project_name: path_to_vrs_file}.
    """
    project_vrs_map = {}
    for root_dir in projects_root_dirs:
        if not root_dir.is_dir():
            logger.warning(f"Projects root directory not found: {root_dir}")
            continue
        for item in root_dir.iterdir(): # Iterate only one level deep for project folders
            if item.is_dir():
                project_name = item.name
                expected_vrs_file = item / f"{project_name}.vrs"
                if expected_vrs_file.is_file():
                    if project_name in project_vrs_map:
                        logger.warning(f"Duplicate project name '{project_name}' found. Using VRS from {expected_vrs_file} over {project_vrs_map[project_name]}")
                    project_vrs_map[project_name] = expected_vrs_file
                # else:
                #     logger.debug(f"No {project_name}.vrs found in {item}")
    return project_vrs_map


def get_default_paths():
    """Gets default paths based on EBIBLE_DATA_DIR environment variable."""
    ebible_data_dir_str = os.getenv("EBIBLE_DATA_DIR")
    if not ebible_data_dir_str:
        logger.error("EBIBLE_DATA_DIR environment variable is not set. Cannot determine default paths.")
        sys.exit(1)
    
    base_path = Path(ebible_data_dir_str)
    defaults = {
        "common_vrs_dir": base_path / "assets",
        "projects_dirs": [base_path / "projects"], # Default to a list with one item
        "output_csv": base_path / "metadata" / "compare_versifications.csv"
    }
    return defaults


def main():
    default_paths = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Compares verse counts from multiple .vrs files and project-specific .vrs files."
    )
    parser.add_argument(
        "--common_vrs_dir",
        type=Path,
        required=False,
        default=default_paths["common_vrs_dir"],
        help=f"Directory containing common .vrs files. Defaults to: {default_paths['common_vrs_dir']}",
    )
    parser.add_argument(
        "--projects_dirs",
        type=Path,
        nargs='+',
        required=False,
        default=default_paths["projects_dirs"],
        help=f"Root directory/directories containing Paratext project folders. Defaults to: {default_paths['projects_dirs']}",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=False,
        default=default_paths["output_csv"],
        help=f"Path to save the output comparison CSV file. Defaults to: {default_paths['output_csv']}",
    )
    args = parser.parse_args()

    # --- 1. Find and parse all VRS files ---
    all_parsed_vrs: Dict[str, Dict[str, Dict[int, int]]] = {} # {source_name: {book: {chap: verse}}}
    
    # Parse common VRS files
    logger.info(f"Looking for common VRS files in: {args.common_vrs_dir}")
    common_vrs_files = find_vrs_files(args.common_vrs_dir)
    if not common_vrs_files:
        logger.warning(f"No common .vrs files found in {args.common_vrs_dir}. Proceeding without them.")
    for vrs_file in common_vrs_files:
        source_name = vrs_file.stem # e.g., "vul" from "vul.vrs"
        logger.info(f"Parsing common VRS: {vrs_file.name} (as '{source_name}')")
        all_parsed_vrs[source_name] = parse_vrs_file(vrs_file)

    # Parse project-specific VRS files
    logger.info(f"Looking for project VRS files in: {args.projects_dirs}")
    project_vrs_map = find_project_vrs_files(args.projects_dirs)
    if not project_vrs_map:
        logger.warning(f"No project-specific .vrs files found in subdirectories of {args.projects_dirs}. Proceeding without them.")
    for project_name, vrs_file in project_vrs_map.items():
        logger.info(f"Parsing project VRS: {vrs_file.name} (for project '{project_name}')")
        all_parsed_vrs[project_name] = parse_vrs_file(vrs_file)

    if not all_parsed_vrs:
        logger.error("No VRS files (common or project) were found or parsed. Cannot generate comparison. Exiting.")
        sys.exit(1)

    # --- 2. Create a master list of all unique (Book, Chapter) pairs ---
    master_book_chapter_set: Set[Tuple[str, int]] = set()
    for source_data in all_parsed_vrs.values():
        for book_id, chapters in source_data.items():
            for chapter_num in chapters.keys():
                master_book_chapter_set.add((book_id, chapter_num))

    if not master_book_chapter_set:
        logger.error("No book/chapter data found in any parsed VRS files. Exiting.")
        sys.exit(1)
        
    # Sort the master list: by book (using BOOK_SORT_KEY), then by chapter number
    sorted_master_list = sorted(
        list(master_book_chapter_set),
        key=lambda bc: (BOOK_SORT_KEY.get(bc[0], float('inf')), bc[0], bc[1])
    )
    logger.info(f"Found {len(sorted_master_list)} unique Book-Chapter pairs across all VRS files.")

    # --- 3. Prepare data for DataFrame ---
    output_data_list = []
    source_names = sorted(all_parsed_vrs.keys()) # Consistent column order

    for book_id, chapter_num in sorted_master_list:
        row = {"Book": book_id, "Chapter": chapter_num}
        for source_name in source_names:
            verse_count = all_parsed_vrs[source_name].get(book_id, {}).get(chapter_num)
            # Use source_name directly as column suffix
            row[f"{source_name}_verses"] = verse_count # Will be NaN if verse_count is None
        output_data_list.append(row)

    # --- 4. Create DataFrame and save to CSV ---
    df = pd.DataFrame(output_data_list)
    
    # Reorder columns to have Book, Chapter first, then sorted source_names
    fixed_cols = ["Book", "Chapter"]
    data_cols = [f"{source_name}_verses" for source_name in source_names]
    df = df[fixed_cols + data_cols]

    try:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        df.to_csv(args.output_csv, index=False, encoding='utf-8')
        logger.info(f"Successfully generated comparison CSV: {args.output_csv}")
    except Exception as e:
        logger.error(f"Error writing CSV file {args.output_csv}: {e}")

if __name__ == "__main__":
    main()
