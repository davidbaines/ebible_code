# filename: compare_versifications.py

import argparse
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dotenv import load_dotenv
load_dotenv()


import pandas as pd

# --- Logging Setup ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

try:
    from machine.scripture.verse_ref import Versification, VersificationType
    from machine.scripture.canon import book_id_to_number, book_number_to_id, LAST_BOOK
    # Note: book_number_to_id and LAST_BOOK might be useful later for creating the master list
    logger.info(f"Successfully imported VersificationType from machine.scripture: {VersificationType.ENGLISH}")
    CAN_IMPORT_VERSE_REF = True
except ImportError as e:
    print(f"Warning: Could not import from verse_ref.py: {e}.", file=sys.stderr)
    sys.exit(1)


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
        "projects_dirs": [base_path / "projects", base_path / "private_projects"], # Default to a list
        "output_csv": base_path / "metadata" / "compare_versifications.csv"
    }
    return defaults


def main():
    default_paths = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Compares verse counts from multiple .vrs files and project-specific .vrs files."
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
    # Store loaded Versification objects: {source_name: Versification_object}
    all_loaded_versifications: Dict[str, Versification] = {}
    
    # Load built-in versifications from sil-machine
    builtin_vrs_source_names = []
    if CAN_IMPORT_VERSE_REF:
        logger.info("Loading built-in versifications from sil-machine library...")
        for vtype in VersificationType:
            if vtype == VersificationType.UNKNOWN: # Skip UNKNOWN type
                continue
            try:
                # The name of the versification object will be like "English", "Vulgate"
                # as defined by Versification._BUILTIN_VERSIFICATION_NAMES_TO_TYPES
                vrs_obj = Versification.get_builtin(vtype)
                # Use the official name from the vrs_obj for the dictionary key and column name
                all_loaded_versifications[vrs_obj.name] = vrs_obj
                builtin_vrs_source_names.append(vrs_obj.name)
                logger.info(f"Successfully loaded built-in versification: {vrs_obj.name} (Type: {vtype.name})")
            except Exception as e:
                logger.error(f"Failed to load built-in versification for type {vtype.name}: {e}")

    # Parse project-specific VRS files
    logger.info(f"Looking for project VRS files in: {args.projects_dirs}")
    project_vrs_map = find_project_vrs_files(args.projects_dirs)
    if not project_vrs_map:
        logger.warning(f"No project-specific .vrs files found in subdirectories of {args.projects_dirs}. Proceeding without them.")
    for project_name, vrs_file in project_vrs_map.items():
        logger.info(f"Parsing project VRS: {vrs_file.name} (for project '{project_name}')")
        try:
            # Use project_name as the Versification object's name
            loaded_vrs = Versification.load(vrs_file, fallback_name=project_name)
            all_loaded_versifications[project_name] = loaded_vrs
        except Exception as e:
            logger.error(f"Failed to load project VRS file {vrs_file.name} for project {project_name}: {e}")


    if not all_loaded_versifications:
        logger.error("No VRS files (common or project) were found or parsed. Cannot generate comparison. Exiting.")
        sys.exit(1)

    # --- 2. Create a master list of all unique (Book, Chapter) pairs ---
    master_book_chapter_set: Set[Tuple[str, int]] = set()
    for vrs_obj in all_loaded_versifications.values():
        for book_num_int in range(1, vrs_obj.get_last_book() + 1):
            book_id_str = book_number_to_id(book_num_int)
            if not book_id_str: # Skip if book_number_to_id returns empty (e.g., invalid num)
                continue
            for chapter_num_int in range(1, vrs_obj.get_last_chapter(book_num_int) + 1):
                # Check if the chapter actually has verses defined, not just implied by get_last_chapter
                if vrs_obj.get_last_verse(book_num_int, chapter_num_int) > 0:
                    master_book_chapter_set.add((book_id_str, chapter_num_int))

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
    common_vrs_ordered_names = []
    
    if CAN_IMPORT_VERSE_REF:
        # The order of builtin_vrs_source_names is already based on VersificationType enum order
        common_vrs_ordered_names = builtin_vrs_source_names
        
        # Project VRS files are those not in the builtin_vrs_source_names list
        project_vrs_names = sorted([
            name for name in all_loaded_versifications.keys() if name not in common_vrs_ordered_names
        ])
        source_names_ordered = common_vrs_ordered_names + project_vrs_names
    else:
        # Fallback if imports failed (though we exit earlier if CAN_IMPORT_VERSE_REF is False for critical parts)
        source_names_ordered = sorted(all_loaded_versifications.keys())

    logger.info(f"Column order for VRS sources: {source_names_ordered}")
    for book_id_str, chapter_num_int in sorted_master_list:
        row = {"Book": book_id_str, "Chapter": chapter_num_int}
        book_num_int = book_id_to_number(book_id_str) if CAN_IMPORT_VERSE_REF else 0

        for source_name in source_names_ordered:
            vrs_obj = all_loaded_versifications.get(source_name)
            verse_count = None
            if vrs_obj and CAN_IMPORT_VERSE_REF and book_num_int > 0:
                # Check if book and chapter are within the defined range for this vrs_obj
                if book_num_int <= vrs_obj.get_last_book() and \
                   chapter_num_int <= vrs_obj.get_last_chapter(book_num_int):
                    verse_count = vrs_obj.get_last_verse(book_num_int, chapter_num_int)
                    if verse_count == 0 and not (book_num_int > vrs_obj.get_last_book() or chapter_num_int > vrs_obj.get_last_chapter(book_num_int)):
                        # If get_last_verse returns 0 for a chapter that *should* exist,
                        # it might mean the chapter is defined but has no verses listed (or 0 verses).
                        # For CSV, we might prefer to show 0 rather than NaN in this specific case.
                        pass # verse_count is already 0
                    elif verse_count == 0: # Book/chapter out of range for this VRS
                        verse_count = None
            
            # Use source_name directly as column suffix
            row[f"{source_name}_verses"] = verse_count
        output_data_list.append(row)

    # --- 4. Create DataFrame and save to CSV ---
    df = pd.DataFrame(output_data_list)
    
    # Reorder columns to have Book, Chapter first, then sorted source_names
    fixed_cols = ["Book", "Chapter"]
    data_cols = [f"{source_name}_verses" for source_name in source_names_ordered]
    df = df[fixed_cols + data_cols]

    try:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        df.to_csv(args.output_csv, index=False, encoding='utf-8')
        logger.info(f"Successfully generated comparison CSV: {args.output_csv}")
    except Exception as e:
        logger.error(f"Error writing CSV file {args.output_csv}: {e}")

if __name__ == "__main__":
    main()
