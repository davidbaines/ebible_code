import argparse
import os
import re
import sys
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Set, Optional
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
    # For consistency with the merged code, we can alias this or pick one.
    # Let's stick to CAN_IMPORT_VERSE_REF as it's more specific to the scripture module.
    CAN_IMPORT_MACHINE = CAN_IMPORT_VERSE_REF 
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

# --- Scoring Weights (from update_project_versification_settings.py) ---
WEIGHT_BOOK = 0.0
WEIGHT_CHAPTER = 0.0
WEIGHT_VERSE_COUNT = 1.0


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
        "csv_file": base_path / "metadata" / "compare_versifications.csv" # Renamed from output_csv
    }
    return defaults

# --- Helper functions for Settings Update (from update_project_versification_settings.py) ---
def get_standard_versification_map() -> Dict[str, Dict]:
    """
    Creates a map from standard versification names (e.g., 'English')
    to their VersificationType enum and integer value.
    """
    if not CAN_IMPORT_MACHINE: # or CAN_IMPORT_VERSE_REF
        logger.warning("Cannot generate standard versification map due to import failure.")
        return {}
        
    std_vrs_map = {}
    for vtype in VersificationType: # type: ignore
        if vtype == VersificationType.UNKNOWN: # type: ignore
            continue
        try:
            vrs_obj = Versification.get_builtin(vtype) # type: ignore
            std_vrs_map[vrs_obj.name] = {"type": vtype, "value": vtype.value} # type: ignore
        except Exception as e:
            logger.warning(f"Could not load built-in versification for {vtype.name}: {e}") # type: ignore
    return std_vrs_map

def get_data_from_df_for_scoring(df: pd.DataFrame, col_name: str) -> Dict[tuple, int]:
    """Extracts {(book, chapter): verses} for a specific column from the DataFrame."""
    # Ensure 'Book' and 'Chapter' columns exist
    if 'Book' not in df.columns or 'Chapter' not in df.columns:
        logger.error("DataFrame for scoring is missing 'Book' or 'Chapter' columns.")
        return {}
    if col_name not in df.columns:
        logger.error(f"Column '{col_name}' not found in DataFrame for scoring.")
        return {}

    series_data = df[['Book', 'Chapter', col_name]].copy()
    series_data.rename(columns={col_name: 'Verses'}, inplace=True)
    series_data.dropna(subset=['Verses'], inplace=True)
    # Convert to int, coercing errors to NaN, then dropna again if any failed conversion
    series_data['Verses'] = pd.to_numeric(series_data['Verses'], errors='coerce')
    series_data.dropna(subset=['Verses'], inplace=True)
    series_data['Verses'] = series_data['Verses'].astype(int)
    return {(row.Book, row.Chapter): row.Verses for _, row in series_data.iterrows()}

def calculate_similarity_score(
    project_v_data: Dict[tuple, int], 
    standard_v_data: Dict[tuple, int],
    invariant_chapters: Set[Tuple[str, int]]
    ) -> float:
    """Calculates the similarity score.
    Book score is based on all books.
    Chapter and Verse scores focus on non-invariant chapters within common books.
    """
    # --- Book Score (overall book presence) ---
    project_books_overall = {book for book, chap in project_v_data}
    if not project_books_overall:
        return 0.0 # No books in project, no similarity

    standard_books_defined_overall = {book for book, chap in standard_v_data}
    common_books = project_books_overall.intersection(standard_books_defined_overall)
    book_score = len(common_books) / len(project_books_overall) if project_books_overall else 0.0

    # --- Filter data to focus on non-invariant chapters within common books ---
    # Project's (book, chapter) pairs that are in common books AND are not invariant
    project_bc_for_detailed_comparison = {
        (b, c) for (b, c) in project_v_data.keys()
        if b in common_books and (b, c) not in invariant_chapters
    }
    # Standard's (book, chapter) pairs that are in common books AND are not invariant
    standard_bc_for_detailed_comparison = {
        (b, c) for (b, c) in standard_v_data.keys()
        if b in common_books and (b, c) not in invariant_chapters
    }

    # --- Chapter Score (for non-invariant chapters in common books) ---
    # How many of the project's non-invariant chapters (in common books) are also defined in the standard's non-invariant set?
    common_differentiating_chapters = project_bc_for_detailed_comparison.intersection(standard_bc_for_detailed_comparison)
    
    num_project_differentiating_chapters = len(project_bc_for_detailed_comparison)
    chapter_score = len(common_differentiating_chapters) / num_project_differentiating_chapters \
        if num_project_differentiating_chapters else 0.0

    # --- Verse Count Score (for common differentiating chapters) ---
    matching_verse_count_differentiating_chapters = 0
    # We iterate over chapters that are common AND differentiating
    for book, chap in common_differentiating_chapters:
        if project_v_data.get((book, chap)) == standard_v_data.get((book, chap)):
            matching_verse_count_differentiating_chapters += 1
    
    num_common_differentiating_chapters_for_verse_score = len(common_differentiating_chapters)
    verse_count_score = matching_verse_count_differentiating_chapters / num_common_differentiating_chapters_for_verse_score \
        if num_common_differentiating_chapters_for_verse_score else 0.0
    
    total_score = (WEIGHT_BOOK * book_score) + \
                  (WEIGHT_CHAPTER * chapter_score) + \
                  (WEIGHT_VERSE_COUNT * verse_count_score)

    # Extended logging for debugging score components
    # project_id_for_log = "N/A" # Placeholder, ideally pass project_id for full context
    # standard_name_for_log = "N/A" # Placeholder, ideally pass standard_name for full context
    # logger.debug(f"Scores for Project vs Standard:")
    # logger.debug(f"  Common Books: {len(common_books)} / Project Books: {len(project_books_overall)}")
    # logger.debug(f"  Book Score: {book_score:.4f}")
    # logger.debug(f"  Project Differentiating Chapters (in common books): {num_project_differentiating_chapters}")
    # logger.debug(f"  Standard Differentiating Chapters (in common books): {len(standard_bc_for_detailed_comparison)}")
    # logger.debug(f"  Common Differentiating Chapters: {len(common_differentiating_chapters)}")
    # logger.debug(f"  Chapter Score (Differentiating): {chapter_score:.4f}")
    # logger.debug(f"  Matching Verse Count (Differentiating): {matching_verse_count_differentiating_chapters}")
    # logger.debug(f"  Verse Count Score (Differentiating): {verse_count_score:.4f}")
    # logger.debug(f"  Total Weighted Score: {total_score:.4f}")

    return total_score

def update_settings_xml(project_path: Path, new_versification_value: int):
    """Updates the Versification tag in the project's Settings.xml file."""
    settings_file = project_path / "Settings.xml"
    if not settings_file.is_file():
        logger.warning(f"Settings.xml not found in {project_path}. Skipping update.")
        return False
    try:
        tree = ET.parse(str(settings_file))
        root = tree.getroot()
        versification_tag = root.find("Versification")
        if versification_tag is None:
            logger.warning(f"<Versification> tag not found in {settings_file}. Skipping update.")
            return False
        old_value = versification_tag.text
        versification_tag.text = str(new_versification_value)
        tree.write(str(settings_file), encoding="utf-8", xml_declaration=False)
        logger.info(f"Updated {settings_file}: <Versification> from '{old_value}' to '{new_versification_value}'.")
        return True
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file {settings_file}: {e}")
    except Exception as e:
        logger.error(f"Error updating XML file {settings_file}: {e}")
    return False

def generate_comparison_csv(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    """Generates the versification comparison CSV file."""
    logger.info("--- Starting CSV Generation Phase ---")
    # The 'args' parameter contains all necessary parsed arguments from the main parser,
    # including projects_dirs and csv_file.
    # --- 1. Load/Parse all VRS files ---
    # Store loaded Versification objects: {source_name: Versification_object}
    all_loaded_versifications: Dict[str, Versification] = {}
    
    # Load built-in versifications from sil-machine
    builtin_vrs_source_names = []
    if CAN_IMPORT_VERSE_REF:
        logger.info("Loading built-in versifications from sil-machine library...")
        for vtype in VersificationType: # type: ignore
            if vtype == VersificationType.UNKNOWN: # type: ignore
                continue
            try:
                # The name of the versification object will be like "English", "Vulgate"
                # as defined by Versification._BUILTIN_VERSIFICATION_NAMES_TO_TYPES # type: ignore
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
            loaded_vrs = Versification.load(vrs_file, fallback_name=project_name) # Reverted to fallback_name
            all_loaded_versifications[project_name] = loaded_vrs
        except Exception as e:
            logger.error(f"Failed to load project VRS file {vrs_file.name} for project {project_name}: {e}")

    if not all_loaded_versifications:
        logger.error("No VRS files (common or project) were found or parsed. Cannot generate comparison. Exiting.")
        return None

    # --- 2. Create a master list of all unique (Book, Chapter) pairs ---
    master_book_chapter_set: Set[Tuple[str, int]] = set()
    for vrs_obj in all_loaded_versifications.values():
        for book_num_int in range(1, vrs_obj.get_last_book() + 1): # type: ignore
            book_id_str = book_number_to_id(book_num_int)
            if not book_id_str: # Skip if book_number_to_id returns empty (e.g., invalid num)
                continue
            for chapter_num_int in range(1, vrs_obj.get_last_chapter(book_num_int) + 1): # type: ignore
                # Check if the chapter actually has verses defined, not just implied by get_last_chapter
                if vrs_obj.get_last_verse(book_num_int, chapter_num_int) > 0:
                    master_book_chapter_set.add((book_id_str, chapter_num_int))

    if not master_book_chapter_set:
        logger.error("No book/chapter data found in any parsed VRS files. Exiting.")
        return None
        
    # Sort the master list: by book (using BOOK_SORT_KEY), then by chapter number
    sorted_master_list = sorted(
        list(master_book_chapter_set),
        key=lambda bc: (BOOK_SORT_KEY.get(bc[0], float('inf')), bc[0], bc[1])
    )
    logger.info(f"Found {len(sorted_master_list)} unique Book-Chapter pairs across all VRS files.")

    # --- 3. Prepare data for DataFrame ---
    output_data_list = []
    # common_vrs_ordered_names will be builtin_vrs_source_names
    
    if CAN_IMPORT_VERSE_REF:
        # The order of builtin_vrs_source_names is already based on VersificationType enum order
        # common_vrs_ordered_names = builtin_vrs_source_names # This is already set
        
        # Project VRS files are those not in the builtin_vrs_source_names list
        project_vrs_names = sorted([
            name for name in all_loaded_versifications.keys() if name not in builtin_vrs_source_names
        ])
        source_names_ordered = builtin_vrs_source_names + project_vrs_names
    else:
        # Fallback if imports failed (though we exit earlier if CAN_IMPORT_VERSE_REF is False for critical parts)
        source_names_ordered = sorted(list(all_loaded_versifications.keys()))

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
                   chapter_num_int <= vrs_obj.get_last_chapter(book_num_int): # type: ignore
                    verse_count = vrs_obj.get_last_verse(book_num_int, chapter_num_int)
                    if verse_count == 0 and not (book_num_int > vrs_obj.get_last_book() or \
                                                 chapter_num_int > vrs_obj.get_last_chapter(book_num_int)): # type: ignore
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
        args.csv_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        df.to_csv(args.csv_file, index=False, encoding='utf-8')
        logger.info(f"Successfully generated comparison CSV: {args.csv_file}")
        return df
    except Exception as e:
        logger.error(f"Error writing CSV file {args.csv_file}: {e}")
        return None

def update_versification_settings(df: pd.DataFrame, args: argparse.Namespace):
    """
    Determines the closest standard versification for projects using data from the DataFrame
    and updates their Settings.xml files.
    """
    logger.info("--- Starting Settings Update Phase ---")

    all_projects_scores_data = [] # To store scores for the new CSV

    std_vrs_map = get_standard_versification_map()
    if not std_vrs_map:
        logger.error("No standard versification mapping available. Cannot proceed with settings update.")
        return

    # --- Identify Invariant Chapters (chapters with same verse count across all defining standard versifications) ---
    standard_vrs_data_map = {}
    for std_name_key in std_vrs_map.keys():
        std_col_name = f"{std_name_key}_verses"
        if std_col_name in df.columns:
            standard_vrs_data_map[std_name_key] = get_data_from_df_for_scoring(df, std_col_name)
        else:
            # This case should ideally not happen if df is generated correctly with all std_vrs_map keys
            logger.warning(f"Standard versification column {std_col_name} (for {std_name_key}) not found in DataFrame. It will be excluded from invariant chapter check.")

    all_chapters_in_standards = set()
    for std_data in standard_vrs_data_map.values():
        all_chapters_in_standards.update(std_data.keys())

    invariant_chapters = set()
    if not all_chapters_in_standards:
        logger.warning("No chapters found in any standard versifications. Cannot identify invariant chapters.")
    else:
        for book_chap_tuple in all_chapters_in_standards:
            defined_verse_counts_for_chapter = []
            for std_data in standard_vrs_data_map.values():
                if book_chap_tuple in std_data: # Check if the standard versification defines this chapter
                    defined_verse_counts_for_chapter.append(std_data[book_chap_tuple])
            
            unique_defined_counts = set(defined_verse_counts_for_chapter)
            if len(unique_defined_counts) <= 1: # 0 or 1 unique verse count means it's invariant
                invariant_chapters.add(book_chap_tuple)
        logger.info(f"Identified {len(invariant_chapters)} invariant chapters out of {len(all_chapters_in_standards)} total unique chapters in standards.")
        # logger.debug(f"Sample of invariant chapters: {sorted(list(invariant_chapters))[:20]}")

    # Standard versification columns in the CSV (e.g., "English_verses")
    # These names come from Versification.get_builtin(vtype).name
    standard_vrs_csv_cols = [f"{name}_verses" for name in std_vrs_map.keys() if f"{name}_verses" in df.columns]
    logger.debug(f"std_vrs_map.keys(): {list(std_vrs_map.keys())}")
    logger.info(f"Identified standard versification CSV columns for scoring: {standard_vrs_csv_cols}")

    # Project columns are all other "*_verses" columns
    all_verses_cols = [col for col in df.columns if col.endswith("_verses")]
    logger.debug(f"All '*_verses' columns from DataFrame: {all_verses_cols}")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")

    project_csv_cols = [col for col in all_verses_cols if col not in standard_vrs_csv_cols]
    
    logger.info(f"Found {len(project_csv_cols)} project columns in CSV to process for settings update.")

    # Get the names of standard versifications to use as column headers in the scores CSV
    standard_versification_names_for_scores_csv = sorted(list(std_vrs_map.keys()))

    for project_col_name in project_csv_cols: # e.g., "aai_verses"
        project_id = project_col_name.replace("_verses", "") # e.g., "aai"
        logger.info(f"--- Processing project for settings update: {project_id} ---")

        project_v_data = get_data_from_df_for_scoring(df, project_col_name)
        if not project_v_data:
            logger.warning(f"No verse data found for project {project_id} in the DataFrame. Skipping settings update for this project.")
            continue

        current_project_scores = {"Project": project_id}
        best_score = -1.0
        best_std_vrs_name_key = None # This will be the key for std_vrs_map (e.g., "English")

        for std_csv_col_name in standard_vrs_csv_cols: # e.g., "English_verses"
            std_vrs_name_key = std_csv_col_name.replace("_verses", "") # e.g. "English"
            if std_vrs_name_key not in std_vrs_map:
                logger.warning(f"Standard versification '{std_vrs_name_key}' from column '{std_csv_col_name}' not in map. Skipping comparison.")
                continue

            standard_v_data = get_data_from_df_for_scoring(df, std_csv_col_name)
            if not standard_v_data:
                logger.debug(f"No data for standard versification {std_vrs_name_key} to compare with {project_id}.")
                continue
            
            current_score = calculate_similarity_score(project_v_data, standard_v_data, invariant_chapters)
            logger.debug(f"Score for {project_id} vs {std_vrs_name_key}: {current_score:.4f}")
            current_project_scores[std_vrs_name_key] = round(current_score, 4) # Store score for CSV

            if current_score > best_score:
                best_score = current_score
                best_std_vrs_name_key = std_vrs_name_key
        
        # Ensure all standard versifications have an entry in the current_project_scores, even if 0 or None
        for std_name in standard_versification_names_for_scores_csv:
            current_project_scores.setdefault(std_name, 0.0) # Default to 0.0 if no score was calculated (e.g. no data for standard)
        all_projects_scores_data.append(current_project_scores)

        if best_std_vrs_name_key:
            vrs_type_info = std_vrs_map[best_std_vrs_name_key]
            logger.info(f"Best match for {project_id}: {best_std_vrs_name_key} (Type: {vrs_type_info['type'].name}, Value: {vrs_type_info['value']}) with score: {best_score:.4f}")

            if not args.dry_run:
                project_path = None
                for p_dir_root in args.projects_dirs:
                    candidate_path = p_dir_root / project_id
                    if candidate_path.is_dir():
                        project_path = candidate_path
                        break
                
                if project_path:
                    update_settings_xml(project_path, vrs_type_info['value'])
                else:
                    logger.warning(f"Could not find project directory for {project_id} in {args.projects_dirs}. Cannot update Settings.xml.")
            else:
                logger.info(f"[DRY RUN] Would update Settings.xml for {project_id} to versification value {vrs_type_info['value']}.")
        else:
            logger.warning(f"Could not determine a best standard versification for project {project_id}.")

    # --- Write the scores to a new CSV file ---
    if all_projects_scores_data:
        # Format weights for filename, replacing '.' with '_' if they are floats, or just use int if they are.
        # Assuming weights can be float, let's format them consistently.
        # Using int() if they are whole numbers to avoid ".0"
        wb_str = str(int(WEIGHT_BOOK)) if WEIGHT_BOOK == int(WEIGHT_BOOK) else str(WEIGHT_BOOK).replace('.', '_')
        wc_str = str(int(WEIGHT_CHAPTER)) if WEIGHT_CHAPTER == int(WEIGHT_CHAPTER) else str(WEIGHT_CHAPTER).replace('.', '_')
        wv_str = str(int(WEIGHT_VERSE_COUNT)) if WEIGHT_VERSE_COUNT == int(WEIGHT_VERSE_COUNT) else str(WEIGHT_VERSE_COUNT).replace('.', '_')

        scores_filename = f"versification_scores_B{wb_str}_C{wc_str}_V{wv_str}.csv"
        scores_csv_path = args.csv_file.parent / scores_filename

        try:
            scores_df = pd.DataFrame(all_projects_scores_data)
            # Ensure consistent column order: Project, then sorted standard versification names
            ordered_score_cols = ["Project"] + standard_versification_names_for_scores_csv
            scores_df = scores_df[ordered_score_cols]
            scores_df.to_csv(scores_csv_path, index=False, encoding='utf-8', float_format='%.4f')
            logger.info(f"Successfully generated scores CSV: {scores_csv_path}")
        except Exception as e:
            logger.error(f"Error writing scores CSV file {scores_csv_path}: {e}")

    logger.info("Settings update phase complete.")


def main():
    if not CAN_IMPORT_VERSE_REF: # or CAN_IMPORT_MACHINE
        logger.error("Critical 'machine' library components could not be imported. Exiting.")
        sys.exit(1)

    default_paths = get_default_paths()
    if not default_paths: # Should not happen if EBIBLE_DATA_DIR is set
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Generates a versification comparison CSV and/or updates project Settings.xml based on it."
    )
    # Arguments for CSV generation and general paths
    parser.add_argument(
        "--projects_dirs",
        type=Path,
        nargs='+',
        required=False,
        default=default_paths["projects_dirs"],
        help=f"Root directory/directories containing Paratext project folders. Defaults to: {default_paths['projects_dirs']}",
    )
    parser.add_argument(
        "--csv_file",
        type=Path,
        required=False,
        default=default_paths["csv_file"],
        help=f"Path to save (if generating) or load (if updating only) the comparison CSV. Defaults to: {default_paths['csv_file']}",
    )
    # Arguments to control script actions
    parser.add_argument(
        "--skip_csv_generation",
        action="store_true",
        help="Skip the CSV generation phase. Requires --csv_file to exist if --update_settings is active.",
    )
    parser.add_argument(
        "--skip_settings_update",
        action="store_true",
        help="Skip the project Settings.xml update phase.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform all calculations for settings update but do not write changes to Settings.xml files."
    )
    args = parser.parse_args()

    comparison_df = None

    if not args.skip_csv_generation:
        comparison_df = generate_comparison_csv(args)
        if comparison_df is None:
            logger.error("CSV generation failed. Aborting.")
            sys.exit(1)
    else:
        logger.info("Skipping CSV generation as per --skip_csv_generation flag.")

    if not args.skip_settings_update:
        if comparison_df is None: # CSV generation was skipped, try to load
            if args.csv_file.is_file():
                logger.info(f"Loading comparison data from existing CSV: {args.csv_file}")
                try:
                    comparison_df = pd.read_csv(args.csv_file)
                except Exception as e:
                    logger.error(f"Failed to load CSV file {args.csv_file}: {e}")
                    sys.exit(1)
            else:
                logger.error(f"CSV file {args.csv_file} not found, and CSV generation was skipped. Cannot proceed with settings update.")
                sys.exit(1)
        
        if comparison_df is not None:
            update_versification_settings(comparison_df, args)
        else:
            # This case should ideally be caught above, but as a safeguard:
            logger.error("No DataFrame available for settings update. Aborting update phase.")
    else:
        logger.info("Skipping settings update as per --skip_settings_update flag.")

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
