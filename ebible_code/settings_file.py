import sys
import codecs
import textwrap
from pathlib import Path
from typing import Iterator, Tuple, Dict, List, Optional, Set
import xml.etree.ElementTree as ET
import logging # Import logging
from collections import defaultdict
from datetime import datetime
import re

from machine.corpora import ParatextTextCorpus
from machine.scripture import Versification, VersificationType, book_id_to_number

#import machine.scripture
#print(f"DEBUG: machine.scripture is loaded from: {machine.scripture.__file__}")
#print(f"DEBUG: sys.path is: {sys.path}")

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info("--- settings_file.py module loaded and logger obtained ---")

# --- Scoring Weights (inspired by update_versifications.py) ---
WEIGHT_BOOK = 0.0  # Weight for matching book presence
WEIGHT_CHAPTER = 0.0  # Weight for matching chapter presence (beyond book)
WEIGHT_VERSE_COUNT = 1.0  # Weight for matching verse counts in differentiating chapters

BOOK_NUM = r"[0-9].\-"
EXCLUDE_ALPHANUMERICS = r"[^\w]"
POST_PART = r"[a-z].+"

# --- Global Dictionaries to be populated ---
LOADED_VRS_OBJECTS: Dict[str, Versification] = {}
VRS_NAME_TO_NUM_STRING: Dict[str, str] = {}
VALID_VRS_NUM_STRINGS: List[str] = []

# --- Constants for VRS file generation (from generate_project_vrs.py) ---
# --- Constants for VRS file generation (from generate_project_vrs.py) ---
BOOK_ORDER = [
    "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT", "1SA", "2SA",
    "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST", "JOB", "PSA", "PRO",
    "ECC", "SNG", "ISA", "JER", "LAM", "EZK", "DAN", "HOS", "JOL", "AMO",
    "OBA", "JON", "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL",
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
    "TOB", "JDT", "ESG", "WIS", "SIR", "BAR", "LJE", "S3Y", "SUS", "BEL",
    "1MA", "2MA", "3MA", "4MA", "1ES", "2ES", "MAN", "PS2", "ODA", "PSS",
    "EZA", "5EZ", "6EZ", "DAG", "LAO", "FRT", "BAK", "OTH", "CNC", "GLO",
    "TDX", "NDX", "XXA", "XXB", "XXC", "XXD", "XXE", "XXF", "XXG"
]
BOOK_SORT_KEY = {book: i for i, book in enumerate(BOOK_ORDER)}

# Regex for parsing verse strings like "1", "1a", "1-2" from corpus VerseRef.verse
VERSE_STR_PATTERN = re.compile(r"(\d+)([a-zA-Z]?)")

def _parse_verse_string_for_vrs(verse_str: str) -> tuple[int, str]:
    """Parses a verse string (e.g., '1', '1a') into number and subdivision for VRS generation."""
    # Simplified parser focusing on the leading integer part for max verse counting.
    if not verse_str: return 0, ""
    match = re.match(r"(\d+)", verse_str) # Match leading digits
    if match:
        return int(match.group(1)), "" # Subdivision not critical for max verse in this context
    try:
        return int(verse_str), "" # Try direct conversion if no complex pattern
    except ValueError:
        logger.warning(f"Could not parse verse string '{verse_str}' into an integer for VRS generation.")
        return 0, ""


def populate_standard_versifications() -> None:
    """
    Loads standard versification files using machine.scripture.VersificationType
    and populates LOADED_VRS_OBJECTS, VRS_NAME_TO_NUM_STRING, and VALID_VRS_NUM_STRINGS.
    """
    global VALID_VRS_NUM_STRINGS # Ensure we modify the global list
    
    if LOADED_VRS_OBJECTS: # Avoid re-populating if already done
        return

    for vtype in VersificationType:
        if vtype == VersificationType.UNKNOWN:
            continue
        try:
            # Versification.get_builtin() loads standard .vrs files
            # distributed with the 'machine' library by name.
            vrs_obj = Versification.get_builtin(vtype) # vrs_obj.name will be "English", "RussianOrthodox", etc.
            if vrs_obj:
                LOADED_VRS_OBJECTS[vrs_obj.name] = vrs_obj
                VRS_NAME_TO_NUM_STRING[vrs_obj.name] = str(vtype.value) # Use enum value
                logger.info(f"Successfully loaded and mapped standard versification: {vrs_obj.name} -> {vtype.value}")
            else:
                logger.warning(f"Could not load standard versification for type: {vtype.name} (returned None)")
        except Exception as e:
            logger.error(f"Error loading standard versification for type {vtype.name}: {e}")

    if not LOADED_VRS_OBJECTS:
        logger.error("No standard versification files were loaded. Scoring will be impaired.")
    
    VALID_VRS_NUM_STRINGS = sorted(list(set(VRS_NAME_TO_NUM_STRING.values())))

# Call at module load time to populate standard versifications
populate_standard_versifications()


def add_settings_file(project_folder: Path, language_code: str) -> None:
    """
    (This function seems to be a simplified version and might not be directly used by ebible.py,
    which calls write_settings_file. However, correcting it for completeness.)
    Creates a minimal Settings.xml file in the project folder using the scoring mechanism.
    """
    versification_name = get_versification_with_scoring(project_folder)
    vrs_num = VRS_NAME_TO_NUM_STRING.get(versification_name, "4") # Default to English "4"

    post_part_val = f"{language_code}.SFM" # Align with write_settings_file

    setting_file_stub = textwrap.dedent(f"""\
        <ScriptureText>
            <Versification>{vrs_num}</Versification>
            <LanguageIsoCode>{language_code}:::</LanguageIsoCode>
            <Naming BookNameForm="41MAT" PostPart="{post_part_val}" PrePart="" />
        </ScriptureText>""")
    setting_file_stub += "\n" # POSIX friendly

    settings_file = project_folder / "Settings.xml"
    with open(settings_file, "w") as settings:
        settings.write(setting_file_stub)


def get_verse_data_from_vrs_obj(vrs_obj: Optional[Versification]) -> Dict[Tuple[str, int], int]:
    """Extracts {(book_id_str, chapter_num_int): max_verse_num_int} from a Versification object."""
    data: Dict[Tuple[str, int], int] = {}
    if not vrs_obj:
        return data # type: ignore
    for book_num_int in range(1, vrs_obj.get_last_book() + 1):
        book_id_str = book_id_to_number(book_num_int) # machine.scripture.book_id_to_number
        if not book_id_str or book_id_str == "UNKNOWN": # Check for valid book ID
            continue
        for chapter_num_int in range(1, vrs_obj.get_last_chapter(book_num_int) + 1):
            max_verse = vrs_obj.get_last_verse(book_num_int, chapter_num_int)
            if max_verse > 0: # Only include chapters with actual verses
                data[(book_id_str, chapter_num_int)] = max_verse
    return data

def calculate_similarity_score_for_settings(
    project_v_data: Dict[Tuple[str, int], int],
    standard_v_data: Dict[Tuple[str, int], int],
    invariant_chapters: Set[Tuple[str, int]]
    ) -> float:
    """Calculates the similarity score. Adapted from update_versifications.py."""
    project_books_overall = {book for book, chap in project_v_data}
    if not project_books_overall:
        return 0.0

    standard_books_defined_overall = {book for book, chap in standard_v_data}
    common_books = project_books_overall.intersection(standard_books_defined_overall)
    book_score = len(common_books) / len(project_books_overall) if project_books_overall else 0.0

    project_bc_for_detailed_comparison = {(b, c) for (b, c) in project_v_data.keys() if b in common_books and (b, c) not in invariant_chapters}
    standard_bc_for_detailed_comparison = {(b, c) for (b, c) in standard_v_data.keys() if b in common_books and (b, c) not in invariant_chapters}
    common_differentiating_chapters = project_bc_for_detailed_comparison.intersection(standard_bc_for_detailed_comparison)
    
    num_project_differentiating_chapters = len(project_bc_for_detailed_comparison)
    chapter_score = len(common_differentiating_chapters) / num_project_differentiating_chapters if num_project_differentiating_chapters else 0.0

    matching_verse_count_differentiating_chapters = 0
    for book, chap in common_differentiating_chapters:
        if project_v_data.get((book, chap)) == standard_v_data.get((book, chap)):
            matching_verse_count_differentiating_chapters += 1
    
    num_common_differentiating_chapters_for_verse_score = len(common_differentiating_chapters)
    verse_count_score = matching_verse_count_differentiating_chapters / num_common_differentiating_chapters_for_verse_score if num_common_differentiating_chapters_for_verse_score else 0.0
    
    total_score = (WEIGHT_BOOK * book_score) + (WEIGHT_CHAPTER * chapter_score) + (WEIGHT_VERSE_COUNT * verse_count_score)
    logger.debug(f"      Similarity score components for {project_v_data.keys()}: Book={book_score:.2f}, Chap={chapter_score:.2f}, Verse={verse_count_score:.2f} -> Total={total_score:.2f}")
    return total_score

# Remove the dummy version of get_versification_with_scoring
# The correct implementation that returns a string name follows.
def get_versification_with_scoring(project_folder: Path) -> str:
    """ Determines the versification for a project by scoring its generated .vrs file
    against standard versifications.
    """
    default_versification_name = "English" # Fallback
    logger.info(f"Get_versification_with_scoring for: {project_folder.name}")

    project_vrs_filename = f"{project_folder.name}.vrs" # Assuming .vrs file is named after the folder
    project_vrs_path = project_folder / project_vrs_filename
    project_vrs_obj: Optional[Versification] = None

    if not project_vrs_path.is_file():
        logger.warning(f"  Project VRS file not found: {project_vrs_path}. Cannot use scoring. Defaulting to '{default_versification_name}'.")
        return default_versification_name
    
    try:
        project_vrs_obj = Versification.load(project_vrs_path, fallback_name=project_folder.name)
        logger.info(f"  Successfully loaded project VRS file: {project_vrs_path}")
    except Exception as e:
        logger.error(f"  Error loading project VRS file {project_vrs_path}: {e}. Defaulting to '{default_versification_name}'.")
        return default_versification_name

    if not project_vrs_obj: # Should be caught by above, but as a safeguard
        return default_versification_name

    project_verse_data = get_verse_data_from_vrs_obj(project_vrs_obj)
    if not project_verse_data:
        logger.warning(f"  No verse data extracted from project VRS {project_vrs_filename}. Defaulting to '{default_versification_name}'.")
        return default_versification_name

    if not LOADED_VRS_OBJECTS:
        logger.error("  No standard versifications (LOADED_VRS_OBJECTS) available for scoring. Defaulting.")
        return default_versification_name

    standard_vrs_data_map: Dict[str, Dict[Tuple[str, int], int]] = {
        name: get_verse_data_from_vrs_obj(obj) for name, obj in LOADED_VRS_OBJECTS.items()
    }

    all_chapters_in_standards: Set[Tuple[str, int]] = set().union(*(std_data.keys() for std_data in standard_vrs_data_map.values()))
    invariant_chapters: Set[Tuple[str, int]] = set()
    if all_chapters_in_standards:
        for book_chap_tuple in all_chapters_in_standards:
            verse_counts_for_chapter = {std_data[book_chap_tuple] for std_data in standard_vrs_data_map.values() if book_chap_tuple in std_data}
            if len(verse_counts_for_chapter) <= 1: # All standards that define it, agree on verse count
                invariant_chapters.add(book_chap_tuple)
        logger.info(f"  Identified {len(invariant_chapters)} invariant chapters among {len(all_chapters_in_standards)} unique standard chapters.")
    else:
        logger.warning("  No chapters found in any standard versifications for invariant chapter check.")

    best_score = -1.0
    best_std_vrs_name = default_versification_name 

    candidate_standard_names = list(LOADED_VRS_OBJECTS.keys())
    if default_versification_name not in candidate_standard_names and default_versification_name in LOADED_VRS_OBJECTS:
         candidate_standard_names.append(default_versification_name) # Ensure English is scored

    for std_name in candidate_standard_names:
        standard_verse_data = standard_vrs_data_map.get(std_name)
        if not standard_verse_data:
            logger.debug(f"  Skipping scoring against {std_name}, no verse data loaded for it.")
            continue
        
        current_score = calculate_similarity_score_for_settings(project_verse_data, standard_verse_data, invariant_chapters)
        logger.info(f"    Score for {project_folder.name} vs {std_name}: {current_score:.4f}")

        if current_score > best_score:
            best_score = current_score
            best_std_vrs_name = std_name
        elif current_score == best_score and std_name == "English": # Tie-breaking preference for English
            best_std_vrs_name = "English"
            logger.info(f"    Tie score with {best_std_vrs_name}, preferring English.")

    logger.info(f"  Best match for {project_folder.name}: {best_std_vrs_name} with score: {best_score:.4f}")
    return best_std_vrs_name


def write_settings_file(
    project_folder: Path,
    language_code: str,
) -> tuple[Optional[Path], str, dict, dict]: # Return path, vrs_num_string, old_settings, new_settings
    """
    Write a Settings.xml file to the project folder and overwrite any existing one.
    The file is very minimal containing only:
      <Versification> (which is inferred from the project)
      <LanguageIsoCode> (which is the first 3 characters of the folder name)
      <Naming> (using BookNameForm="41MAT" and PostPart="{language_code}.SFM")
      <Naming> (which is the naming convention "MAT" indicating no digits prior to the 3 letter book code)
      <FileNamePrePart> (which is the language code)

    When a settings file is created, the path to it is returned.
    Otherwise None is returned.

    Note that the "Naming->PostPart" section will use {language_code}.SFM
    Returns:
        A tuple containing the path to the settings file and the inferred versification number.
    """
    default_vrs_num_string = "4" # Default to English
    settings_file = project_folder / "Settings.xml"
    # Initialize old_settings with specific keys and None values
    old_settings = {
        "old_Versification": None,
        "old_LanguageIsoCode": None,
        "old_BookNameForm": None,
        "old_PostPart": None,
        "old_PrePart": None,
    }
    new_settings: Dict[str, Optional[str]] = {} # Ensure new_settings is typed

    # Add a Settings.xml file to a project folder.
    if project_folder.is_dir():
        # --- Read existing settings ---
        if settings_file.exists():
            try:
                tree = ET.parse(settings_file)
                root = tree.getroot()
                old_settings["old_Versification"] = root.findtext("Versification")
                old_settings["old_LanguageIsoCode"] = root.findtext("LanguageIsoCode")
                naming = root.find("Naming")
                if naming is not None:
                    old_settings["old_BookNameForm"] = naming.get("BookNameForm")
                    old_settings["old_PostPart"] = naming.get("PostPart")
                    old_settings["old_PrePart"] = naming.get("PrePart")
            except ET.ParseError:
                logger.warning(f"Could not parse existing {settings_file}. Old values will be None.")
            except Exception as e:
                 logger.warning(f"Error reading existing {settings_file}: {e}. Old values will be None.")

        # --- Determine new settings ---
        versification_name = get_versification_with_scoring(project_folder)
        vrs_num_string = VRS_NAME_TO_NUM_STRING.get(versification_name, default_vrs_num_string)
        if vrs_num_string not in VALID_VRS_NUM_STRINGS:
            raise ValueError(f"Invalid versification: {vrs_num_string}")

        # Define new values for reporting
        new_settings = {
            "new_Versification": vrs_num_string,
            "new_LanguageIsoCode": f"{language_code}:::",
            "new_BookNameForm": "41MAT", # Consistent naming scheme
            "new_PostPart": f"{language_code}.SFM",
            "new_PrePart": "",
        }

        # --- Write new settings file ---
        setting_file_text = textwrap.dedent(
            f"""\
            <ScriptureText>
                <Versification>{new_settings['new_Versification']}</Versification>
                <LanguageIsoCode>{new_settings['new_LanguageIsoCode']}</LanguageIsoCode>
                <Naming BookNameForm="{new_settings['new_BookNameForm']}" PostPart="{new_settings['new_PostPart']}" PrePart="{new_settings['new_PrePart']}" />
            </ScriptureText>"""
        )
        # Optional: Add a newline at the end if desired for POSIX compatibility
        setting_file_text += "\n"

        with open(settings_file, "w") as settings:
            settings.write(setting_file_text)
        return settings_file, vrs_num_string, old_settings, new_settings
    else:
        # Project folder doesn't exist
        return None, default_vrs_num_string, old_settings, new_settings # Return None path, default vrs, empty dicts


def generate_vrs_from_project(project_path: Path) -> Optional[Path]:
    """
    Generates a .vrs file from the actual content of a Paratext project.
    The .vrs file is named after the project folder and saved within it.
    Returns the path to the generated .vrs file, or None if generation failed.
    """
    project_name = project_path.name
    logger.info(f"Generating project .vrs for: {project_name} at {project_path}")

    if not project_path.is_dir():
        logger.error(f"Project path does not exist or is not a directory: {project_path}")
        return None

    try:
        # ParatextTextCorpus expects a string path
        corpus = ParatextTextCorpus(str(project_path))
    except Exception as e:
        logger.error(f"Could not initialize ParatextTextCorpus for {project_name}: {e}")
        return None

    # Structure: {book_id: {chapter_num: max_verse_num}}
    verse_data = defaultdict(lambda: defaultdict(int))
    processed_verse_refs_count = 0

    for text_row in corpus: # Iterate directly over the corpus
        processed_verse_refs_count += 1
        verse_ref = text_row.ref # Get the VerseRef from the TextRow
        
        book_id = verse_ref.book.upper() # Ensure uppercase book ID
        chapter_num = int(verse_ref.chapter) # Convert chapter to int
        
        # verse_ref.verse can be like '1', '1a', '1-2'. We need the primary numeric part.
        # For '1-2', parse_verse_string('1-2') might fail if not handled.
        # ParatextTextCorpus usually yields individual verse refs, so '1-2' as verse_ref.verse is less common.
        # We take the part before any hyphen (for ranges) and then parse the number.
        verse_num_str_to_parse = verse_ref.verse.split('-')[0] 
        verse_num, _ = _parse_verse_string_for_vrs(verse_num_str_to_parse)

        if chapter_num > 0 and verse_num > 0: # Ensure valid chapter and verse numbers
            verse_data[book_id][chapter_num] = max(
                verse_data[book_id][chapter_num], verse_num
            )
        elif chapter_num <= 0:
            logger.warning(f"Skipping verse_ref with invalid chapter {chapter_num} in {book_id} for {project_name}")

    if processed_verse_refs_count == 0:
        logger.warning(f"No verse references found in project {project_name}. Skipping VRS generation.")
        return None
    if not verse_data:
        logger.warning(f"No valid verse data collected for project {project_name} after processing {processed_verse_refs_count} refs. Skipping VRS generation.")
        return None

    # --- Generate .vrs file content ---
    vrs_lines = [
        f"# Versification for project: {project_name}",
        f"# Generated on: {datetime.now().isoformat()}",
        "#",
        "# List of books, chapters, verses. One line per book.",
        "# One entry for each chapter. Verse number is the maximum verse number for that chapter.",
        "#-----------------------------------------------------------"
    ]

    sorted_books = sorted(verse_data.keys(), key=lambda b: (BOOK_SORT_KEY.get(b, float('inf')), b))

    for book_id in sorted_books:
        chapters = verse_data[book_id]
        if not chapters: continue
        chapter_strings = [f"{ch_num}:{max_v}" for ch_num, max_v in sorted(chapters.items()) if max_v > 0]
        if chapter_strings:
            vrs_lines.append(f"{book_id} {' '.join(chapter_strings)}")

    vrs_filepath = project_path / f"{project_name}.vrs"
    try:
        with open(vrs_filepath, "w", encoding="utf-8") as f:
            for line in vrs_lines:
                f.write(line + "\n")
        logger.info(f"Successfully generated VRS file: {vrs_filepath}")
        return vrs_filepath
    except IOError as e:
        logger.error(f"Could not write VRS file {vrs_filepath}: {e}")
        return None
