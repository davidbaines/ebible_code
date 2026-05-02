import os
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
from machine.scripture import Versification, VersificationType, book_id_to_number, book_number_to_id

#import machine.scripture
#print(f"DEBUG: machine.scripture is loaded from: {machine.scripture.__file__}")
#print(f"DEBUG: sys.path is: {sys.path}")

# Get logger for this module
logger = logging.getLogger(__name__)
logger.debug("--- settings_file.py module loaded and logger obtained ---")

BOOK_NUM = r"[0-9].\-"
EXCLUDE_ALPHANUMERICS = r"[^\w]"
POST_PART = r"[a-z].+"

# --- Global Dictionaries to be populated ---
LOADED_VRS_OBJECTS: Dict[str, Versification] = {}
VALID_VRS_NUM_STRINGS: List[str] = []
STANDARD_VERSE_DATA: Dict[VersificationType, Dict[Tuple[str, int], int]] = {}
INVARIANT_CHAPTERS: Set[Tuple[str, int]] = set()

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
    and populates LOADED_VRS_OBJECTS, VRS_NAME_TO_NUM_STRING, VALID_VRS_NUM_STRINGS,
    STANDARD_VERSE_DATA, and INVARIANT_CHAPTERS.
    """
    global VALID_VRS_NUM_STRINGS

    if LOADED_VRS_OBJECTS:
        return

    for vtype in VersificationType:
        if vtype == VersificationType.UNKNOWN:
            continue
        try:
            vrs_obj = Versification.get_builtin(vtype)
            if vrs_obj:
                LOADED_VRS_OBJECTS[vrs_obj.name] = vrs_obj
                STANDARD_VERSE_DATA[vtype] = get_verse_data_from_vrs_obj(vrs_obj)
                logger.debug(f"Loaded standard versification: {vrs_obj.name} -> {vtype.value}")
            else:
                logger.warning(f"Could not load standard versification for type: {vtype.name} (returned None)")
        except Exception as e:
            logger.error(f"Error loading standard versification for type {vtype.name}: {e}")

    if not LOADED_VRS_OBJECTS:
        logger.error("No standard versification files were loaded. Scoring will be impaired.")

    VALID_VRS_NUM_STRINGS = sorted(str(vt.value) for vt in VersificationType if vt != VersificationType.UNKNOWN)

    # Invariant: present in ALL 6 standards AND all agree on verse count.
    # DC-only chapters (absent from some standards) are NOT invariant.
    num_standards = len(STANDARD_VERSE_DATA)
    if num_standards > 0:
        all_chapters: Set[Tuple[str, int]] = set().union(*(data.keys() for data in STANDARD_VERSE_DATA.values()))
        for bc in all_chapters:
            verse_counts = [STANDARD_VERSE_DATA[vt].get(bc) for vt in STANDARD_VERSE_DATA]
            if all(v is not None for v in verse_counts) and len(set(verse_counts)) == 1:
                INVARIANT_CHAPTERS.add(bc)
        logger.info(f"Computed {len(INVARIANT_CHAPTERS)} invariant chapters from {num_standards} standard versifications.")




def get_verse_data_from_vrs_obj(vrs_obj: Optional[Versification]) -> Dict[Tuple[str, int], int]:
    """Extracts {(book_id_str, chapter_num_int): max_verse_num_int} from a Versification object."""
    data: Dict[Tuple[str, int], int] = {}
    #print(f"DEBUG: vrs_obj is {vrs_obj}" , vrs_obj type is {type(vrs_obj)}")
    #print(f"DEBUG: vrs_obj.get_last_book() is {vrs_obj.get_last_book() if vrs_obj else 'None'}")
    #exit()
    if not vrs_obj:
        return data # type: ignore
    for book_num_int in range(1, vrs_obj.get_last_book() + 1):
        book_id_str = book_number_to_id(book_num_int)
        #print(f"DEBUG: book_num_int={book_num_int}, book_id_str={book_id_str}")
        if not book_id_str or book_id_str == "UNKNOWN": # Check for valid book ID
            continue
        for chapter_num_int in range(1, vrs_obj.get_last_chapter(book_num_int) + 1):
            max_verse = vrs_obj.get_last_verse(book_num_int, chapter_num_int)
            if max_verse > 0: # Only include chapters with actual verses
                data[(book_id_str, chapter_num_int)] = max_verse
    return data

def compute_versification_scores(
    project_verse_data: Dict[Tuple[str, int], int]
) -> dict:
    """
    Scores a project's verse data against all standard versifications.

    Compares every non-spurious project chapter against each standard.
    Scores are percentages (0.0–100.0): matching_chapters / total_project_chapters * 100.

    Returns a dict with:
      'scores': Dict[VersificationType, float]  — percentage 0.0–100.0
      'mismatch_counts': Dict[VersificationType, int]  — ALL-chapter mismatches per standard
      'project_differentiating_chapters': Set[Tuple[str, int]]  — for reporting only
      'total_project_chapters': int
    """
    # Exclude spurious Versification.load() placeholder chapters: entries where the project
    # has max_verse=1 but at least one standard has more. These arise because Versification.load()
    # fills placeholder (chapter 1, verse 1) for all 66 canonical books not in the .vrs file.
    spurious_chapters: Set[Tuple[str, int]] = {
        bc for bc in project_verse_data
        if project_verse_data[bc] == 1
        and any(std.get(bc, 0) > 1 for std in STANDARD_VERSE_DATA.values())
    }
    # Valid (non-spurious) chapters used for all scoring.
    valid_chapters: Dict[Tuple[str, int], int] = {
        bc: v for bc, v in project_verse_data.items() if bc not in spurious_chapters
    }
    total_project_chapters = len(valid_chapters)
    # Differentiating chapters: non-invariant valid chapters (retained for reporting only).
    project_differentiating_chapters: Set[Tuple[str, int]] = {
        bc for bc in valid_chapters if bc not in INVARIANT_CHAPTERS
    }

    scores: Dict[VersificationType, float] = {}
    mismatch_counts: Dict[VersificationType, int] = {}

    for vt, std_data in STANDARD_VERSE_DATA.items():
        if total_project_chapters == 0:
            scores[vt] = 0.0
            mismatch_counts[vt] = 0
        else:
            mismatches = sum(
                1 for bc, v in valid_chapters.items()
                if v != std_data.get(bc)
            )
            scores[vt] = ((total_project_chapters - mismatches) / total_project_chapters) * 100
            mismatch_counts[vt] = mismatches

    return {
        'scores': scores,
        'mismatch_counts': mismatch_counts,
        'project_differentiating_chapters': project_differentiating_chapters,
        'total_project_chapters': total_project_chapters,
    }


# Tie-break preference order when multiple versifications share the top score.
# Ordered by frequency in eBible corpus: most common first.
_VT_PREFERENCE_ORDER = [
    VersificationType.ENGLISH,
    VersificationType.ORIGINAL,
    VersificationType.RUSSIAN_PROTESTANT,
    VersificationType.RUSSIAN_ORTHODOX,
    VersificationType.SEPTUAGINT,
    VersificationType.VULGATE,
]


def estimate_versification(project_path: Path) -> VersificationType:
    """Estimates the versification type for a project by scoring its .vrs file against all standards.

    Returns a concrete VersificationType (never UNKNOWN). When the best score is below
    VERSIFICATION_MATCH_THRESHOLD, returns ENGLISH as a safe fallback; the caller can detect
    this via describe_versification_match() which sets status='unknown'.
    """
    project_name = project_path.name
    project_vrs_path = project_path / f"{project_name}.vrs"

    if not project_vrs_path.is_file():
        logger.warning(f"VRS file not found: {project_vrs_path}. Defaulting to ENGLISH.")
        return VersificationType.ENGLISH

    try:
        project_vrs_obj = Versification.load(project_vrs_path, fallback_name=project_name)
    except Exception as e:
        logger.warning(f"Could not load VRS file {project_vrs_path}: {e}. Defaulting to ENGLISH.")
        return VersificationType.ENGLISH

    if not project_vrs_obj:
        return VersificationType.ENGLISH

    project_verse_data = get_verse_data_from_vrs_obj(project_vrs_obj)
    if not project_verse_data:
        logger.warning(f"No verse data in {project_vrs_path}. Defaulting to ENGLISH.")
        return VersificationType.ENGLISH

    result = compute_versification_scores(project_verse_data)
    scores: Dict[VersificationType, float] = result['scores']

    if not scores:
        return VersificationType.ENGLISH

    try:
        threshold = float(os.environ.get('VERSIFICATION_MATCH_THRESHOLD', '0.0'))
    except ValueError:
        threshold = 0.0

    best_score = max(scores.values())

    if best_score < threshold:
        logger.info(f"{project_name}: Best score {best_score:.1f}% < threshold {threshold:.1f}%. Returning ENGLISH (unknown fallback).")
        return VersificationType.ENGLISH

    best_types = {vt for vt, s in scores.items() if s == best_score}
    for vt in _VT_PREFERENCE_ORDER:
        if vt in best_types:
            return vt
    return next(iter(best_types))


def write_settings_file(
    project_folder: Path,
    language_code: str,
    versification: VersificationType,
    language_name_in_english: str = "",  # from CSV column: languageNameInEnglish
    full_name: str = "",                 # from CSV column: title
) -> bool:
    """Write Settings.xml to project_folder. Returns True on success, False on failure."""
    if not project_folder.is_dir():
        logger.warning(f"Project folder does not exist: {project_folder}")
        return False

    # PostPart must match the prefix used by rename_usfm (folder_name[:3]).
    file_prefix = project_folder.name[:3]
    settings_file = project_folder / "Settings.xml"

    setting_file_text = textwrap.dedent(
        f"""\
        <ScriptureText>
            <Language>{language_name_in_english}</Language>
            <Encoding>65001</Encoding>
            <FullName>{full_name}</FullName>
            <Name>{project_folder.name}</Name>
            <Versification>{versification.value}</Versification>
            <LanguageIsoCode>{language_code}:::</LanguageIsoCode>
            <Naming BookNameForm="41MAT" PostPart="{file_prefix}.SFM" PrePart="" />
        </ScriptureText>"""
    )
    setting_file_text += "\n"

    try:
        with open(settings_file, "w", encoding="utf-8") as f:
            f.write(setting_file_text)
        return True
    except OSError as e:
        logger.error(f"Could not write Settings.xml for {project_folder.name}: {e}")
        return False


def generate_vrs_from_project(project_path: Path) -> Optional[Path]:
    """
    Generates a .vrs file from the actual content of a Paratext project.
    The .vrs file is named after the project folder and saved within it.
    Returns the path to the generated .vrs file, or None if generation failed.
    """
    project_name = project_path.name
    logger.info(f"Attempting to generate project .vrs for: {project_name} at {project_path}")

    if not project_path.is_dir():
        logger.error(f"Project path does not exist or is not a directory: {project_path}")
        return None

    try:
        # Use UsfmFileTextCorpus which does not strictly require Settings.xml.
        # Provide a default versification (English) for parsing purposes, as the project's
        # true versification is what we are trying to determine.
        from machine.corpora import UsfmFileTextCorpus # Import UsfmFileTextCorpus
        default_vrs_for_parsing = Versification.get_builtin(VersificationType.ENGLISH)

        # Since all the projects are from eBible.org, they all have the same ".SFM" extension.
        corpus = UsfmFileTextCorpus(str(project_path), file_pattern="*.SFM", versification=default_vrs_for_parsing)
        
    except Exception as e: # Catching generic Exception here might be too broad; consider more specific exceptions from ParatextTextCorpus
        logger.error(f"Could not initialize UsfmFileTextCorpus for {project_name}: {e}")
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
    
    logger.debug(f"Processed {processed_verse_refs_count} verse references for {project_name}.")

    if processed_verse_refs_count == 0:
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


# Populate standard versifications at module load time (after all functions are defined).
populate_standard_versifications()
