import codecs
import textwrap
from datetime import datetime
from glob import iglob
from os import listdir
from pathlib import Path
from typing import Iterator, Tuple
import xml.etree.ElementTree as ET
import re

import yaml
from machine.corpora import ParatextTextCorpus, extract_scripture_corpus
from machine.scripture.verse_ref import VerseRef

vrs_to_num: dict[str, int] = {
    "Original": 1,
    "Septuagint": 2,
    "Vulgate": 3,
    "English": 4,
    "Russian Protestant": 5,
    "Russian Orthodox": 6,
}

BOOK_NUM = r"[0-9].\-"

EXCLUDE_ALPHANUMERICS = r"[^\w]"

POST_PART = r"[a-z].+"

def log_and_print(file, s, type="Info") -> None:
    with open(file, "a") as log:
        log.write(
            f"{type.upper()}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {s}\n"
        )
    print(s)


def add_settings_file(project_folder, language_code):
    versification = get_versification(project_folder)
    setting_file_stub = f"""<ScriptureText>
    <Versification>{versification}</Versification>
    <LanguageIsoCode>{language_code}:::</LanguageIsoCode>
    <Naming BookNameForm="41MAT" PostPart="{project_folder.name}.SFM" PrePart="" />
</ScriptureText>"""

    settings_file = project_folder / "Settings.xml"
    with open(settings_file, "w") as settings:
        settings.write(setting_file_stub)


def get_vrs_diffs() -> dict[str, dict[int, dict[int, list[str]]]]:
    """
    gets the differences in versifications from vrs_diffs.yaml
    return: the versification differences
    """
    with open("assets/vrs_diffs.yaml", "r") as file:
        try:
            vrs_diffs = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return vrs_diffs


def check_vref(
    prev: VerseRef,
    vrs_diffs: dict[str, dict[int, dict[int, list[str]]]],
    versifications: list[str],
    ruled_out: list[str],
) -> tuple[list[str], list[str]]:
    """
    checks a verse reference to try to determine the versification
    param prev: the verse reference to check
    param vrs_diffs: the list of differences in the versifications
    param versifications: the list of possible versifications
    param ruled_out: the list of versifications that have been ruled out
    return: the list of possible versifications and the list of versifications that have been ruled out
    """
    # print(f"  check_vref: Checking {prev} against {len(versifications)} possibilities: {versifications}") # Optional: Very verbose
    try:
        curr = vrs_diffs[prev.book]["last_chapter"]
        key = prev.chapter_num
    except:
        try:
            curr = vrs_diffs[prev.book][prev.chapter_num]
            key = prev.verse_num
        except:
            return versifications, ruled_out
    try:
        curr_versifications = curr[key].copy()
    except:
        return versifications, ruled_out
    if len(curr_versifications) == 1:
        return curr_versifications, ruled_out
    for num, versifs in curr.items():
        if num != key:
            for versif in versifs:
                if not versif in ruled_out:
                    # print(f"    check_vref: Ruling out {versif} (based on other verses in chapter/book)") # Optional
                    ruled_out.append(versif)
    to_remove = []
    for versif in curr_versifications:
        if versif in ruled_out:
            # print(f"    check_vref: Removing {versif} (already ruled out)") # Optional
            to_remove.append(versif)
    for versif in to_remove:
        curr_versifications.remove(versif)
    if curr_versifications and len(curr_versifications) < len(versifications):
        return curr_versifications, ruled_out
    return versifications, ruled_out


def get_book_names(project_folder: Path) -> list[str]:
    """
    gets the book names from the specified bible
    param project_folder: the path to the project folder
    return: a list of book names with their corresponding file names
    """
    names = []
    books = (book.name for book in project_folder.glob("*.SFM"))
    for book in books:
        name = re.sub(BOOK_NUM, "", book)
        name = re.sub(EXCLUDE_ALPHANUMERICS, "", name)
        name = re.sub(POST_PART, "", name)
        names.append((name, book))
    return names


def stream_verse_refs_from_file(usfm_path: Path, book_code: str) -> Iterator[VerseRef]:
    """
    Reads a USFM file line by line and yields VerseRef objects for each verse.

    Args:
        usfm_path: The path to the USFM file.
        book_code: The 3-letter canonical book code (e.g., "GEN", "MAT").

    Yields:
        VerseRef: A VerseRef object for each verse found in the file.
    """
    # Use English versification for parsing verse strings.
    # The specific versification system used for parsing doesn't affect the
    # book/chapter/verse numbers needed by check_vref.
    parser_vrs = Versification.get_instance("English")

    current_chapter = 0
    try:
        # Use codecs.open for robust encoding handling
        with codecs.open(usfm_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                # Check for chapter marker
                chapter_match = re.search(r"\\c\s+(\d+)", line)
                if chapter_match:
                    try:
                        current_chapter = int(chapter_match.group(1))
                    except ValueError:
                        # Handle cases where chapter number isn't a valid int
                        print(f"Warning: Invalid chapter marker in {usfm_path} line {line_num}: {line.strip()}")
                        current_chapter = 0 # Reset chapter until next valid \c
                    continue  # Move to the next line after finding a chapter

                # Check for verse marker only if we have a valid current chapter
                if current_chapter > 0:
                    # Match verse number, potentially handling ranges like 1-2 or segments like 1a
                    # For versification check, we only care about the starting verse number.
                    verse_match = re.search(r"\\v\s+(\d+)", line)
                    if verse_match:
                        try:
                            verse_num = int(verse_match.group(1))
                            # Create and yield the VerseRef object
                            vref = VerseRef.from_string(
                                f"{book_code} {current_chapter}:{verse_num}", parser_vrs
                            )
                            yield vref
                        except ValueError:
                            # Handle cases where verse number isn't a valid int
                            print(f"Warning: Invalid verse marker in {usfm_path} line {line_num}: {line.strip()}")
                        except Exception as e_vref:
                            # Catch potential errors during VerseRef creation
                            print(f"Error creating VerseRef for {book_code} {current_chapter}:{verse_match.group(1)} in {usfm_path}: {e_vref}")

    except FileNotFoundError:
        # Handle case where the file doesn't exist
        print(f"Error: File not found {usfm_path}")
        # pass # Or raise the error, depending on desired behavior
    except Exception as e:
        # Handle other potential file reading errors
        print(f"Error reading file {usfm_path}: {e}")
        # pass # Or raise


def get_versification(
    project_folder: Path,
    vrs_diffs: dict[str, dict[int, dict[int, list[str]]]],
) -> str:
    """
    Gets the versification of the given bible by streaming USFM files directly.
    param project_folder: the path to the project folder
    param vrs_diffs: the list of differences in the versifications.
    return: the versification of the given bible
    """
    default_versification = "English"
    print(f"\n--- get_versification for: {project_folder.name} ---") # Add project identifier

    versifications = list(vrs_to_num.keys())
    ruled_out = []
    processed_first_vref = False
    # Use VerseRef objects if possible, otherwise adapt check_vref
    prev_vref = None
    print(f"  Initial versifications ({len(versifications)}): {versifications}")

    book_files = get_book_names(project_folder)  # List of (canonical_name, filename)
    file_map = {name: fname for name, fname in book_files}

    # Iterate through books relevant to versification diffs found in the project
    for book_code in vrs_diffs.keys():
        if book_code in file_map:
            usfm_path = project_folder / file_map[book_code]
            if usfm_path.is_file():
                # Stream verse references directly from the original file
                for vref in stream_verse_refs_from_file(
                    usfm_path, book_code
                ):  # New helper function
                    if not processed_first_vref:
                        prev_vref = vref
                        processed_first_vref = True
                        continue

                    # Check if chapter changed to trigger check_vref on the *previous* verse
                    if (
                        vref.book != prev_vref.book
                        or vref.chapter_num != prev_vref.chapter_num
                    ):
                        if prev_vref:  # Ensure we have a valid previous verse
                            print(f"  -> Chapter/Book change detected at {vref}. Checking {prev_vref}...")
                            versifications, ruled_out = check_vref(
                                prev_vref, vrs_diffs, versifications, ruled_out
                            )
                            if len(versifications) == 1:
                                print(f"  --> Determined: {versifications[0]} (during loop)")
                                return versifications[
                                    0
                                ]  # Found conclusive versification

                    prev_vref = vref  # Update previous verse ref

                    # Optional: Check current vref immediately if needed by logic?
                    # The original logic checked prev when chapter changed. Let's stick to that.
                    print(f"  Finished book {book_code}. Remaining versifications: {versifications}")
                else:
                    print(f"  Skipping {book_code}: File not found at {usfm_path}")
            # else: # Optional: Log if a book from vrs_diffs isn't in the project
                # print(f"  Skipping {book_code}: Not found in project folder.")
    
    print(f"  Finished all relevant books.")            
    # Final check for the very last verse processed
    if prev_vref:  # Check if any verse was processed at all
        print(f"  -> Final check for last processed verse: {prev_vref}")
        versifications, ruled_out = check_vref(
            prev_vref, vrs_diffs, versifications, ruled_out
        )
    else:
        # This case happens if no relevant books were found or no verses were streamed
        print(f"  Warning: No verse references were processed for {project_folder.name}.")
        # Keep the initial list if nothing was processed, let ambiguity logic handle it.

    if not versifications:
        # Fallback if no verses were processed or logic failed
        print(
            f"Warning: Could not determine versification for {project_folder}, defaulting."
        )
        # Default to English as per the old logic's fallback? Or the first in the list?
        return "English"  # Or list(vrs_to_num.keys())[0]

    return versifications[0]


def write_settings_file(
    project_folder: Path,
    language_code: str,
    vrs_diffs: dict[str, dict[int, dict[int, list[str]]]],
) -> tuple[Path, int, dict, dict]: # Return path, vrs_num, old_settings, new_settings
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
        A tuple containing:
            - Path to the settings file (or None if failed)
            - Inferred versification number (defaulting to 4)
            - Dictionary of old settings values (or defaults if file missing/malformed)
            - Dictionary of new settings values
    """
    default_vrs_num = 4 # Default to English
    settings_file = project_folder / "Settings.xml"
    old_settings = {
        "old_Versification": None,
        "old_LanguageIsoCode": None,
        "old_BookNameForm": None,
        "old_PostPart": None,
        "old_PrePart": None,
    }
    new_settings = {}

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
                print(f"Warning: Could not parse existing {settings_file}. Old values will be None.")
            except Exception as e:
                 print(f"Warning: Error reading existing {settings_file}: {e}. Old values will be None.")

        # --- Determine new settings ---
        try:
            versification_name = get_versification(project_folder, vrs_diffs)
            # Safely get the number, default to default_vrs_num if name not found or invalid
            vrs_num = vrs_to_num.get(versification_name, default_vrs_num)
            if not isinstance(vrs_num, int): # Ensure it's an integer
                 print(f"Warning: Versification lookup for '{versification_name}' returned non-integer {vrs_num}. Defaulting to {default_vrs_num}.")
                 vrs_num = default_vrs_num
        except Exception as e:
            print(f"Error during get_versification for {project_folder}: {e}. Defaulting versification to {default_vrs_num}.")
            vrs_num = default_vrs_num

        # Define new values for reporting
        new_settings = {
            "new_Versification": vrs_num,
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
        return settings_file, vrs_num, old_settings, new_settings
    else:
        # Project folder doesn't exist
        return None, default_vrs_num, old_settings, new_settings # Return None path, default vrs, empty dicts
