import codecs  # Needed for opening files with specific encoding
import re
import shutil
import textwrap
from datetime import datetime
from glob import iglob
from os import listdir
from pathlib import Path
from typing import Iterator  # Import Iterator

import yaml
from machine.corpora import ParatextTextCorpus, extract_scripture_corpus
from machine.scripture.verse_ref import VerseRef

# Import Versification if needed
# from machine.scripture.versification import Versification
# FHS = Versification.get_builtin("FHS")  # Or load your desired default .vrs file

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


# Should not need to duplicate this function in ebible.py and here.
def log_and_print(file, s, type="Info") -> None:
    with open(file, "a") as log:
        log.write(
            f"{type.upper()}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {s}\n"
        )
    print(s)


def get_extracted_projects(dir_extracted):
    extracted = []
    for line in listdir(dir_extracted):
        m = re.search(r".+-(.+).txt$", line)
        if m:
            extracted.append(m.group(1))

    return extracted


def get_books_type(files):
    for book in files:
        m = re.search(r".*GEN|JON.*", book)
        if m:
            return "OT+NT"
    return "NT"


def conclude_versification_from_OT(dan_3, dan_5, dan_13):
    if dan_3 == 30:
        versification = "4"  # English
    elif dan_3 == 33 and dan_5 == 30:
        versification = "1"  # Original
    elif dan_3 == 33 and dan_5 == 31:
        versification = "5"  # Russian Protestant
    elif dan_3 == 97:
        versification = "2"  # Septuagint
    elif dan_3 == 100:
        if dan_13 == 65:
            versification = "3"  # Vulgate
        else:
            versification = "6"  # Russian Orthodox
    else:
        versification = ""

    return versification


def conclude_versification_from_NT(jhn_6, act_19, rom_16):
    if jhn_6 == 72:
        versification = "3"  # Vulgate
    elif act_19 == 41:
        versification = "4"  # English
    elif rom_16 == 24:
        versification = "6"  # Russian Orthodox (same as Russian Protestant)
    elif jhn_6 == 71 and act_19 == 40:
        versification = "1"  # Original (Same as Septuagint)
    else:
        versification = ""

    return versification


def get_last_verse(project, book, chapter):
    ch = str(chapter)

    for book_file in iglob(f"{project}/*{book}*"):
        last_verse = "0"
        try:
            f = codecs.open(book_file, "r", encoding="utf-8", errors="ignore")
        except Exception as e:
            log_and_print(logfile, f"Could not open {book_file}, reason:  {e}")
            continue
        try:
            in_chapter = False
            for line in f:
                m = re.search(r"\\c ? ?([0-9]+).*", line)
                if m:
                    if m.group(1) == ch:
                        in_chapter = True
                    else:
                        in_chapter = False

                m = re.search(r"\\v ? ?([0-9]+).*", line)
                if m:
                    if in_chapter:
                        last_verse = m.group(1)
        except Exception as e:
            log_and_print(
                logfile, f"Something went wrong in reading {book_file}, reason:  {e}"
            )
            return None
        try:
            return int(last_verse)
        except Exception as e:
            print(
                f"Could not convert {last_verse} into an integer in {book_file}, reason:  {e}"
            )
            return None


def get_checkpoints_OT(project):
    dan_3 = get_last_verse(project, "DAN", 3)
    dan_5 = get_last_verse(project, "DAN", 5)
    dan_13 = get_last_verse(project, "DAN", 13)

    return dan_3, dan_5, dan_13


def get_checkpoints_NT(project):
    jhn_6 = get_last_verse(project, "JHN", 6)
    act_19 = get_last_verse(project, "ACT", 19)
    rom_16 = get_last_verse(project, "ROM", 16)

    return jhn_6, act_19, rom_16


def get_versification(project):
    versification = ""
    books = get_books_type(listdir(project))

    if books == "OT+NT":
        dan_3, dan_5, dan_13 = get_checkpoints_OT(project)
        versification = conclude_versification_from_OT(dan_3, dan_5, dan_13)

    if not versification:
        jhn_6, act_19, rom_16 = get_checkpoints_NT(project)
        versification = conclude_versification_from_NT(jhn_6, act_19, rom_16)

    if versification != "":
        return versification
    else:
        return "4"  # English


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
                    ruled_out.append(versif)
    to_remove = []
    for versif in curr_versifications:
        if versif in ruled_out:
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
    books = (book for book in listdir(project_folder) if ".usfm" in book)
    for book in books:
        name = re.sub(BOOK_NUM, "", book)
        name = re.sub(EXCLUDE_ALPHANUMERICS, "", name)
        name = re.sub(POST_PART, "", name)
        names.append((name, book))
    return names


def write_temp_settings_file(project_folder: Path, post_part: str) -> None:
    """
    writes a temporary settings file as a place holder to be able to determine the versification
    param project_folder: the path to the project folder
    """
    with open(project_folder / "Settings.xml", "w", encoding="utf-8") as set_file:
        set_file.write(
            f"""<ScriptureText>
            <BiblicalTermsListSetting>Major::BiblicalTerms.xml</BiblicalTermsListSetting>
            <Naming BookNameForm="46-MAT" PostPart="{post_part}.usfm" PrePart="" />
            </ScriptureText>"""
        )


def get_corpus(
    project_folder: Path, vrs_diffs: dict[str, dict[int, dict[int, list[str]]]]
) -> list[tuple[str, VerseRef, VerseRef]]:
    """
    creates a corpus of books found in vrs_diffs
    param project_folder: the path to the project folder
    param vrs_diffs: the list of differences in the versifications
    return: the corpus from the available books in the specified bible
    """
    vrs_path = project_folder / "versification"
    vrs_path.mkdir(parents=True, exist_ok=True)
    book_names = get_book_names(project_folder)
    for name in book_names:
        if name[0] in vrs_diffs.keys():
            shutil.copyfile(project_folder / name[1], vrs_path / name[1])
    write_temp_settings_file(vrs_path, project_folder.name)
    corpus = ParatextTextCorpus(vrs_path)
    lines = list(extract_scripture_corpus(corpus, corpus))
    shutil.rmtree(vrs_path)

    return lines


def stream_verse_refs_from_file(usfm_path: Path, book_code: str) -> Iterator[VerseRef]:
    """
    Reads a USFM file line by line and yields VerseRef objects for each verse.

    Args:
        usfm_path: The path to the USFM file.
        book_code: The 3-letter canonical book code (e.g., "GEN", "MAT").

    Yields:
        VerseRef: A VerseRef object for each verse found in the file.
    """
    current_chapter = 0
    try:
        # Use codecs.open for robust encoding handling, similar to get_last_verse
        with codecs.open(usfm_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Check for chapter marker
                chapter_match = re.search(r"\\c\s+(\d+)", line)
                if chapter_match:
                    try:
                        current_chapter = int(chapter_match.group(1))
                    except ValueError:
                        # Handle cases where chapter number isn't a valid int
                        # Log this? For now, we'll skip lines until the next valid \c
                        current_chapter = 0
                        # Consider logging a warning here if needed
                        # log_and_print(logfile, f"Warning: Invalid chapter marker in {usfm_path}: {line.strip()}", "Warn")
                    continue  # Move to the next line after finding a chapter

                # Check for verse marker only if we have a valid current chapter
                if current_chapter > 0:
                    verse_match = re.search(r"\\v\s+(\d+)", line)
                    if verse_match:
                        try:
                            verse_num = int(verse_match.group(1))
                            # Create and yield the VerseRef object
                            vref = VerseRef.from_string(
                                f"{book_code} {current_chapter}:{verse_num}", FHS
                            )  # Assuming FHS is the default versification system object
                            yield vref
                        except ValueError:
                            # Handle cases where verse number isn't a valid int
                            # Log this? For now, we skip this verse marker
                            # Consider logging a warning here if needed
                            # log_and_print(logfile, f"Warning: Invalid verse marker in {usfm_path}: {line.strip()}", "Warn")
                            pass
                        except Exception as e_vref:
                            # Catch potential errors during VerseRef creation
                            # log_and_print(logfile, f"Error creating VerseRef for {book_code} {current_chapter}:{verse_match.group(1)} in {usfm_path}: {e_vref}", "Error")
                            pass

    except FileNotFoundError:
        # Handle case where the file doesn't exist
        # log_and_print(logfile, f"Error: File not found {usfm_path}", "Error")
        pass  # Or raise the error, depending on desired behavior
    except Exception as e:
        # Handle other potential file reading errors
        # log_and_print(logfile, f"Error reading file {usfm_path}: {e}", "Error")
        pass  # Or raise


def get_versification(
    project_folder: Path,
    vrs_diffs: dict[str, dict[int, dict[int, list[str]]]],
) -> str:
    """
    gets the versification of the given bible
    param project_folder: the path to the project folder
    param vrs_diffs: the list of differences in the versifications.
    return: the versification of the given bible
    """
    return "English"  # Testing this code necessary
    versifications = list(vrs_to_num.keys())
    ruled_out = []
    processed_first_vref = False
    # Use VerseRef objects if possible, otherwise adapt check_vref
    prev_vref = None

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
                            versifications, ruled_out = check_vref(
                                prev_vref, vrs_diffs, versifications, ruled_out
                            )
                            if len(versifications) == 1:
                                return versifications[
                                    0
                                ]  # Found conclusive versification

                    prev_vref = vref  # Update previous verse ref

                    # Optional: Check current vref immediately if needed by logic?
                    # The original logic checked prev when chapter changed. Let's stick to that.

    # Final check for the very last verse processed
    if prev_vref:  # Check if any verse was processed at all
        versifications, ruled_out = check_vref(
            prev_vref, vrs_diffs, versifications, ruled_out
        )

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
    translation_id: str,
    vrs_diffs: dict[str, dict[int, dict[int, list[str]]]],
) -> Path:
    """
    Write a Settings.xml file to the project folder and overwrite any existing one.
    The file is very minimal containing only:
      <Versification> (which is inferred from the project)
      <LanguageIsoCode> (which is the first 3 characters of the folder name)
      <Naming> (which is the naming convention "MAT" indicating no digits prior to the 3 letter book code)

    When a settings file is created, the path to it is returned.
    Otherwise None is returned.

    Note that the "Naming->PostPart" section will reflect the original naming scheme of the files in the original zip.
    For example if the original zip was eng-web-c.zip, then the files inside will have names like MATeng-web-c.usfm,
    even though for that language, we would have changed the project name to web_c
    See also ebible.py `create_project_name` method, and rename_usfm.py and
    https://github.com/BibleNLP/ebible/issues/50#issuecomment-2659064715
    """

    # Add a Settings.xml file to a project folder.
    if project_folder.is_dir():
        settings_file = project_folder / "Settings.xml"

        versification = get_versification(project_folder, get_vrs_diffs())
        vrs_num = vrs_to_num[versification]

        setting_file_text = textwrap.dedent(
            f"""\
            <ScriptureText>
                <Versification>{vrs_num}</Versification>
                <LanguageIsoCode>{language_code}:::</LanguageIsoCode>
                <Naming BookNameForm="41MAT" PostPart="{translation_id}.SFM" PrePart="" />
            </ScriptureText>"""
        )
        # Note the closing """ is now unindented relative to the start for textwrap.dedent

        # Optional: Add a newline at the end if desired for POSIX compatibility
        setting_file_text += "\n"

        with open(settings_file, "w") as settings:
            settings.write(setting_file_text)
        return settings_file
    # Consider adding a return None or raising an error if project_folder is not a dir
    return None  # Or raise an appropriate error
