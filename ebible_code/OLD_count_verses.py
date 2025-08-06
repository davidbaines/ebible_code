# filename: parse_sfm_verse_counts.py

import csv
import re
import sys
from pathlib import Path
from collections import defaultdict
import types # Added for introspection helper
import argparse
import inspect
from inspect import getmembers
from types import FunctionType
from pathlib import PurePath # Added for introspection helper
from machine.corpora import ParatextTextCorpus
#from machine.scripture import VerseRef # Import VerseRef for type checking/help


def attrs(obj):
    disallowed_properties = {
        name for name, value in getmembers(type(obj)) if isinstance(value, (property, FunctionType))
    }
    return {name: getattr(obj, name) for name in api(obj) if name not in disallowed_properties and hasattr(obj, name)}

def api(obj):
    """Helper to get public API members."""
    return [name for name in dir(obj) if not name.startswith('_')]


def print_table(rows):

    for arg, value in rows:
        if isinstance(value, PurePath):
            print(f"{str(arg):<30} : {str(value):<41} | {str(type(value)):<30} | {str(value.exists()):>6}")
        else:
            print(f"{str(arg):<30} : {str(value):<41} | {str(type(value)):<30}")

    print()


def show_attrs(cli_args, actions=[]):

    if cli_args:
        arg_rows = [(k, v) for k, v in cli_args.__dict__.items() if v is not None]

        print("Command line arguments:")
        print_table(arg_rows)
    
    if actions:
        for action in actions:
            print(type(action), action)

def introspect_object(obj, depth=2, indent="", visited=None):
    """
    Recursively introspects an object's attributes and methods.

    Args:
        obj: The object to introspect.
        depth: Maximum recursion depth.
        indent: String for indentation (used internally).
        visited: Set of object IDs already visited (used internally to prevent loops).
    """
    if depth < 0:
        return

    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        print(f"{indent}* Already visited object of type {type(obj).__name__} (id: {obj_id})")
        return

    visited.add(obj_id)

    try:
        obj_repr = repr(obj)
        # Limit representation length for clarity
        if len(obj_repr) > 100:
            obj_repr = obj_repr[:100] + "..."
    except Exception:
        obj_repr = "[Error getting repr]"

    print(f"{indent}Object: {obj_repr}")
    print(f"{indent}Type: {type(obj).__name__}")

    # Basic types we don't need to recurse into deeply
    basic_types = (int, float, str, bool, list, dict, set, tuple, bytes, type(None), PurePath)
    # Types that often cause recursion issues or aren't useful to inspect deeply here
    skip_types = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, property)

    try:
        # Using inspect.getmembers is often more informative than dir()
        members = inspect.getmembers(obj)
    except Exception as e:
        print(f"{indent}  [Error getting members: {e}]")
        members = []

    if not members:
         print(f"{indent}  (No members found or error getting them)")

    for name, member_obj in members:
        # Skip private/special members for cleaner output, unless depth is high
        if name.startswith("_") and depth < 3 and name != "__dict__": # Allow __dict__
             continue

        try:
            member_repr = repr(member_obj)
            if len(member_repr) > 80:
                member_repr = member_repr[:80] + "..."
        except Exception:
            member_repr = "[Error getting repr]"

        member_type = type(member_obj)
        print(f"{indent}  - {name} ({member_type.__name__}): {member_repr}")

        # Recurse if depth allows, it's not a basic/skip type, and not visited
        if depth > 0 and not isinstance(member_obj, basic_types + skip_types):
             # Check if it's a class attribute vs instance attribute if needed
             # Check if it's callable (method) - we usually don't recurse into methods' results
             if not callable(member_obj):
                  introspect_object(member_obj, depth - 1, indent + "    ", visited.copy()) # Pass copy of visited

# Standard Bible book order (useful for sorting output)
BOOK_ORDER = [
    "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT", "1SA", "2SA",
    "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST", "JOB", "PSA", "PRO",
    "ECC", "SNG", "ISA", "JER", "LAM", "EZK", "DAN", "HOS", "JOL", "AMO",
    "OBA", "JON", "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL",
    # NT
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
    # Deuterocanon / Apocrypha (add more as needed based on VRS files)
    "TOB", "JDT", "ESG", "WIS", "SIR", "BAR", "LJE", "S3Y", "SUS", "BEL",
    "1MA", "2MA", "3MA", "4MA", "1ES", "2ES", "MAN", "PS2", "ODA", "PSS",
    "EZA", "5EZ", "6EZ", "DAG", "LAO"
    # Add any other potential book codes found in VRS files or SFM data
]

# Create a mapping for sorting
BOOK_SORT_KEY = {book: i for i, book in enumerate(BOOK_ORDER)}

def get_translation_id(project_path: Path) -> str:
    """Derives a translation ID from the project folder name."""
    return project_path.name

# Helper to parse verse string from corpus (assuming this exists from previous diff)
# Regex for parsing verse strings like "1", "1a", "1-2", "1a-2b" etc. from corpus object
VERSE_STR_PATTERN = re.compile(r"(\d+)([a-zA-Z]?)")
def parse_verse_string(verse_str: str) -> tuple[int, str]:
    match = VERSE_STR_PATTERN.match(verse_str)
    if match:
        return int(match.group(1)), match.group(2)
    try:
        # Handle cases like verse "1" without letters
        return int(verse_str), ""
    except ValueError:
        # Fallback for unexpected formats
        return 0, ""

def parse_project_verse_counts(project_path: Path):
    """
    Parses all SFM files in a project directory to find max verse counts.

    Args:
        project_path: Path object pointing to the Bible project directory.

    Returns:
        A tuple containing:
        - verse_counts: A dictionary {book_id: {chapter: max_verse}}.
        - subdivisions_log: A list of strings logging found verse subdivisions.
    """

    if not project_path.is_dir():
        print(f"Error: Project path '{project_path}' not found or is not a directory.", file=sys.stderr)
        return None, None

    # Use defaultdict for easier handling of new books/chapters
    # Structure: {book_id: {chapter_num: max_verse_num}}
    verse_counts = defaultdict(lambda: defaultdict(int))
    subdivisions_log = []
    current_book = None

    print(f"Processing project: {project_path.name}")

    try:
        # ParatextTextCorpus expects a string path
        corpus = ParatextTextCorpus(str(project_path))

        # --- Use the recursive introspection function ---
        print("\n--- Recursive Introspection Start ---")
        introspect_object(corpus, depth=1) # Start with depth 1 or 2

        # --- Optionally, introspect the first verse ref specifically ---
        first_verse_ref = next(iter(corpus.verse_refs), None) # Corrected attribute name
        if first_verse_ref:
            print("\n--- Introspecting First VerseRef ---")
            introspect_object(first_verse_ref, depth=0) # Depth 0 just shows its direct info
            print("----------------------------------")

        print("--- Recursive Introspection End ---\n")

        exit() # Uncomment this line if you just want to see the introspection output


        processed_verses = 0

        for verse_ref in corpus.verse_refs:

            processed_verses += 1
            book_id = verse_ref.book.upper() # Ensure uppercase book ID
            chapter_num = verse_ref.chapter
            verse_str = verse_ref.verse # This might be '1', '1a', '1-2' etc.

            # Parse the verse string to get the primary number and any subdivision
            verse_num, subdivision_char = parse_verse_string(verse_str)

            if verse_num == 0 and verse_str != "0": # Check if parsing failed unexpectedly
                print(f"    Warning: Could not parse verse string '{verse_str}' for {book_id} {chapter_num}", file=sys.stderr)
                continue

            # Update max verse count for the current chapter
            verse_counts[book_id][chapter_num] = max(
                verse_counts[book_id][chapter_num], verse_num
            )

            # Log if a subdivision character exists
            if subdivision_char:
                # Note: We don't have the specific SFM filename easily here, log without it
                log_entry = f"{book_id} {chapter_num}:{verse_num}{subdivision_char}"
                if log_entry not in subdivisions_log: # Avoid duplicate logs
                        subdivisions_log.append(log_entry)

        if processed_verses == 0:
                print(f"Warning: No verses processed by ParatextTextCorpus in '{project_path}'. SFM files might be missing, empty, or unparsable.", file=sys.stderr)
                # Return empty structures if no verses found
                return {}, []

    except FileNotFoundError:
         print(f"Error: ParatextTextCorpus could not find project path '{project_path}' or required files within it.", file=sys.stderr)
         return None, None
    except Exception as e:
        print(f"Error initializing or processing with ParatextTextCorpus for '{project_path}': {e}", file=sys.stderr)
        print("Ensure the project directory contains valid SFM files.", file=sys.stderr)
        return None, None

    return verse_counts, subdivisions_log

def write_verse_counts_csv(project_path: Path, translation_id: str, verse_counts: dict):
    """Writes the parsed verse counts to a CSV file in the project directory."""
    output_filename = project_path / f"verse_counts_{translation_id}.csv"
    print(f"Writing output to: {output_filename}")

    # Prepare data for CSV, sorting by book order then chapter number
    output_data = []
    # Sort books based on BOOK_SORT_KEY, putting unknown books at the end alphabetically
    sorted_books = sorted(verse_counts.keys(), key=lambda b: (BOOK_SORT_KEY.get(b, float('inf')), b))

    for book in sorted_books:
        # Sort chapters numerically
        sorted_chapters = sorted(verse_counts[book].keys())
        for chapter in sorted_chapters:
            max_verse = verse_counts[book][chapter]
            # Only write rows if a max verse was actually found (i.e., > 0)
            if max_verse > 0:
                 output_data.append([book, chapter, max_verse])
            else:
                 print(f"  Note: Skipping {book} {chapter} as no verses were found.", file=sys.stderr)


    if not output_data:
        print("Warning: No verse data found to write to CSV.", file=sys.stderr)
        return

    try:
        with output_filename.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Book', 'Chapter', 'MaxVerse']) # Write header
            writer.writerows(output_data)
        print("CSV file written successfully.")
    except IOError as e:
        print(f"Error writing CSV file {output_filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while writing the CSV: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Parse SFM files in a Bible project folder to extract maximum verse counts per chapter.")
    parser.add_argument("project_folder", help="Path to the Bible project folder containing SFM files.")
    args = parser.parse_args()

    project_path = Path(args.project_folder).resolve() # Get absolute path

    verse_counts, subdivisions_log = parse_project_verse_counts(project_path)

    if verse_counts is None: # Indicates a fatal error during parsing setup
        sys.exit(1)

    if verse_counts:
        translation_id = get_translation_id(project_path)
        write_verse_counts_csv(project_path, translation_id, verse_counts)
    else:
        print("No verse counts generated.")

    if subdivisions_log:
        print("\n--- Found Verse Subdivisions ---")
        for entry in sorted(list(set(subdivisions_log))): # Sort and unique entries
            print(entry)
        print("------------------------------")
    else:
        print("\nNo verse subdivisions (e.g., v1a, v1b) detected.")

if __name__ == "__main__":
    main()
