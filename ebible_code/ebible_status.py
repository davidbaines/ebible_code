"""ebible_status.py contains functions for downloading and processing data from eBible.org.

This version uses a status file (metadata/ebible_status.csv) to track progress and avoid
re-running steps unnecessarily.

Workflow:
1. Read or initialize ebible_status.csv, merging info from translations.csv.
2. Filter translations based on command-line arguments.
3. Determine required actions (download, unzip, licence check) based on status
   dates, --max-age-days, and --force_download.
4. Perform downloads, updating status.
5. Perform unzipping, renaming, settings file creation, and licence extraction,
   updating status.
6. Save the updated ebible_status.csv.
7. Print commands for bulk extraction using SILNLP.
"""

import argparse
import os
import shutil
import sys

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from random import randint
from time import sleep, strftime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import regex
import requests
from bs4 import BeautifulSoup # Keep this import
from dotenv import load_dotenv
from rename_usfm import get_destination_file_from_book
from settings_file import write_settings_file, get_versification, get_vrs_diffs
from tqdm import tqdm

global headers
headers: Dict[str, str] = {
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0",
}

# --- Configuration ---
TODAY_STR = datetime.now(timezone.utc).date().isoformat()
STATUS_FILENAME = "ebible_status.csv"
TRANSLATIONS_FILENAME = "translations.csv"

# Define the columns for the status file
# Start with original columns from translations.csv
ORIGINAL_COLUMNS = [
    "languageCode", "translationId", "languageName", "languageNameInEnglish",
    "dialect", "homeDomain", "title", "description", "Redistributable",
    "Copyright", "UpdateDate", "publicationURL", "OTbooks", "OTchapters",
    "OTverses", "NTbooks", "NTchapters", "NTverses", "DCbooks", "DCchapters",
    "DCverses", "FCBHID", "Certified", "inScript", "swordName", "rodCode",
    "textDirection", "downloadable", "font", "shortTitle", "PODISBN", "script",
    "sourceDate"
]

# Add new status tracking columns
STATUS_COLUMNS = [
    "status_download_path", "status_download_date", "status_unzip_path",
    "status_unzip_date", "status_extract_path", "status_extract_date",
     "status_extract_renamed_date",
    "status_last_error", "status_inferred_versification" # Added new column
]

# Add new licence tracking columns
LICENCE_COLUMNS = [
    "licence_ID", "licence_File", "licence_Language", "licence_Dialect",
    "licence_Vernacular_Title", "licence_Licence_Type", "licence_Licence_Version",
    "licence_CC_Licence_Link", "licence_Copyright_Holder", "licence_Copyright_Years",
    "licence_Translation_by", "licence_date_read" 
]
# ALL_STATUS_COLUMNS is updated automatically by concatenating the lists
ALL_STATUS_COLUMNS = ORIGINAL_COLUMNS + LICENCE_COLUMNS + STATUS_COLUMNS 

# --- Utility Functions ---

def log_and_print(file: Path, messages, log_type="Info") -> None:
    """Logs messages to a file and prints them to the console."""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(file, "a", encoding='utf-8') as log:
            if isinstance(messages, str):
                log.write(f"{log_type}: {now_str} {messages}\n")
                print(messages)
            elif isinstance(messages, list):
                for message in messages:
                    log.write(f"{log_type}: {now_str} {message}\n")
                    print(message)
            elif isinstance(messages, pd.Series): # Handle pandas Series (like value_counts)
                msg_str = messages.to_string()
                log.write(f"{log_type}: {now_str}\n{msg_str}\n")
                print(msg_str)
            else: # Handle other types like value_counts() output
                 msg_str = str(messages)
                 log.write(f"{log_type}: {now_str}\n{msg_str}\n")
                 print(msg_str)

    except Exception as e:
        print(f"Error writing to log file {file}: {e}")
        # Also print the original message to console if logging failed
        if isinstance(messages, str):
            print(messages)
        elif isinstance(messages, list):
            for message in messages:
                print(message)
        else:
            print(str(messages))


def make_directories(dirs_to_create: List[Path]) -> None:
    """Creates directories if they don't exist."""
    for dir_to_create in dirs_to_create:
        dir_to_create.mkdir(parents=True, exist_ok=True)


def download_url_to_file(url: str, file: Path, headers: Dict = headers) -> bool:
    """Downloads a URL to a local file, returns True on success."""
    try:
        r = requests.get(url, headers=headers, timeout=60) # Added timeout
        r.raise_for_status() # Raise an exception for bad status codes
        with open(file, "wb") as out_file:
            out_file.write(r.content)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # Clean up potentially incomplete file
        if file.exists():
            try:
                file.unlink()
            except OSError as unlink_e:
                 print(f"Error removing incomplete download {file}: {unlink_e}")
        return False


def build_zip_filename(translation_id: str, date_str: str) -> str:
    """Builds the standard zip filename."""
    # Ensure date_str is in YYYY-MM-DD format if it's a date object
    if isinstance(date_str, date):
        date_str = date_str.isoformat()
    return f"{translation_id}--{date_str}.zip"


def is_date_older_than(date_str: Optional[str], max_days: int) -> bool:
    """Checks if a date string (YYYY-MM-DD) is older than max_days ago."""
    if pd.isna(date_str) or not isinstance(date_str, str) or not date_str:
        return True # Treat missing or invalid dates as old
    try:
        record_date = date.fromisoformat(date_str)
        cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=max_days)
        return record_date < cutoff_date
    except ValueError:
        return True # Treat parse errors as old


def choose_yes_no(prompt: str) -> bool:
    """Prompts user for Y/N input."""
    choice: str = " "
    while choice not in ["n", "y"]:
        try:
            choice = input(prompt).strip()[0].lower()
        except IndexError:
            pass # Handle empty input
    return choice == "y"


def check_folders_exist(folders: list, base: Path, logfile: Path):
    """Checks if required folders exist, prompts to create if missing."""
    missing_folders: List[Path] = [folder for folder in folders if not folder.is_dir()]

    print(f"The base folder is : {base}")

    if missing_folders:
        print(
            f"\nThe following {len(missing_folders)} folders are required but missing:"
        )
        for folder in missing_folders:
            print(folder)

        print(f"\nBase folder check:    {base} ")
        if choose_yes_no("Create missing folders and continue? (Y/N): "):
            make_directories(missing_folders)
            log_and_print(logfile, f"Created required folders within {base}\n")
        else:
            print("Aborting script.")
            sys.exit() # Use sys.exit for clarity
    else:
        # Only log if the log folder itself exists (which it should after make_directories)
        if logfile.parent.exists():
             log_and_print(logfile, f"All required folders exist in {base}")
        else:
             print(f"Log folder {logfile.parent} does not exist, cannot log folder check.")


# --- Scanning Functions for Existing Data ---

def scan_download_folder(translation_id: str, downloads_folder: Path) -> Optional[tuple[str, str]]:
    """Scans downloads folder for the latest zip file for a translationId."""
    potential_zips: List[Path] = sorted(
        downloads_folder.glob(f"{translation_id}--*-*-*.zip"),
        key=os.path.getmtime, # Sort by modification time
        reverse=True # Latest first
    )

    if not potential_zips:
        return None

    latest_zip = potential_zips[0]
    try:
        # Extract date from filename if possible, otherwise use file mtime
        match = regex.search(r"--(\d{4}-\d{2}-\d{2})\.zip$", latest_zip.name)
        if match:
            date_str = match.group(1)
        else:
            # Fallback to file modification time
            mtime = latest_zip.stat().st_mtime
            date_str = datetime.fromtimestamp(mtime, tz=timezone.utc).date().isoformat()
        return str(latest_zip.resolve()), date_str
    except Exception:
        # If anything goes wrong, fallback to mtime
        mtime = latest_zip.stat().st_mtime
        date_str = datetime.fromtimestamp(mtime, tz=timezone.utc).date().isoformat()
        return str(latest_zip.resolve()), date_str

def scan_project_folder(project_path: Path) -> Optional[tuple[str, str]]:
    """Checks if a project folder exists and returns its path and modification date."""
    if project_path.is_dir():
        try:
            # Use the folder's modification time as a proxy for unzip date
            mtime = project_path.stat().st_mtime
            date_str = datetime.fromtimestamp(mtime, tz=timezone.utc).date().isoformat()
            return str(project_path.resolve()), date_str
        except Exception:
            return None # Error getting stat
    return None

def scan_corpus_file(extract_path: Path) -> Optional[tuple[str, str]]:
    """Checks if an extracted corpus file exists and returns its path and modification date."""
    if extract_path.is_file():
        try:
            mtime = extract_path.stat().st_mtime
            date_str = datetime.fromtimestamp(mtime, tz=timezone.utc).date().isoformat()
            return str(extract_path.resolve()), date_str
        except Exception:
            return None # Error getting stat
    return None


# --- Core Logic Functions ---

def initialize_or_load_status(status_path: Path, translations_path: Path, logfile: Path) -> pd.DataFrame:
    """Loads the status CSV, or creates it from the translations CSV if it doesn't exist."""
    if status_path.exists():
        log_and_print(logfile, f"Loading existing status file: {status_path}")
        try:
            status_df = pd.read_csv(status_path, keep_default_na=False, na_values=['']) # Treat empty strings as NA
            # Verify essential columns exist
            if not 'translationId' in status_df.columns:
                 raise ValueError("Status file missing 'translationId' column.")
            # Add any missing columns with default NaN values
            for col in ALL_STATUS_COLUMNS:
                if col not in status_df.columns:
                    log_and_print(logfile, f"Adding missing column '{col}' to status DataFrame.")
                    status_df[col] = np.nan
            # Ensure correct order
            status_df = status_df[ALL_STATUS_COLUMNS]

        except Exception as e:
            log_and_print(logfile, f"Error loading status file {status_path}: {e}. Attempting to rebuild.", log_type="Error")
            status_path.unlink(missing_ok=True) # Remove corrupted file
            return initialize_or_load_status(status_path, translations_path, logfile) # Recurse to rebuild

    else:
        log_and_print(logfile, f"Status file not found. Creating new one: {status_path}")
        if not translations_path.exists():
             log_and_print(logfile, f"Error: translations file missing at {translations_path}. Cannot create status file.", log_type="Critical")
             sys.exit(1)
        try:
            # Read translations, ensuring 'translationId' is string
            trans_df = pd.read_csv(translations_path, dtype={'translationId': str}, keep_default_na=False, na_values=[''])
            if 'translationId' not in trans_df.columns:
                 raise ValueError("Translations file missing 'translationId' column.")

            # Create status DataFrame with all columns
            status_df = pd.DataFrame(columns=ALL_STATUS_COLUMNS)

            # Copy data from translations_df for matching columns
            for col in ORIGINAL_COLUMNS:
                 if col in trans_df.columns:
                      status_df[col] = trans_df[col]
                 else:
                      log_and_print(logfile, f"Warning: Column '{col}' not found in {translations_path}", log_type="Warn")
                      status_df[col] = np.nan # Add as empty column if missing

            # Initialize new status/licence columns with NaN
            for col in STATUS_COLUMNS + LICENCE_COLUMNS:
                status_df[col] = np.nan

            # Ensure translationId is the index for easier merging later
            # status_df.set_index('translationId', inplace=True) # Let's keep it as a column for now

        except Exception as e:
            log_and_print(logfile, f"Error creating status file from {translations_path}: {e}", log_type="Critical")
            sys.exit(1)

    # --- Merge upstream changes (optional but recommended) ---
    # This adds new translations from translations.csv to status.csv
    # It does NOT update existing rows in status.csv from translations.csv by default
    try:
        trans_df = pd.read_csv(translations_path, dtype={'translationId': str}, keep_default_na=False, na_values=[''])
        if 'translationId' not in trans_df.columns:
             raise ValueError("Translations file missing 'translationId' column during merge check.")

        existing_ids = set(status_df['translationId'].astype(str))
        upstream_ids = set(trans_df['translationId'].astype(str))
        new_ids = list(upstream_ids - existing_ids)

        if new_ids:
            log_and_print(logfile, f"Found {len(new_ids)} new translations in {translations_path}. Adding to status.")
            new_rows_df = trans_df[trans_df['translationId'].isin(new_ids)].copy()

            # Prepare new rows with all status columns, initializing non-original ones
            full_new_rows = pd.DataFrame(columns=ALL_STATUS_COLUMNS)
            for col in ORIGINAL_COLUMNS:
                 if col in new_rows_df.columns:
                      full_new_rows[col] = new_rows_df[col]
                 else:
                      full_new_rows[col] = np.nan
            for col in STATUS_COLUMNS + LICENCE_COLUMNS:
                 full_new_rows[col] = np.nan

            status_df = pd.concat([status_df, full_new_rows], ignore_index=True)
            # Consider saving immediately after adding new rows?
            # status_df.to_csv(status_path, index=False)

        removed_ids = list(existing_ids - upstream_ids)
        if removed_ids:
             log_and_print(logfile, f"Warning: {len(removed_ids)} translations exist in status but not in upstream {translations_path}. They will be kept but may be outdated.", log_type="Warn")
             # Optionally, mark them as inactive or remove them:
             # status_df = status_df[~status_df['translationId'].isin(removed_ids)]

    except Exception as e:
        log_and_print(logfile, f"Error merging upstream changes from {translations_path}: {e}", log_type="Error")

    # Ensure data types are reasonable (especially for boolean checks later)
    status_df['Redistributable'] = status_df['Redistributable'].astype(str).str.lower() == 'true'
    status_df['downloadable'] = status_df['downloadable'].astype(str).str.lower() == 'true'
    # Convert verse counts safely to numeric, coercing errors to NaN, then fill NaN with 0
    for col in ['OTverses', 'NTverses']:
        status_df[col] = pd.to_numeric(status_df[col], errors='coerce').fillna(0).astype(int)

    return status_df

def scan_and_update_status(
    status_df: pd.DataFrame,
    downloads_folder: Path,
    projects_folder: Path,
    private_projects_folder: Path,
    corpus_folder: Path,
    private_corpus_folder: Path,
    logfile: Path
) -> pd.DataFrame:
    """Scans data folders to update status DataFrame for entries with missing info."""
    log_and_print(logfile, "Scanning existing data folders to update status file...")
    updated_count = 0
    for index, row in tqdm(status_df.iterrows(), total=len(status_df), desc="Scanning Folders"):
        translation_id = row['translationId']
        lang_code = row['languageCode']
        is_redist = row['Redistributable'] # Assumes this column is correctly typed bool

        # Scan Downloads
        if pd.isna(row['status_download_date']):
            scan_result = scan_download_folder(translation_id, downloads_folder)
            if scan_result:
                status_df.loc[index, 'status_download_path'] = scan_result[0]
                status_df.loc[index, 'status_download_date'] = scan_result[1]
                updated_count += 1

        # Scan Projects (Unzipped)
        if pd.isna(row['status_unzip_date']):
            proj_base = projects_folder if is_redist else private_projects_folder
            proj_name = translation_id
            project_dir = proj_base / proj_name
            scan_result = scan_project_folder(project_dir)
            if scan_result:
                status_df.loc[index, 'status_unzip_path'] = scan_result[0]
                status_df.loc[index, 'status_unzip_date'] = scan_result[1]
                updated_count += 1

        # Scan Corpus (Extracted) - Note: Extract path is less certain, depends on SILNLP output format
        # This part might need adjustment if SILNLP naming changes. Assuming standard {lang}-{proj_name}.txt
        # Also, status_extract_path/date might be less critical now, but we can scan for it.
        if pd.isna(row['status_extract_date']):
            proj_name = translation_id
            corpus_base = corpus_folder if is_redist else private_corpus_folder
            # Construct expected extract filename - THIS IS AN ASSUMPTION
            expected_extract_filename = f"{lang_code}-{proj_name}.txt"
            extract_path = corpus_base / expected_extract_filename
            scan_result = scan_corpus_file(extract_path)
            if scan_result:
                status_df.loc[index, 'status_extract_path'] = scan_result[0]
                status_df.loc[index, 'status_extract_date'] = scan_result[1]
                updated_count += 1

    if updated_count > 0:
        log_and_print(logfile, f"Scan complete. Updated status for {updated_count} entries based on existing files.")
    else:
        log_and_print(logfile, "Scan complete. No missing status information updated from existing files.")

    return status_df


def ensure_extract_paths(
    status_df: pd.DataFrame,
    corpus_folder: Path,
    private_corpus_folder: Path,
    logfile: Path
) -> pd.DataFrame:
    """Calculates and fills the status_extract_path column if missing."""
    log_and_print(logfile, "Ensuring status_extract_path is populated...")
    for index, row in status_df.iterrows():
        if pd.isna(row['status_extract_path']):
            lang_code = row['languageCode']
            translation_id = row['translationId']
            is_redist = row['Redistributable'] # Assumes bool
            proj_name = translation_id
            corpus_base = corpus_folder if is_redist else private_corpus_folder
            expected_extract_filename = f"{lang_code}-{proj_name}.txt"
            status_df.loc[index, 'status_extract_path'] = str((corpus_base / expected_extract_filename).resolve())
    return status_df


def filter_translations(df: pd.DataFrame, allow_non_redistributable: bool, verse_threshold: int, regex_filter: Optional[str], logfile: Path) -> pd.DataFrame:
    """Filters the DataFrame based on criteria."""
    initial_count = len(df)
    log_and_print(logfile, f"Initial translations in status file: {initial_count}")

    # 1. Filter by downloadable flag
    df = df[df['downloadable'] == True]
    log_and_print(logfile, f"Translations after 'downloadable' filter: {len(df)}")

    # 2. Filter by redistributable flag (if applicable)
    if not allow_non_redistributable:
        df = df[df['Redistributable'] == True]
        log_and_print(logfile, f"Translations after 'Redistributable' filter: {len(df)}")

    # 3. Filter by verse count
    df = df[(df['OTverses'] + df['NTverses']) >= verse_threshold]
    log_and_print(logfile, f"Translations after verse count filter (>= {verse_threshold}): {len(df)}")

    # 4. Apply regex filter (if provided)
    if regex_filter:
        try:
            df = df[df['translationId'].astype(str).str.match(regex_filter, na=False)]
            log_and_print(logfile, f"Translations after regex filter ('{regex_filter}'): {len(df)}")
        except regex.error as e:
            log_and_print(logfile, f"Invalid regex filter '{regex_filter}': {e}. Skipping filter.", log_type="Error")

    final_count = len(df)
    log_and_print(logfile, f"Filtered down to {final_count} translations to process.")
    return df


def determine_actions(df: pd.DataFrame, max_age_days: int, force_download: bool, downloads_folder: Path, projects_folder: Path, private_projects_folder: Path) -> pd.DataFrame:
    """Adds boolean columns indicating required actions."""

    df['action_needed_download'] = False
    df['action_needed_unzip'] = False
    df['action_needed_licence'] = False

    for index, row in df.iterrows():
        # --- Download Check ---
        needs_download = False
        if force_download:
            needs_download = True
        elif is_date_older_than(row['status_download_date'], max_age_days):
            needs_download = True
        elif pd.isna(row['status_download_path']) or not Path(row['status_download_path']).exists():
             # Check if file exists only if date is recent
             needs_download = True

        df.loc[index, 'action_needed_download'] = needs_download

        # --- Unzip Check ---
        needs_unzip = False
        if needs_download: # If downloading, must unzip
            needs_unzip = True
        elif force_download: # Force implies re-unzip too
            needs_unzip = True
        elif is_date_older_than(row['status_unzip_date'], max_age_days):
            needs_unzip = True
        elif pd.isna(row['status_unzip_path']) or not Path(row['status_unzip_path']).exists():
             # Check if dir exists only if date is recent
             needs_unzip = True

        df.loc[index, 'action_needed_unzip'] = needs_unzip

        # --- Licence Check ---
        needs_licence = False
        if needs_unzip: # If unzipping, must re-check licence
            needs_licence = True
        elif force_download: # Force implies re-check
            needs_licence = True
        elif is_date_older_than(row['licence_date_read'], max_age_days):
            needs_licence = True
        # No path check needed here, as licence data is in the status file itself

        df.loc[index, 'action_needed_licence'] = needs_licence

    return df


def download_required_files(df: pd.DataFrame, base_url: str, folder: Path, logfile: Path) -> pd.DataFrame:
    """Downloads files marked with action_needed_download."""
    translations_to_download = df[df['action_needed_download']]
    count = len(translations_to_download)
    log_and_print(logfile, f"Attempting to download {count} zip files...")

    downloaded_count = 0
    for index, row in tqdm(translations_to_download.iterrows(), total=count, desc="Downloading"):
        translation_id = row['translationId']
        url = f"{base_url}{translation_id}_usfm.zip"
        # Always use today's date for new downloads
        local_filename = build_zip_filename(translation_id, TODAY_STR)
        local_path = folder / local_filename

        log_and_print(logfile, f"Downloading {url} to {local_path}")
        if download_url_to_file(url, local_path):
            df.loc[index, 'status_download_path'] = str(local_path.resolve())
            df.loc[index, 'status_download_date'] = TODAY_STR
            df.loc[index, 'status_last_error'] = np.nan # Clear previous error on success
            downloaded_count += 1
            log_and_print(logfile, f"Success: Saved {url} as {local_path}")
            sleep(randint(1, 3000) / 1000) # Shorter sleep?
        else:
            df.loc[index, 'status_download_path'] = np.nan # Clear path on failure
            df.loc[index, 'status_download_date'] = np.nan # Clear date on failure
            df.loc[index, 'status_last_error'] = f"Download failed: {url}"
            log_and_print(logfile, f"Failed: Could not download {url}", log_type="Error")

    log_and_print(logfile, f"Finished downloading. Successfully downloaded {downloaded_count}/{count} files.")
    return df


def unzip_and_process_files(df: pd.DataFrame, downloads_folder: Path, projects_folder: Path, private_projects_folder: Path, vrs_diffs_data: Dict, logfile: Path) -> pd.DataFrame:
    """Unzips, renames, creates settings, and extracts licence for required projects."""
    translations_to_unzip = df[df['action_needed_unzip']]
    count = len(translations_to_unzip)
    log_and_print(logfile, f"Attempting to unzip and process {count} projects...")

    processed_count = 0
    for index, row in tqdm(translations_to_unzip.iterrows(), total=count, desc="Unzipping/Processing"):
        translation_id = row['translationId']
        lang_code = row['languageCode']
        is_redist = row['Redistributable']
        download_path_str = row['status_download_path']

        if pd.isna(download_path_str):
            log_and_print(logfile, f"Skipping unzip for {translation_id}: No valid download path found.", log_type="Warn")
            df.loc[index, 'status_last_error'] = "Unzip skipped: Missing download path"
            continue

        download_path = Path(download_path_str)
        if not download_path.exists():
             log_and_print(logfile, f"Skipping unzip for {translation_id}: Download path {download_path} not found.", log_type="Warn")
             df.loc[index, 'status_last_error'] = f"Unzip skipped: Download not found at {download_path}"
             continue

        unzip_base_dir = projects_folder if is_redist else private_projects_folder
        proj_name = translation_id
        project_dir = unzip_base_dir / proj_name

        log_and_print(logfile, f"Processing {translation_id}: Unzipping {download_path} to {project_dir}")

        # --- Unzip ---
        try:
            # Clean existing directory before unzipping
            if project_dir.exists():
                log_and_print(logfile, f"Removing existing directory: {project_dir}")
                shutil.rmtree(project_dir)
            project_dir.mkdir(parents=True, exist_ok=True)

            shutil.unpack_archive(download_path, project_dir)
            df.loc[index, 'status_unzip_path'] = str(project_dir.resolve())
            df.loc[index, 'status_unzip_date'] = TODAY_STR
            df.loc[index, 'status_last_error'] = np.nan # Clear error on successful unzip
            log_and_print(logfile, f"Successfully unzipped to {project_dir}")

            # --- Post-Unzip Processing ---
            # Rename USFM files
            rename_usfm(project_dir, logfile)

            # Write Settings.xml
             # Unpack all return values, even if old/new dicts aren't used here yet
            settings_path, vrs_num, _, _ = write_settings_file(project_dir, lang_code, translation_id, vrs_diffs_data)
            df.loc[index, 'status_inferred_versification'] = vrs_num # Store the inferred versification number

            # Extract Licence Details (only if needed or forced)
            if row['action_needed_licence']:
                 df = get_and_update_licence_details(df, index, project_dir, logfile)

            processed_count += 1

        except (shutil.ReadError, FileNotFoundError, OSError, Exception) as e:
            log_and_print(logfile, f"Error processing {translation_id} at {project_dir}: {e}", log_type="Error")
            df.loc[index, 'status_unzip_path'] = np.nan
            df.loc[index, 'status_unzip_date'] = np.nan
            df.loc[index, 'status_last_error'] = f"Processing error: {e}"
            # Clean up potentially corrupted unzip dir
            if project_dir.exists():
                 try:
                      shutil.rmtree(project_dir)
                 except OSError as rm_e:
                      log_and_print(logfile, f"Could not remove failed unzip dir {project_dir}: {rm_e}", log_type="Warn")

    log_and_print(logfile, f"Finished processing. Successfully processed {processed_count}/{count} projects.")
    return df


def rename_usfm(project_dir: Path, logfile: Path):
    """Renames USFM files within the project directory."""
    log_and_print(logfile, f"Renaming USFM files in {project_dir}")
    renamed_count = 0
    try:
        usfm_paths = list(project_dir.glob("*.usfm"))
        for old_usfm_path in usfm_paths:
            new_sfm_path = get_destination_file_from_book(old_usfm_path)
            if new_sfm_path == old_usfm_path:
                continue
            if new_sfm_path.is_file():
                new_sfm_path.unlink() # Remove existing target

            # log_and_print(logfile, f"Renaming {old_usfm_path.name} to {new_sfm_path.name}")
            old_usfm_path.rename(new_sfm_path)
            renamed_count += 1
        if renamed_count > 0:
             log_and_print(logfile, f"Renamed {renamed_count} USFM files.")
    except Exception as e:
        log_and_print(logfile, f"Error renaming USFM files in {project_dir}: {e}", log_type="Error")



def get_and_update_licence_details(df: pd.DataFrame, index, project_dir: Path, logfile: Path) -> pd.DataFrame:
    """Extracts licence details from copr.htm and updates the DataFrame row."""
    copyright_path = project_dir / "copr.htm"
    log_and_print(logfile, f"Extracting licence info for {project_dir.name} from {copyright_path}")

    # Clear previous licence data for this row first
    for col in LICENCE_COLUMNS:
        if col != 'licence_date_read': # Keep date read until success
             df.loc[index, col] = np.nan

    if not copyright_path.exists():
        log_and_print(logfile, f"Unable to find {copyright_path}", log_type="Warn")
        df.loc[index, 'status_last_error'] = f"Licence check failed: copr.htm not found"
        df.loc[index, 'licence_date_read'] = TODAY_STR # Mark as checked today, even if failed
        return df

    entry = {} # Use a temporary dict
    entry["licence_ID"] = project_dir.name
    entry["licence_File"] = str(copyright_path.resolve())

    try:
        with open(copyright_path, "r", encoding="utf-8") as copr:
            html = copr.read()
            soup = BeautifulSoup(html, "lxml")

        cclink = soup.find(href=regex.compile("creativecommons"))
        if cclink:
            ref = cclink.get("href")
            if ref:
                entry["licence_CC_Licence_Link"] = ref
                # More robust CC parsing
                cc_match = regex.search(r"/licenses/([a-z\-]+)/([\d\.]+)", ref)
                if cc_match:
                    entry["licence_Licence_Type"] = cc_match.group(1)
                    entry["licence_Licence_Version"] = cc_match.group(2)
                else: # Handle simpler cases like /by/4.0/
                     cc_match_simple = regex.search(r"/licenses/([a-z\-]+)/?", ref)
                     if cc_match_simple:
                          entry["licence_Licence_Type"] = cc_match_simple.group(1)
                          # Try to find version elsewhere if needed

        titlelink = soup.find(href=regex.compile(f"https://ebible.org/{entry['licence_ID']}"))
        if titlelink and titlelink.string:
            entry["licence_Vernacular_Title"] = titlelink.string.strip()

        # Extract text, handle potential missing <p> or body
        body_tag = soup.body
        if body_tag and body_tag.p:
             copy_strings = [s.strip() for s in body_tag.p.stripped_strings if s.strip()]
        elif body_tag:
             copy_strings = [s.strip() for s in body_tag.stripped_strings if s.strip()]
        else:
             copy_strings = []
             log_and_print(logfile, f"Warning: No body or paragraph tag found in {copyright_path}", log_type="Warn")


        # Simpler text parsing logic
        is_public_domain = False
        for i, text in enumerate(copy_strings):
            if "public domain" in text.lower():
                is_public_domain = True
                break # Assume PD overrides other info
            elif "copyright ©" in text.lower():
                 entry["licence_Copyright_Years"] = text # Keep full string for now
                 if i + 1 < len(copy_strings):
                      entry["licence_Copyright_Holder"] = copy_strings[i+1]
            elif text.lower().startswith("language:"):
                 if i + 1 < len(copy_strings):
                      entry["licence_Language"] = copy_strings[i+1]
            elif text.lower().startswith("dialect"): # Handles "Dialect:" or "Dialect (if applicable):"
                 # Take rest of string after colon, or the next string if current is just "Dialect:"
                 parts = text.split(":", 1)
                 if len(parts) > 1 and parts[1].strip():
                      entry["licence_Dialect"] = parts[1].strip()
                 elif i + 1 < len(copy_strings):
                      entry["licence_Dialect"] = copy_strings[i+1]
            elif "translation by" in text.lower():
                 entry["licence_Translation_by"] = text # Keep full string

        if is_public_domain:
            entry["licence_Copyright_Holder"] = "Public Domain"
            entry["licence_Licence_Type"] = "Public Domain" # Standardize
            entry["licence_Copyright_Years"] = "" # Clear years for PD

        # --- Data Cleaning/Defaults ---
        if pd.isna(entry.get("licence_Licence_Type")):
             if "Public Domain" == entry.get("licence_Copyright_Holder"):
                  entry["licence_Licence_Type"] = "Public Domain"
             elif entry.get("licence_CC_Licence_Link"):
                  entry["licence_Licence_Type"] = "CC (Unknown Version)" # Indicate CC link exists but type/version parse failed
             else:
                  entry["licence_Licence_Type"] = "Unknown" # Default if no other info

        # Apply specific known fixes (example)
        if entry["licence_ID"] in ["engwmb", "engwmbb"]:
             entry["licence_Copyright_Holder"] = "Public Domain"
             entry["licence_Licence_Type"] = "Public Domain"

        # Update DataFrame row
        for col_suffix, value in entry.items():
             # col_name = f"licence_{col_suffix}" # Prefix already included in entry keys
             if col_suffix in df.columns:
                  df.loc[index, col_suffix] = value
             else:
                  log_and_print(logfile, f"Warning: Licence key '{col_suffix}' not a column in DataFrame.", log_type="Warn")

        df.loc[index, 'licence_date_read'] = TODAY_STR
        df.loc[index, 'status_last_error'] = np.nan # Clear error on success
        log_and_print(logfile, f"Successfully extracted licence info for {project_dir.name}")

    except Exception as e:
        log_and_print(logfile, f"Error parsing licence file {copyright_path}: {e}", log_type="Error")
        df.loc[index, 'status_last_error'] = f"Licence parse error: {e}"
        df.loc[index, 'licence_date_read'] = TODAY_STR # Mark as checked today, even if failed

    return df

def check_and_update_licences(df: pd.DataFrame, logfile: Path) -> pd.DataFrame:
    """Checks and updates licence details for projects that weren't re-unzipped but need a licence check."""
    # Filter for rows needing licence check BUT NOT unzip (as unzip handles its own licence check)
    licence_check_candidates = df[df['action_needed_licence'] & ~df['action_needed_unzip']]
    count = len(licence_check_candidates)

    if count == 0:
        log_and_print(logfile, "No existing projects require a separate licence check.")
        return df

    log_and_print(logfile, f"Performing licence check for {count} existing projects...")
    checked_count = 0
    for index, row in tqdm(licence_check_candidates.iterrows(), total=count, desc="Checking Licences"):
        project_path_str = row['status_unzip_path']

        if pd.isna(project_path_str):
            log_and_print(logfile, f"Skipping licence check for {row['translationId']}: Missing unzip path.", log_type="Warn")
            continue

        project_dir = Path(project_path_str)
        if project_dir.is_dir():
            df = get_and_update_licence_details(df, index, project_dir, logfile)
            checked_count += 1
        else:
            log_and_print(logfile, f"Skipping licence check for {row['translationId']}: Project directory {project_dir} not found.", log_type="Warn")

    log_and_print(logfile, f"Finished separate licence check. Updated {checked_count}/{count} projects.")
    return df

def rename_extracted_files(
    df: pd.DataFrame,
    corpus_folder: Path,
    private_corpus_folder: Path,
    logfile: Path
) -> pd.DataFrame:
    """Renames extracted files from SILNLP output format to {translation_id}.txt."""
    # Filter for rows where extract exists but rename hasn't happened
    rename_candidates = df[df['status_extract_date'].notna() & df['status_extract_renamed_date'].isna()]
    count = len(rename_candidates)

    if count == 0:
        # log_and_print(logfile, "No extracted files require renaming.") # Reduce noise
        return df

    log_and_print(logfile, f"Attempting to rename {count} extracted files...")
    renamed_count = 0
    target_file_exists_count = 0
    for index, row in tqdm(rename_candidates.iterrows(), total=count, desc="Renaming Extracts"):
        lang_code = row['languageCode']
        translation_id = row['translationId']
        is_redist = row['Redistributable']
        corpus_base = corpus_folder if is_redist else private_corpus_folder

        # --- Determine potential SILNLP output filenames ---
        # Pattern 1: lang-lang-id.txt (e.g., abt-abt-maprik.txt)
        silnlp_output_name1 = f"{lang_code}-{translation_id}.txt"
        # Pattern 2: lang-id.txt (e.g., aoj-aoj.txt - SILNLP might handle simple cases differently)
        # This might be the same as translation_id.txt if lang==id prefix
        silnlp_output_name2 = f"{lang_code}-{translation_id}.txt" # Corrected: This was same as pattern 1, should be simpler
        # Let's assume the most likely pattern is lang-translation_id.txt based on input project name
        silnlp_output_path = corpus_base / silnlp_output_name1

        target_name = f"{translation_id}.txt"
        target_path = corpus_base / target_name

        if silnlp_output_path.exists() and silnlp_output_path != target_path:
            if target_path.exists():
                #log_and_print(logfile, f"Warning: Target rename path {target_path} already exists. Skipping rename for {translation_id}.", log_type="Warn")
                target_file_exists_count += 1
            else:
                try:
                    silnlp_output_path.rename(target_path)
                    df.loc[index, 'status_extract_renamed_date'] = TODAY_STR
                    df.loc[index, 'status_extract_path'] = str(target_path.resolve()) # Update path to correct one
                    renamed_count += 1
                    log_and_print(logfile, f"Renamed {silnlp_output_path.name} to {target_path.name}")
                except OSError as e:
                    log_and_print(logfile, f"Error renaming {silnlp_output_path} to {target_path}: {e}", log_type="Error")
                    df.loc[index, 'status_last_error'] = f"Extract rename failed: {e}"
        elif target_path.exists():
             # If the target already exists, assume it was renamed previously or SILNLP produced correct name
             if pd.isna(row['status_extract_renamed_date']):
                  df.loc[index, 'status_extract_renamed_date'] = TODAY_STR # Mark as done
                  log_and_print(logfile, f"Target file {target_path.name} already exists, marking rename as complete.")
        # else: # File not found with expected SILNLP pattern
            # log_and_print(logfile, f"Could not find expected SILNLP output file {silnlp_output_path.name} for {translation_id}", log_type="Warn")

    if renamed_count :
        log_and_print(logfile, f"Finished renaming. Renamed {renamed_count} files.")

    if target_file_exists_count:
        log_and_print(logfile, f"Didn't rename {target_file_exists_count} files that already exist.")
    return df


# --- Function for --update-settings mode ---

def update_all_settings(
    status_df: pd.DataFrame,
    projects_folder: Path,
    private_projects_folder: Path,
    vrs_diffs: Dict,
    logfile: Path
) -> pd.DataFrame:
    """
    Iterates through project folders, regenerates Settings.xml, and updates status_df.
    """
    log_and_print(logfile, "--- Running in --update-settings mode ---")
    settings_report_data = [] # List to store data for the CSV report
    processed_folders = 0

    # Ensure status_df has translationId as index for quick lookup
    status_df_indexed = status_df.set_index('translationId', drop=False)

    for base_folder in [projects_folder, private_projects_folder]:
        log_and_print(logfile, f"Scanning folder: {base_folder}")
        if not base_folder.is_dir():
            log_and_print(logfile, f"Warning: Folder not found, skipping: {base_folder}", log_type="Warn")
            continue

        for project_dir in base_folder.iterdir():
            if project_dir.is_dir():
                processed_folders += 1
                translation_id = project_dir.name
                if translation_id in status_df_indexed.index:
                    index = status_df_indexed.index.get_loc(translation_id) # Get integer index location
                    row = status_df_indexed.iloc[index]
                    lang_code = row['languageCode']
                    if pd.isna(lang_code):
                        log_and_print(logfile, f"Skipping {translation_id}: Missing languageCode in status file.", log_type="Warn")
                        continue

                    log_and_print(logfile, f"Updating settings for {translation_id} in {project_dir}")
                    settings_path, vrs_num, old_vals, new_vals = write_settings_file(project_dir, lang_code, vrs_diffs)

                    if settings_path:
                        # Use the original DataFrame and integer index to update
                        status_df.loc[status_df['translationId'] == translation_id, 'status_inferred_versification'] = vrs_num
                        
                        # Add data to report
                        report_entry = {
                            "translationId": translation_id,
                            "settings_path": str(settings_path.resolve()),
                            **old_vals,
                            **new_vals,
                        }
                        settings_report_data.append(report_entry)
                    else:
                        log_and_print(logfile, f"Failed to write settings for {translation_id}", log_type="Error")
                        status_df.loc[status_df['translationId'] == translation_id, 'status_last_error'] = "Settings update failed"
                else:
                    log_and_print(logfile, f"Skipping {project_dir}: No entry found in status file for translationId '{translation_id}'.", log_type="Warn")

    # --- Write the settings update report ---
    if settings_report_data:
        report_df = pd.DataFrame(settings_report_data)
        report_path = logfile.parent.parent / "metadata" / "settings_update.csv" # Place in metadata folder
        try:
            report_df.to_csv(report_path, index=False, encoding='utf-8')
            log_and_print(logfile, f"Saved settings update report to {report_path}")
        except Exception as e:
            log_and_print(logfile, f"Error saving settings update report to {report_path}: {e}", log_type="Error")

    log_and_print(logfile, f"--- Settings update complete. Processed {processed_folders} potential project folders. ---")
    return status_df

# --- Main Execution ---

def main() -> None:
    load_dotenv()

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Download, unzip and process eBible translations using a status file."
    )
    parser.add_argument(
        "-f", "--filter", default=None,
        help="Regex filter for translationId (e.g., 'eng-.*' or '^(fra|spa)')",
    )
    parser.add_argument(
        "-d", "--force_download", default=False, action="store_true",
        help="Force download, unzip, and licence check for all filtered translations.",
    )
    parser.add_argument(
        "--allow_non_redistributable", default=False, action="store_true",
        help="Include non-redistributable (private) translations.",
    )
    # --download_only might need rethinking with status file, maybe remove or adapt?
    parser.add_argument(
        "--download_only", default=False, action="store_true",
        help="Stop after downloading zip files.",
    )
    parser.add_argument(
        "--max-age-days", default=None, type=int,
        help="Max age in days for downloaded/unzipped files before re-processing. Overrides .env.",
    )
    parser.add_argument(
        "--base-folder", default=None,
        help="Override base folder location (defaults to EBIBLE_DATA_DIR from .env or './_ebible_data').",
    )
    parser.add_argument(
        "--verse-threshold", default=400, type=int,
        help="Minimum total OT+NT verses required for a translation to be processed.",
    )
    parser.add_argument(
        "--update-settings", default=False, action="store_true",
        help="Run in a mode to only update Settings.xml for existing projects and exit.",
    )
    args: argparse.Namespace = parser.parse_args()

    # --- Determine Base Path ---
    if args.base_folder:
        base = Path(args.base_folder).resolve()
        print(f"Using base folder from command line: {base}")
    elif os.getenv("EBIBLE_DATA_DIR"):
        base = Path(os.getenv("EBIBLE_DATA_DIR")).resolve()
        print(f"Using base folder from EBIBLE_DATA_DIR env var: {base}")
    else:
        # Default relative to the script's location might be safer
        base = (Path(__file__).parent / "_ebible_data").resolve()
        # repo_root = Path(__file__).parent.parent.parent.parent # Old assumption
        # base = repo_root / "_ebible_data"
        print(f"Using default base folder: {base}")

    # --- Define Paths ---
    corpus_folder: Path = base / "corpus"
    downloads_folder: Path = base / "downloads"
    private_corpus_folder: Path = base / "private_corpus"
    private_projects_folder: Path = base / "private_projects"
    projects_folder: Path = base / "projects"
    metadata_folder: Path = base / "metadata"
    logs_folder: Path = base / "logs"

    # --- Setup Logging ---
    logs_folder.mkdir(parents=True, exist_ok=True) # Ensure log dir exists first
    year, month, day, hour, minute = map(int, strftime("%Y %m %d %H %M").split())
    log_suffix: str = f"_{year}_{month:02d}_{day:02d}-{hour:02d}_{minute:02d}.log"
    log_filename: str = "ebible_status" + log_suffix
    logfile: Path = logs_folder / log_filename
    print(f"Logging to: {logfile}")

    # --- Check Folders ---
    required_folders = [
        corpus_folder, downloads_folder, private_corpus_folder,
        private_projects_folder, projects_folder, metadata_folder, logs_folder
    ]
    check_folders_exist(required_folders, base, logfile)

    # --- Determine Max Age ---
    max_age_days = args.max_age_days
    if max_age_days is None:
        env_max_age = os.getenv("MAX_AGE_DAYS")
        if env_max_age and env_max_age.isdigit():
            max_age_days = int(env_max_age)
            log_and_print(logfile, f"Using MAX_AGE_DAYS={max_age_days} from .env file.")
        else:
            max_age_days = 365 # Hardcoded default
            log_and_print(logfile, f"Using default max_age_days={max_age_days}.")
    else:
        log_and_print(logfile, f"Using --max-age-days={max_age_days} from command line.")

    # --- Download translations.csv if needed ---
    translations_csv_url: str = r"https://ebible.org/Scriptures/translations.csv"
    translations_csv: Path = metadata_folder / TRANSLATIONS_FILENAME
    if not translations_csv.is_file() or args.force_download:
        log_and_print(logfile, f"Downloading {translations_csv_url} to {translations_csv}")
        if not download_url_to_file(translations_csv_url, translations_csv):
            log_and_print(logfile, f"Critical: Failed to download {translations_csv}. Aborting.", log_type="Critical")
            sys.exit(1)
    else:
        log_and_print(logfile, f"{translations_csv} already exists.")

    # --- Load or Initialize Status ---
    status_path = metadata_folder / STATUS_FILENAME
    status_df = initialize_or_load_status(status_path, translations_csv, logfile)

    # --- Handle --update-settings mode ---
    if args.update_settings:
        vrs_diffs = get_vrs_diffs()
        updated_status_df = update_all_settings(
            status_df.copy(), # Pass a copy to avoid modifying original before save
            projects_folder,
            private_projects_folder,
            vrs_diffs,
            logfile
        )
        
        # Save the updated status file
        try: # Save status_df which was updated within update_all_settings
            updated_status_df.to_csv(status_path, index=False)
            log_and_print(logfile, f"\nSaved updated status after settings update to {status_path}")
        except Exception as e:
            log_and_print(logfile, f"Error saving status file {status_path} after settings update: {e}", log_type="Error")
        # Print SILNLP commands and exit
        print_silnlp_commands(logs_folder, log_suffix, private_projects_folder, private_corpus_folder, projects_folder, corpus_folder, logfile)
        sys.exit(0)

    # --- Scan existing folders to update status if necessary---
    status_df = scan_and_update_status(
        status_df,
        downloads_folder,
        projects_folder,
        private_projects_folder,
        corpus_folder,
        private_corpus_folder,
        logfile
    )

    # --- Ensure extract paths are calculated for reporting ---
    status_df = ensure_extract_paths(
        status_df,
        corpus_folder,
        private_corpus_folder,
        logfile
    )

    # --- Filter Translations ---
    filtered_df = filter_translations(
        status_df,
        args.allow_non_redistributable,
        args.verse_threshold,
        args.filter,
        logfile
    )

    if filtered_df.empty:
        log_and_print(logfile, "No translations match the specified filters. Exiting.")
        # Save status file even if empty? Maybe not necessary.
        # filtered_df.to_csv(status_path, index=False)
        sys.exit(0)

    # --- Determine Actions ---
    actions_df = determine_actions(
        filtered_df, max_age_days, args.force_download,
        downloads_folder, projects_folder, private_projects_folder
    )

    # --- Execute Actions ---
    eBible_url: str = r"https://ebible.org/Scriptures/"

    # Download
    actions_df = download_required_files(actions_df, eBible_url, downloads_folder, logfile)

    # # Option to stop after download (re-evaluate if needed)
    # if args.download_only:
    #     log_and_print(logfile, "Stopping after download phase as requested.")
    #     # Save status now
    #     status_df.update(actions_df) # Update the main df with changes
    #     status_df.to_csv(status_path, index=False)
    #     log_and_print(logfile, f"Saved updated status to {status_path}")
    #     sys.exit(0)

    # Unzip, Rename, Settings, Licence
    vrs_diffs_data = get_vrs_diffs()
    actions_df = unzip_and_process_files(
        actions_df, downloads_folder, projects_folder,
        private_projects_folder, vrs_diffs_data, logfile
    )

    # Perform licence checks for existing projects if needed
    actions_df = check_and_update_licences(actions_df, logfile)

    # --- Perform post-extraction renaming ---
    # Run this before saving status, so rename date gets saved
    actions_df = rename_extracted_files(actions_df, corpus_folder, private_corpus_folder, logfile)
    
    # --- Update Main Status DataFrame and Save ---
    # Use update() which aligns on index (translationId if set, otherwise row number)
    # Ensure index is set correctly if needed, or update based on 'translationId' column
    status_df.set_index('translationId', inplace=True, drop=False) # Set index temporarily
    actions_df.set_index('translationId', inplace=True, drop=False)
    status_df.update(actions_df)
    status_df.reset_index(drop=True, inplace=True) # Remove index before saving

    try:
        status_df.to_csv(status_path, index=False)
        log_and_print(logfile, f"\nSaved updated status for {len(status_df)} translations to {status_path}")
    except Exception as e:
        log_and_print(logfile, f"Error saving status file {status_path}: {e}", log_type="Error")

    # --- Perform post-extraction renaming (Run again after save? Maybe not needed if run before save) ---
    # Renaming is now done before saving the main status_df update.

    # --- Report Missing Extracts ---
    # Re-scan folders to update status one last time before reporting
    status_df = scan_and_update_status(status_df, downloads_folder, projects_folder, private_projects_folder, corpus_folder, private_corpus_folder, logfile)

    missing_extracts_df = status_df[status_df['status_extract_date'].isna() & status_df['downloadable'] & ((status_df['OTverses'] + status_df['NTverses']) >= args.verse_threshold)]
    # Apply filters again if needed, or assume we only care about potentially processable ones
    # missing_extracts_df = filter_translations(missing_extracts_df, args.allow_non_redistributable, args.verse_threshold, args.filter, logfile) # Re-filter if strict reporting needed

    if not missing_extracts_df.empty:
        log_and_print(logfile, f"\nWarning: {len(missing_extracts_df)} translations appear to be missing extracted corpus files (.txt):", log_type="Warn")
        for index, row in missing_extracts_df.iterrows():
            log_and_print(logfile, f"  - {row['translationId']}: Expected at {row['status_extract_path']}", log_type="Warn")
    
    # --- Final Info ---
    log_and_print(logfile, "\nLicence Type Summary (Processed Translations):")
    # Filter actions_df for successfully processed ones if needed, or show all filtered
    log_and_print(logfile, actions_df['licence_Licence_Type'].value_counts(dropna=False))

    # --- Print SILNLP Commands ---
    print_silnlp_commands(logs_folder, log_suffix, private_projects_folder, private_corpus_folder, projects_folder, corpus_folder, logfile)

def print_silnlp_commands(logs_folder, log_suffix, private_projects_folder, private_corpus_folder, projects_folder, corpus_folder, logfile):
    # Define extract log paths using the same suffix
    public_extract_log: Path = logs_folder / ("extract_public" + log_suffix)
    private_extract_log: Path = logs_folder / ("extract_private" + log_suffix)    

    log_and_print(
        logfile,
        [
            "\n--- Next Step: Bulk Extraction ---",
            "Use SILNLP's bulk_extract_corpora tool.",
            "Ensure you have SILNLP installed and configured (e.g., via poetry).",
            "\nCommand for PRIVATE projects:",
            f"poetry run python -m silnlp.common.bulk_extract_corpora --input \"{private_projects_folder}\" --output \"{private_corpus_folder}\" --error-log \"{private_extract_log}\"",
            "\nCommand for PUBLIC projects:",
            f"poetry run python -m silnlp.common.bulk_extract_corpora --input \"{projects_folder}\" --output \"{corpus_folder}\" --error-log \"{public_extract_log}\"",
            "\n---------------------------------"
        ],
    )

if __name__ == "__main__":
    main()
