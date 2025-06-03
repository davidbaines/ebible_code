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
6. Perform internal text extraction for projects, updating status.
7. Save the updated ebible_status.csv.
"""

import argparse
import os
import shutil
import logging # Import logging
import sys
from contextlib import ExitStack # Potentially needed for extract_scripture_corpus if used directly

# --- CONFIGURE LOGGING before importing from settings_file ---
log_format = '%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
root_logger = logging.getLogger() # Get root logger
root_logger.setLevel(logging.INFO) # Set minimum level to capture
root_logger.debug("Set DEBUG message level")

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from random import randint
from time import sleep, strftime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import regex
import requests
from bs4 import BeautifulSoup
from machine.corpora import ParatextTextCorpus, create_versification_ref_corpus, extract_scripture_corpus
from dotenv import load_dotenv
from rename_usfm import get_destination_file_from_book
from settings_file import generate_vrs_from_project, write_settings_file 
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


STATUS_COLUMNS = [
    "status_download_path", "status_download_date", "status_unzip_path",
    "status_unzip_date", "status_settings_xml_date", # Settings.xml creation date
    "status_extract_path", "status_extract_date", # For internally extracted and finalized text
    "status_last_error", "status_inferred_versification"
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
        logging.error(f"Error downloading {url}: {e}")
        # Clean up potentially incomplete file
        if file.exists():
            try:
                file.unlink()
            except OSError as unlink_e:
                 logging.error(f"Error removing incomplete download {file}: {unlink_e}")
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


def check_folders_exist(folders: list, base: Path):
    """Checks if required folders exist, prompts to create if missing."""
    missing_folders: List[Path] = [folder for folder in folders if not folder.is_dir()]

    logging.info(f"The base folder is : {base}")

    if missing_folders:
        logging.info(
            f"\nThe following {len(missing_folders)} folders are required but missing:"
        )
        for folder in missing_folders:
            logging.info(folder)

        logging.info(f"\nBase folder check:    {base} ")
        if choose_yes_no("Create missing folders and continue? (Y/N): "):
            make_directories(missing_folders)
            logging.info(f"Created required folders within {base}\n")
        else:
            print("Aborting script.")
            sys.exit() # Use sys.exit for clarity
    else:
        # Log that folders exist
        logging.info(f"All required folders exist in {base}")


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

def initialize_or_load_status(status_path: Path, translations_path: Path) -> pd.DataFrame:
    """Loads the status CSV, or creates it from the translations CSV if it doesn't exist."""
    if status_path.exists():
        logging.info(f"Loading existing status file: {status_path}")
        try:
            status_df = pd.read_csv(status_path, keep_default_na=False, na_values=['']) # Treat empty strings as NA
            # Verify essential columns exist
            if not 'translationId' in status_df.columns:
                raise ValueError("Status file missing 'translationId' column.")

            # Handle migration from old column names for extraction
            if 'status_extract_renamed_date' in status_df.columns and 'status_extract_date' in STATUS_COLUMNS:
                # If new 'status_extract_date' is empty and old 'status_extract_renamed_date' has data, migrate it.
                # This assumes 'status_extract_path' from old system pointed to the SILNLP output.
                # The new 'status_extract_path' will be for the final file.
                # This migration is a bit tricky as paths also change.
                # For simplicity, we'll prioritize new columns. Users might need to reprocess for full path accuracy if migrating.
                logging.info("Detected 'status_extract_renamed_date'. Will prioritize new 'status_extract_date' if populated during scan/processing.")

            # Add any missing columns with default NaN values
            for col in ALL_STATUS_COLUMNS:
                if col not in status_df.columns:
                    logging.info(f"Adding missing column '{col}' to status DataFrame.")
                    status_df[col] = np.nan
            
            # Ensure correct order
            # Filter ALL_STATUS_COLUMNS to only include those actually present in status_df after additions, then reorder
            status_df = status_df[[col for col in ALL_STATUS_COLUMNS if col in status_df.columns]]

        except Exception as e:
            logging.error(f"Error loading status file {status_path}: {e}. Attempting to rebuild.")
            status_path.unlink(missing_ok=True) # Remove corrupted file
            return initialize_or_load_status(status_path, translations_path) # Recurse to rebuild

    else:
        logging.info(f"Status file not found. Creating new one: {status_path}")
        if not translations_path.exists():
            logging.critical(f"Error: translations file missing at {translations_path}. Cannot create status file.")
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
                    logging.warning(f"Column '{col}' not found in {translations_path}")
                    status_df[col] = np.nan # Add as empty column if missing

            # Initialize new status/licence columns with NaN
            for col in STATUS_COLUMNS + LICENCE_COLUMNS:
                status_df[col] = np.nan

            # Ensure translationId is the index for easier merging later
            # status_df.set_index('translationId', inplace=True) # Let's keep it as a column for now

        except Exception as e:
            logging.critical(f"Error creating status file from {translations_path}: {e}")
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
            logging.info(f"Found {len(new_ids)} new translations in {translations_path}. Adding to status.")
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
            logging.warning(f"{len(removed_ids)} translations exist in status but not in upstream {translations_path}. They will be kept but may be outdated.")
            # Optionally, mark them as inactive or remove them:
            # status_df = status_df[~status_df['translationId'].isin(removed_ids)]

    except Exception as e:
        logging.error(f"Error merging upstream changes from {translations_path}: {e}")

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
    private_corpus_folder: Path
) -> pd.DataFrame:
    """Scans data folders to update status DataFrame for entries with missing info."""
    logging.info("Scanning existing data folders to update status file...")
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

        # Scan Corpus (Extracted) - now looks for final <translationId>.txt
        if pd.isna(row['status_extract_date']):
            proj_name = translation_id
            corpus_base = corpus_folder if is_redist else private_corpus_folder
            # Construct final expected extract filename
            expected_extract_filename = f"{proj_name}.txt"
            extract_path = corpus_base / expected_extract_filename
            scan_result = scan_corpus_file(extract_path)
            if scan_result:
                status_df.loc[index, 'status_extract_path'] = scan_result[0]
                status_df.loc[index, 'status_extract_date'] = scan_result[1]
                updated_count += 1

    if updated_count > 0:
        logging.info(f"Scan complete. Updated status for {updated_count} entries based on existing files.")
    else:
        logging.info("Scan complete. No missing status information updated from existing files.")

    return status_df


def ensure_extract_paths(
    status_df: pd.DataFrame,
    corpus_folder: Path,
    private_corpus_folder: Path
) -> pd.DataFrame:
    """Calculates and fills the status_extract_path column if missing."""
    logging.info("Ensuring status_extract_path is populated...")
    for index, row in status_df.iterrows():
        if pd.isna(row['status_extract_path']):
            translation_id = row['translationId']
            is_redist = row['Redistributable'] # Assumes bool
            corpus_base = corpus_folder if is_redist else private_corpus_folder
            final_extract_filename = f"{translation_id}.txt" # Final name
            status_df.loc[index, 'status_extract_path'] = str((corpus_base / final_extract_filename).resolve())
    return status_df


def filter_translations(df: pd.DataFrame, allow_non_redistributable: bool, verse_threshold: int, regex_filter: Optional[str]) -> pd.DataFrame:
    """Filters the DataFrame based on criteria."""
    initial_count = len(df)
    logging.info(f"Initial translations in status file: {initial_count}")

    # 0. Filter out translations with existing errors (auto-skip failed)
    df_with_errors = df[df['status_last_error'].notna() & (df['status_last_error'] != '')]
    if not df_with_errors.empty:
        logging.info(f"Skipping {len(df_with_errors)} translations with pre-existing errors in 'status_last_error'.")
    df = df[df['status_last_error'].isna() | (df['status_last_error'] == '')]

    # 1. Filter by downloadable flag
    df = df[df['downloadable'] == True]
    logging.info(f"Translations after 'downloadable' filter: {len(df)}")

    # 2. Filter by redistributable flag (if applicable)
    if not allow_non_redistributable:
        df = df[df['Redistributable'] == True]
        logging.info(f"Translations after 'Redistributable' filter: {len(df)}")

    # 3. Filter by verse count
    df = df[(df['OTverses'] + df['NTverses']) >= verse_threshold]
    logging.info(f"Translations after verse count filter (>= {verse_threshold}): {len(df)}")

    # 4. Apply regex filter (if provided)
    if regex_filter:
        try:
            df = df[df['translationId'].astype(str).str.match(regex_filter, na=False)]
            logging.info(f"Translations after regex filter ('{regex_filter}'): {len(df)}")
        except regex.error as e:
            logging.error(f"Invalid regex filter '{regex_filter}': {e}. Skipping filter.")

    final_count = len(df)
    logging.info(f"Filtered down to {final_count} translations to process.")
    return df


def determine_actions(df: pd.DataFrame, max_age_days: int, force_download: bool, downloads_folder: Path, projects_folder: Path, private_projects_folder: Path) -> pd.DataFrame:
    """Adds boolean columns indicating required actions."""

    df['action_needed_download'] = False
    df['action_needed_unzip'] = False
    df['action_needed_licence'] = False
    df['action_needed_extract'] = False # New action for internal extraction

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

        # --- Text Extraction Check ---
        needs_extract = False
        if needs_unzip: # If re-unzipping, must re-extract
            needs_extract = True
        elif force_download: # Force implies re-extract too
            needs_extract = True
        elif is_date_older_than(row['status_extract_date'], max_age_days): # Using the new combined date
            needs_extract = True
        elif pd.isna(row['status_extract_path']) or not Path(row['status_extract_path']).exists():
            # Check if final extracted file exists
            needs_extract = True
        
        df.loc[index, 'action_needed_extract'] = needs_extract

    return df


def download_required_files(df: pd.DataFrame, base_url: str, folder: Path) -> pd.DataFrame:
    """Downloads files marked with action_needed_download."""
    translations_to_download = df[df['action_needed_download']]
    count = len(translations_to_download)
    logging.info(f"Attempting to download {count} zip files...")

    downloaded_count = 0
    for index, row in tqdm(translations_to_download.iterrows(), total=count, desc="Downloading"):
        translation_id = row['translationId']
        url = f"{base_url}{translation_id}_usfm.zip"
        # Always use today's date for new downloads
        local_filename = build_zip_filename(translation_id, TODAY_STR)
        local_path = folder / local_filename

        logging.info(f"Downloading {url} to {local_path}")
        if download_url_to_file(url, local_path):
            df.loc[index, 'status_download_path'] = str(local_path.resolve())
            df.loc[index, 'status_download_date'] = TODAY_STR
            df.loc[index, 'status_last_error'] = np.nan # Clear previous error on success
            downloaded_count += 1
            logging.info(f"Success: Saved {url} as {local_path}")
            sleep(randint(1, 3000) / 1000) # Shorter sleep?
        else:
            df.loc[index, 'status_download_path'] = np.nan # Clear path on failure
            df.loc[index, 'status_download_date'] = np.nan # Clear date on failure
            df.loc[index, 'status_last_error'] = f"Download failed: {url}"
            logging.error(f"Failed: Could not download {url}")

    logging.info(f"Finished downloading. Successfully downloaded {downloaded_count}/{count} files.")
    return df


def unzip_and_process_files(df: pd.DataFrame, downloads_folder: Path, projects_folder: Path, private_projects_folder: Path) -> pd.DataFrame:
    """Unzips, renames, creates settings, and extracts licence for required projects."""
    translations_to_unzip = df[df['action_needed_unzip']]
    count = len(translations_to_unzip)
    logging.info(f"Attempting to unzip and process {count} projects...")

    processed_count = 0
    for index, row in tqdm(translations_to_unzip.iterrows(), total=count, desc="Unzipping/Processing"):
        translation_id = row['translationId']
        lang_code = row['languageCode']
        is_redist = row['Redistributable']
        download_path_str = row['status_download_path']

        if pd.isna(download_path_str):
            logging.warning(f"Skipping unzip for {translation_id}: No valid download path found.")
            df.loc[index, 'status_last_error'] = "Unzip skipped: Missing download path"
            continue

        download_path = Path(download_path_str)
        if not download_path.exists():
            logging.warning(f"Skipping unzip for {translation_id}: Download path {download_path} not found.")
            df.loc[index, 'status_last_error'] = f"Unzip skipped: Download not found at {download_path}"
            continue

        unzip_base_dir = projects_folder if is_redist else private_projects_folder
        proj_name = translation_id
        project_dir = unzip_base_dir / proj_name

        logging.info(f"Processing {translation_id}: Unzipping {download_path} to {project_dir}")

        # --- Unzip ---
        try:
            # Clean existing directory before unzipping
            if project_dir.exists():
                logging.info(f"Removing existing directory: {project_dir}")
                shutil.rmtree(project_dir)
            project_dir.mkdir(parents=True, exist_ok=True)

            shutil.unpack_archive(download_path, project_dir)
            df.loc[index, 'status_unzip_path'] = str(project_dir.resolve())
            df.loc[index, 'status_unzip_date'] = TODAY_STR
            df.loc[index, 'status_last_error'] = np.nan # Clear error on successful unzip
            logging.info(f"Successfully unzipped to {project_dir}")

            # --- Post-Unzip Processing ---
            # Rename USFM files
            rename_usfm(project_dir)

            # Generate project-specific .vrs file before writing Settings.xml
            try:
                logging.info(f"Generating project .vrs file for {translation_id}")
                generate_vrs_from_project(project_dir)
                # df.loc[index, "status_project_vrs_generated_date"] = TODAY_STR # This status column might be removed later
            except Exception as e_vrs:
                logging.error(f"Error generating project .vrs for {translation_id}: {e_vrs}")
                df.loc[index, "status_last_error"] = f"Project VRS generation failed: {e_vrs}"
                # Continue to attempt settings file write; scoring will use fallback or default.

            # Write Settings.xml
            # Unpack all return values, even if old/new dicts aren't used here yet
            settings_path, vrs_num, _, _ = write_settings_file(project_dir, lang_code)
            df.loc[index, 'status_inferred_versification'] = vrs_num # Store the inferred versification number
            if settings_path: df.loc[index, 'status_settings_xml_date'] = TODAY_STR

            # Extract Licence Details (only if needed or forced)
            if row['action_needed_licence']:
                df = get_and_update_licence_details(df, index, project_dir)

            processed_count += 1

        except (shutil.ReadError, FileNotFoundError, OSError) as e: # Catch more specific IO/OS errors
            logging.error(f"Error processing {translation_id} at {project_dir}: {e}")
            df.loc[index, 'status_unzip_path'] = np.nan
            df.loc[index, 'status_unzip_date'] = np.nan
            df.loc[index, 'status_last_error'] = f"Processing error: {e}"
            # Clean up potentially corrupted unzip dir
            if project_dir.exists():
                try:
                    shutil.rmtree(project_dir)
                except OSError as rm_e:
                    logging.warning(f"Could not remove failed unzip dir {project_dir}: {rm_e}")

    logging.info(f"Finished processing. Successfully processed {processed_count}/{count} projects.")
    return df


def rename_usfm(project_dir: Path):
    """Renames USFM files within the project directory."""
    logging.info(f"Renaming USFM files in {project_dir}")
    renamed_count = 0
    try:
        usfm_paths = list(project_dir.glob("*.usfm"))
        for old_usfm_path in usfm_paths:
            new_sfm_path = get_destination_file_from_book(old_usfm_path)
            if new_sfm_path == old_usfm_path:
                continue
            if new_sfm_path.is_file():
                new_sfm_path.unlink()

            logging.debug(f"Renaming {old_usfm_path.name} to {new_sfm_path.name}") # Use debug level
            old_usfm_path.rename(new_sfm_path)
            renamed_count += 1
        if renamed_count > 0:
            logging.info(f"Renamed {renamed_count} USFM files.")
    except Exception as e:
        logging.error(f"Error renaming USFM files in {project_dir}: {e}")


def get_and_update_licence_details(df: pd.DataFrame, index, project_dir: Path) -> pd.DataFrame:
    """Extracts licence details from copr.htm and updates the DataFrame row."""
    copyright_path = project_dir / "copr.htm"
    logging.debug(f"Extracting licence info for {project_dir.name} from {copyright_path}")

    # Clear previous licence data for this row first
    for col in LICENCE_COLUMNS:
        if col != 'licence_date_read': # Keep date read until success or new attempt
            df.loc[index, col] = np.nan

    if not copyright_path.exists():
        logging.warning(f"Unable to find {copyright_path}")
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
            logging.warning(f"Warning: No body or paragraph tag found in {copyright_path}")


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
                logging.warning(f"Licence key '{col_suffix}' not a column in DataFrame.")

        df.loc[index, 'licence_date_read'] = TODAY_STR
        df.loc[index, 'status_last_error'] = np.nan # Clear error on success
        logging.debug(f"Successfully extracted licence info for {project_dir.name}")

    except Exception as e:
        logging.error(f"Error parsing licence file {copyright_path}: {e}")
        df.loc[index, 'status_last_error'] = f"Licence parse error: {e}"
        df.loc[index, 'licence_date_read'] = TODAY_STR # Mark as checked today, even if failed

    return df


def check_and_update_licences(df: pd.DataFrame) -> pd.DataFrame:
    """Checks and updates licence details for projects that weren't re-unzipped but need a licence check."""
    # Filter for rows needing licence check BUT NOT unzip (as unzip handles its own licence check)
    licence_check_candidates = df[df['action_needed_licence'] & ~df['action_needed_unzip']]
    count = len(licence_check_candidates)

    if count == 0:
        logging.info(f"No existing projects require a separate licence check.")
        return df

    logging.info(f"Performing licence check for {count} existing projects...")
    checked_count = 0
    for index, row in tqdm(licence_check_candidates.iterrows(), total=count, desc="Checking Licences"):
        project_path_str = row['status_unzip_path']

        if pd.isna(project_path_str):
            logging.warning(f"Skipping licence check for {row['translationId']}: Missing unzip path.")
            continue

        project_dir = Path(project_path_str)
        if project_dir.is_dir():
            df = get_and_update_licence_details(df, index, project_dir)
            checked_count += 1
        else:
            logging.warning(f"Skipping licence check for {row['translationId']}: Project directory {project_dir} not found.")

    logging.info(f"Finished separate licence check. Updated {checked_count}/{count} projects.")
    return df


def extract_project( # Renamed from _perform_text_extraction_for_project
    project_dir_path: Path,
    output_file_path: Path,
    translation_id: str # For logging
) -> tuple[bool, Optional[str], int]: # Returns: Success_flag, error_message, segment_count
    """
    Extracts text from a Paratext project directory to a specified output file.
    Assumes Settings.xml is already present in project_dir_path.
    """
    logging.debug(f"Starting text extraction for {translation_id} from {project_dir_path} to {output_file_path}")
    try:
        # ParatextTextCorpus uses Settings.xml within project_dir_path for versification.
        # include_markers=False is the default for typical parallel corpus text.
        project_corpus = ParatextTextCorpus(str(project_dir_path), include_markers=False)
        
        # ref_corpus is based on the standard vref.txt that machine.corpora knows.
        # This ensures alignment to a common verse reference scheme.
        ref_corpus = create_versification_ref_corpus()

        # The user wants all books, so no filtering logic for include_books/exclude_books is applied here.

        segment_count = 0
        # extract_scripture_corpus is a context manager.
        with output_file_path.open("w", encoding="utf-8", newline="\n") as output_stream, \
             extract_scripture_corpus(project_corpus, ref_corpus) as output_lines_iterator:
            
            for line_content, _ref_vref, _project_vref in output_lines_iterator:
                output_stream.write(line_content + "\n")
                segment_count += 1
        
        if segment_count == 0:
            logging.warning(f"Extraction for {translation_id} resulted in 0 segments. Output file {output_file_path} created but is empty.")

        logging.debug(f"Successfully extracted {segment_count} segments for {translation_id} to {output_file_path}")
        return True, None, segment_count

    except Exception as e:
        logging.error(f"Error during text extraction for {translation_id} from {project_dir_path} to {output_file_path}: {e}", exc_info=True)
        if output_file_path.exists():
            try:
                output_file_path.unlink()
                logging.info(f"Removed incomplete output file: {output_file_path}")
            except OSError as unlink_e:
                logging.error(f"Error removing incomplete extraction output {output_file_path}: {unlink_e}")
        return False, str(e), 0


def extract_and_finalize_texts(
    df: pd.DataFrame,
    corpus_folder: Path,
    private_corpus_folder: Path
) -> pd.DataFrame:
    """
    Performs internal text extraction for projects marked with 'action_needed_extract'.
    """
    translations_to_extract = df[df['action_needed_extract']]
    count = len(translations_to_extract)
    logging.info(f"Attempting to extract text for {count} projects...")

    extracted_count = 0
    # Assign tqdm iterator to a variable to allow dynamic description updates
    pbar = tqdm(translations_to_extract.iterrows(), total=count, desc="Initializing extraction...")
    for index, row in pbar:
        translation_id = row['translationId']
        project_unzip_path_str = row['status_unzip_path']
        is_redist = row['Redistributable']

        # Default display names for tqdm, in case of early skip
        project_display_name = translation_id
        output_display_name = f"{translation_id}.txt (target)"

        if pd.isna(project_unzip_path_str):
            # Update description for skipped item
            pbar.set_description_str(f"Skipping {translation_id} (no unzip_path)")
            logging.warning(f"Skipping extraction for {translation_id}: No valid unzip path.")
            df.loc[index, 'status_last_error'] = "Extraction skipped: Missing unzip path"
            continue
        
        project_dir_path = Path(project_unzip_path_str)
        project_display_name = project_dir_path.name # Actual project folder name

        if not project_dir_path.is_dir():
            pbar.set_description_str(f"Skipping {project_display_name} (project dir not found)")
            logging.warning(f"Skipping extraction for {translation_id}: Project directory not found at {project_dir_path}.")
            df.loc[index, 'status_last_error'] = f"Extraction skipped: Project dir not found at {project_dir_path}"
            continue

        target_corpus_folder = corpus_folder if is_redist else private_corpus_folder
        # Output filename is now directly the final name, e.g., KJV.txt
        output_file_path = target_corpus_folder / f"{translation_id}.txt"
        output_display_name = output_file_path.name # Actual output file name

        # Set the description for the current item being actively processed
        pbar.set_description_str(f"Extracting {project_display_name} -> {output_display_name}")
        
        success, error_msg, _segment_count = extract_project( # Call the renamed function
            project_dir_path, output_file_path, translation_id
        )

        if success:
            df.loc[index, 'status_extract_path'] = str(output_file_path.resolve())
            df.loc[index, 'status_extract_date'] = TODAY_STR
            df.loc[index, 'status_last_error'] = np.nan # Clear error on success
            extracted_count +=1
        else:
            df.loc[index, 'status_extract_path'] = np.nan
            df.loc[index, 'status_extract_date'] = np.nan
            df.loc[index, 'status_last_error'] = f"Extraction failed: {error_msg}"

    logging.info(f"Finished text extraction. Successfully extracted {extracted_count}/{count} projects.")
    return df


# --- Function for --update-settings mode ---

def update_all_settings(
    status_df: pd.DataFrame,
    projects_folder: Path,
    private_projects_folder: Path,
    report_path: Path
) -> pd.DataFrame:
    """
    Iterates through project folders, regenerates Settings.xml, and updates status_df.
    """
    logging.info("--- Running in --update-settings mode ---")
    settings_report_data = [] # List to store data for the CSV report
    processed_folders = 0

    # Ensure status_df has translationId as index for quick lookup
    status_df_indexed = status_df.set_index('translationId', drop=False)

    for base_folder in [projects_folder, private_projects_folder]:
        logging.info(f"Scanning folder: {base_folder}")
        if not base_folder.is_dir():
            logging.warning(f"Folder not found, skipping: {base_folder}")
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
                        logging.warning(f"Skipping {translation_id}: Missing languageCode in status file.")
                        continue

                    # Ensure project .vrs exists before updating settings with scoring
                    project_vrs_file = project_dir / f"{translation_id}.vrs"
                    if not project_vrs_file.is_file():
                        logging.info(f"Generating missing project .vrs file for {translation_id} before settings update.")
                        try:
                            generate_vrs_from_project(project_dir)
                            # status_df.loc[status_df["translationId"] == translation_id, "status_project_vrs_generated_date"] = TODAY_STR # type: ignore
                        except Exception as e_vrs_update:
                            logging.error(f"Error generating project .vrs for {translation_id} during settings update: {e_vrs_update}")
                            # Continue, write_settings_file might handle missing .vrs with a default

                    logging.info(f"Updating settings for {translation_id} in {project_dir}")
                    settings_path, vrs_num, old_vals, new_vals = write_settings_file(project_dir, lang_code)

                    if settings_path:
                        # Use the original DataFrame and integer index to update
                        status_df.loc[status_df['translationId'] == translation_id, 'status_inferred_versification'] = vrs_num
                        status_df.loc[status_df['translationId'] == translation_id, 'status_settings_xml_date'] = TODAY_STR

                        # Add data to report
                        report_entry = {
                            "translationId": translation_id,
                            "settings_path": str(settings_path.resolve()),
                            **old_vals,
                            **new_vals,
                        }
                        settings_report_data.append(report_entry)
                    else:
                        logging.error(f"Failed to write settings for {translation_id}")
                        status_df.loc[status_df['translationId'] == translation_id, 'status_last_error'] = "Settings update failed"
                else:
                    logging.warning(f"Skipping {project_dir}: No entry found in status file for translationId '{translation_id}'.")

    # --- Write the settings update report ---
    if settings_report_data:
        report_df = pd.DataFrame(settings_report_data)
        try:
            report_df.to_csv(report_path, index=False, encoding='utf-8')
            logging.info(f"Saved settings update report to {report_path}")
        except Exception as e:
            logging.error(f"Error saving settings update report to {report_path}: {e}")

    logging.info(f"--- Settings update complete. Processed {processed_folders} potential project folders. ---")
    return status_df



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
        raise RuntimeError("Can't determine the location of the eBible Data folder.")
        
    # --- Define Paths ---
    corpus_folder: Path = base / "corpus"
    downloads_folder: Path = base / "downloads"
    private_corpus_folder: Path = base / "private_corpus"
    private_projects_folder: Path = base / "private_projects"
    projects_folder: Path = base / "projects"
    metadata_folder: Path = base / "metadata"
    logs_folder: Path = base / "logs" # Define logs_folder earlier
    
    # --- Setup Logging ---
    logs_folder.mkdir(parents=True, exist_ok=True) # Ensure log dir exists first
    year, month, day, hour, minute = map(int, strftime("%Y %m %d %H %M").split())
    log_suffix: str = f"_{year}_{month:02d}_{day:02d}-{hour:02d}_{minute:02d}"
    log_filename: str = f"ebible_status{log_suffix}.log"
    logfile: Path = logs_folder / log_filename
    logging.info(f"Logging to: {logfile}")


    # --- Check Folders ---
    required_folders = [
        corpus_folder, downloads_folder, private_corpus_folder,
        private_projects_folder, projects_folder, metadata_folder, logs_folder
    ]
    check_folders_exist(required_folders, base)

    # --- Determine Max Age ---
    max_age_days = args.max_age_days
    if max_age_days is None:
        env_max_age = os.getenv("MAX_AGE_DAYS")
        if env_max_age and env_max_age.isdigit():
            max_age_days = int(env_max_age)
            logging.info(f"Using MAX_AGE_DAYS={max_age_days} from .env file.")
        else:
            max_age_days = 365 # Hardcoded default
            logging.info(f"Using default max_age_days={max_age_days}.")
    else:
        logging.info(f"Using --max-age-days={max_age_days} from command line.")

    # --- Download translations.csv if needed ---
    translations_csv_url: str = r"https://ebible.org/Scriptures/translations.csv"
    translations_csv: Path = metadata_folder / TRANSLATIONS_FILENAME
    if not translations_csv.is_file() or args.force_download:
        logging.info(f"Downloading {translations_csv_url} to {translations_csv}")
        if not download_url_to_file(translations_csv_url, translations_csv):
            logging.critical(f"Failed to download {translations_csv}. Aborting.")
            sys.exit(1)
    else:
        logging.info(f"{translations_csv} already exists.")

    # --- Load or Initialize Status ---
    status_path = metadata_folder / STATUS_FILENAME
    status_df = initialize_or_load_status(status_path, translations_csv)

    # --- Handle --update-settings mode ---
    if args.update_settings:
        report_path = metadata_folder / "settings_update.csv"
        updated_status_df = update_all_settings(
            status_df.copy(), # Pass a copy to avoid modifying original before save
            projects_folder,
            private_projects_folder,
            report_path
        )
        
        # Save the updated status file
        try: # Save status_df which was updated within update_all_settings
            updated_status_df.to_csv(status_path, index=False)
            logging.info(f"\nSaved updated status after settings update to {status_path}")
        except Exception as e:
            logging.error(f"Error saving status file {status_path} after settings update: {e}")
        # Print SILNLP commands and exit
        logging.info("Settings update mode finished. No further extraction commands needed as extraction is internal.")
        sys.exit(0)

    # --- Scan existing folders to update status if necessary---
    status_df = scan_and_update_status(
        status_df,
        downloads_folder,
        projects_folder,
        private_projects_folder,
        corpus_folder,
        private_corpus_folder
    )

    # --- Ensure extract paths are calculated for reporting ---
    status_df = ensure_extract_paths(
        status_df,
        corpus_folder,
        private_corpus_folder
    )

    # --- Filter Translations ---
    filtered_df = filter_translations(
        status_df,
        args.allow_non_redistributable,
        args.verse_threshold,
        args.filter
    )

    if filtered_df.empty:
        logging.info("No translations match the specified filters. Exiting.")
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
    actions_df = download_required_files(actions_df, eBible_url, downloads_folder)

    # Option to stop after download (re-evaluate if needed)
    if args.download_only:
        logging.info(f"Stopping after download phase as requested.")
        # Save status now
        status_df.update(actions_df) # Update the main df with changes
        status_df.to_csv(status_path, index=False)
        logging.info(f"Saved updated status to {status_path}")
        sys.exit(0)

    # Unzip, Rename, Settings, Licence
    actions_df = unzip_and_process_files(
        actions_df, downloads_folder, projects_folder,
        private_projects_folder
    )

    # Perform licence checks for existing projects if needed
    actions_df = check_and_update_licences(actions_df)

    # --- Perform Internal Text Extraction ---
    actions_df = extract_and_finalize_texts(
        actions_df,
        corpus_folder,
        private_corpus_folder
    )

    # --- Update Main Status DataFrame and Save ---
    # Use update() which aligns on index (translationId if set, otherwise row number)
    # Ensure index is set correctly if needed, or update based on 'translationId' column
    status_df.set_index('translationId', inplace=True, drop=False) # Set index temporarily
    actions_df.set_index('translationId', inplace=True, drop=False)
    status_df.update(actions_df)
    status_df.reset_index(drop=True, inplace=True) # Remove index before saving

    try:
        status_df.to_csv(status_path, index=False)
        logging.info(f"\nSaved updated status for {len(status_df)} translations to {status_path}")
    except Exception as e:
        logging.error(f"Error saving status file {status_path}: {e}")

    # --- Report Missing Extracts ---
    # Re-scan folders to update status one last time before reporting
    status_df = scan_and_update_status(status_df, downloads_folder, projects_folder, private_projects_folder, corpus_folder, private_corpus_folder)

    missing_extracts_df = status_df[status_df['status_extract_date'].isna() & status_df['downloadable'] & ((status_df['OTverses'] + status_df['NTverses']) >= args.verse_threshold)]
    # Apply filters again if needed, or assume we only care about potentially processable ones
    # missing_extracts_df = filter_translations(missing_extracts_df, args.allow_non_redistributable, args.verse_threshold, args.filter, ) # Re-filter if strict reporting needed

    if not missing_extracts_df.empty:
        logging.warning(f"\n{len(missing_extracts_df)} translations appear to be missing extracted corpus files (.txt):")
        for index, row in missing_extracts_df.iterrows():
            logging.warning(f"  - {row['translationId']}: Expected at {row['status_extract_path']}")
    
    # --- Final Info ---
    logging.info("\nLicence Type Summary (Processed Translations):")
    # Filter actions_df for successfully processed ones if needed, or show all filtered
    # Handle pandas Series logging
    licence_counts = actions_df['licence_Licence_Type'].value_counts(dropna=False)
    logging.info(f"\n{licence_counts.to_string()}")

    logging.info("\n--- ebible.py processing finished ---")


if __name__ == "__main__":
    main()