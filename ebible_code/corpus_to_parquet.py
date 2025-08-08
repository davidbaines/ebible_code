import os
import re
import sys
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import pyarrow
from tqdm import tqdm

# --- Script Logic ---


def parse_vref(vref_line):
    """Parses a vref line (e.g., 'GEN 1:1') into book, chapter, verse."""
    match = re.match(r"(\w{3})\s+(\d{1,3}):(\d{1,3})", vref_line)
    if match:
        return match.groups()  # Returns (book, chapter, verse) as strings
    else:
        # Handle potential malformed lines if necessary, or return None/raise error
        print(f"Warning: Could not parse vref line: '{vref_line}'", file=sys.stderr)
        return None, None, None


def main():
    """Main function to prepare the Hugging Face dataset."""

    # 1. Get EBIBLE_DATA_DIR environment variable
    load_dotenv()

    ebible_data_dir_str = os.getenv("EBIBLE_DATA_DIR")
    if not ebible_data_dir_str:
        print(
            "Error: Environment variable EBIBLE_DATA_DIR is not set.", file=sys.stderr
        )
        print(
            "Please set it to the path of your ebible_data repository.", file=sys.stderr
        )
        sys.exit(1)

    ebible_data_dir = Path(ebible_data_dir_str)
    if not ebible_data_dir.is_dir():
        print(
            f"Error: EBIBLE_DATA_DIR path does not exist or is not a directory: {ebible_data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Construct necessary paths
    vref_path = ebible_data_dir / "metadata" / os.getenv("VREF_FILENAME")
    metadata_path = ebible_data_dir / "metadata" / os.getenv("METADATA_FILENAME")

    hf_output_dir = ebible_data_dir / (os.getenv("HUGGINGFACE_OUTPUT_FOLDER"))

    if not hf_output_dir.is_dir():
        print(f"Error: Could not find HUGGINGFACE_OUTPUT_FOLDER: {hf_output_dir}.")
        sys.exit(1)

    hf_main_parquet_file = hf_output_dir / os.getenv(
        "HUGGINGFACE_MAIN_PARQUET_FILENAME"
    )
    hf_metadata_parquet_file = hf_output_dir / os.getenv(
        "HUGGINGFACE_METADATA_PARQUET_FILENAME"
    )

    print(f"Using EBIBLE_DATA_DIR:           {ebible_data_dir}")
    print(f"Reading vref from:               {vref_path}")
    print(f"Reading metadata from:           {metadata_path}")
    print(f"Output folder for parquet files: {hf_output_dir}")

    # 3. Load and parse vref.txt
    print("\n--- Loading vref.txt ---")
    try:
        with open(vref_path, "r", encoding="utf-8") as f_vref:
            vref_lines = [
                line.strip() for line in f_vref if line.strip()
            ]  # Read non-empty lines
        vref_length = len(vref_lines)
        print(f"Successfully read {vref_length} verse references.")

        if vref_length == 0:
            print(
                "Error: vref.txt is empty or contains only whitespace.", file=sys.stderr
            )
            sys.exit(1)

        # Parse vref lines into components
        parsed_vrefs = [parse_vref(line) for line in vref_lines]
        books, chapters, verses = zip(
            *[(b, c, v) for b, c, v in parsed_vrefs if b is not None]
        )  # Filter out failed parses

        # Check if parsing failed for some lines
        if len(books) != vref_length:
            print(
                f"Warning: {vref_length - len(books)} vref lines could not be parsed.",
                file=sys.stderr,
            )
            # Decide if this is critical - for now, we proceed with parsed lines
            # If strict matching is needed, you might want to exit here.

        # Create the initial DataFrame
        df = pd.DataFrame(
            {
                "book": list(books),
                "chapter": pd.to_numeric(chapters),  # Convert chapters to numeric
                "verse": pd.to_numeric(verses),  # Convert verses to numeric
            }
        )
        # Use the original vref_lines as index if needed later, or just rely on row number
        # df.index = vref_lines[:len(df)] # Assign only if lengths match after filtering bad parses

    except FileNotFoundError:
        print(f"Error: vref.txt not found at {vref_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading or parsing {vref_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Load and filter metadata
    print("\n--- Loading Metadata ---")
    try:
        metadata_df = pd.read_csv(
            metadata_path, keep_default_na=False, dtype={"Redistributable": str}
        )  # Read Redistributable as string
        print(f"Read {len(metadata_df)} total metadata entries.")

        # --- Filtering ---
        # 1. Check required columns exist
        required_cols = [
            "translationId",
            "status_extract_path",
            "Redistributable",
            "status_extract_date",
            "status_extract_hash",
        ]
        missing_cols = [col for col in required_cols if col not in metadata_df.columns]
        if missing_cols:
            print(
                f"Error: Metadata CSV missing required columns: {', '.join(missing_cols)}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 2. Filter by Redistributable == "True" (case-sensitive string comparison)
        filtered_df = metadata_df[metadata_df["Redistributable"] == "True"].copy()
        print(f"Found {len(filtered_df)} redistributable translations.")

        # 3. Filter out entries where path contains 'private_corpus'
        # Ensure status_extract_path is treated as string
        filtered_df["status_extract_path"] = filtered_df["status_extract_path"].astype(
            str
        )
        original_count = len(filtered_df)
        filtered_df = filtered_df[
            ~filtered_df["status_extract_path"].str.contains(
                "private_corpus", case=False, na=False
            )
        ]
        print(
            f"Removed {original_count - len(filtered_df)} entries referencing 'private_corpus'."
        )

        # 4. Drop entries with missing or invalid paths (optional but good practice)
        filtered_df = filtered_df.dropna(subset=["status_extract_path"])
        filtered_df = filtered_df[filtered_df["status_extract_path"] != ""]

        # 5. Check extraction completion status
        extraction_warnings = []
        pre_extraction_count = len(filtered_df)

        # Create boolean masks for extraction status
        has_extract_date = filtered_df["status_extract_date"].notna() & (
            filtered_df["status_extract_date"] != ""
        )
        has_extract_hash = filtered_df["status_extract_hash"].notna() & (
            filtered_df["status_extract_hash"] != ""
        )

        # Case 1: Both fields empty - skip silently (not extracted)
        both_empty = ~has_extract_date & ~has_extract_hash

        # Case 2: Only one field populated - add to warnings list
        only_date = has_extract_date & ~has_extract_hash
        only_hash = ~has_extract_date & has_extract_hash

        # Case 3: Both fields populated - proceed with processing
        both_populated = has_extract_date & has_extract_hash

        # Collect warnings for incomplete extraction status
        for idx, row in filtered_df[only_date].iterrows():
            extraction_warnings.append(
                f"For translation {row['translationId']} the ebible_status file contains a status_extract_date without a status_extract_hash"
            )

        for idx, row in filtered_df[only_hash].iterrows():
            extraction_warnings.append(
                f"For translation {row['translationId']} the ebible_status file contains a status_extract_hash without a status_extract_date"
            )

        # Filter to only include translations with complete extraction status
        filtered_df = filtered_df[both_populated].copy()

        print(
            f"After extraction status validation: {len(filtered_df)} translations ready for processing."
        )
        print(f"Silently skipped {sum(both_empty)} translations (not extracted).")
        print(
            f"Found {len(extraction_warnings)} translations with incomplete extraction status."
        )

        # Display specific warning messages for incomplete extraction status
        if extraction_warnings:
            print("\n--- Extraction Status Warnings ---")
            for warning in extraction_warnings:
                print(f"Warning: {warning}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading or filtering {metadata_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Process translation files
    print("\n--- Processing Translation Files ---")
    skipped_translations = []  # List to store info about skipped files
    translation_data = {}  # Dictionary to hold {translation_id: [line1, line2,...]}

    for index, row in tqdm(
        filtered_df.iterrows(), total=len(filtered_df), desc="Translations"
    ):
        translation_id = row["translationId"]
        text_file_path_str = row["status_extract_path"]

        # Handle potential relative paths in metadata - assume relative to EBIBLE_DATA_DIR if not absolute
        text_file_path = Path(text_file_path_str)
        if not text_file_path.is_absolute():
            # This assumes paths like 'corpus/eng-engESV.txt' are relative to ebible_data_dir
            # Adjust if the paths in the CSV are relative to a different base
            text_file_path = ebible_data_dir / text_file_path

        if not text_file_path.exists():
            # This should not happen since we filtered for complete extraction status earlier
            # If we reach here, it means both status_extract_date and status_extract_hash are populated
            # but the file is missing - this indicates a data integrity issue
            print(
                f"Error: Expected extracted file not found for {translation_id} (extraction status indicates complete): {text_file_path}",
                file=sys.stderr,
            )
            skipped_translations.append(
                {
                    "id": translation_id,
                    "path": str(text_file_path),
                    "reason": "Expected file missing (extraction marked complete but file not found)",
                }
            )
            continue

        try:
            with open(text_file_path, "r", encoding="utf-8") as f_text:
                # Read all lines, preserving empty lines and stripping only trailing newline/whitespace
                text_lines = [line.rstrip("\r\n") for line in f_text]

            # --- Line Count Check ---
            if len(text_lines) != vref_length:
                print(
                    f"Warning: Line count mismatch for {translation_id} ({len(text_lines)} lines) vs vref.txt ({vref_length} lines). Skipping.",
                    file=sys.stderr,
                )
                skipped_translations.append(
                    {
                        "id": translation_id,
                        "path": str(text_file_path),
                        "reason": f"Line count mismatch ({len(text_lines)} vs {vref_length})",
                    }
                )
                continue

            # Store data for later concatenation
            # Ensure the column name is valid (pandas might handle some cases, but good to be safe)
            # For Hugging Face, simple IDs are usually fine.
            if translation_id in translation_data:
                # This check might be redundant if translationId is unique in metadata, but safe to keep
                print(
                    f"Warning: Duplicate translationId '{translation_id}' encountered during processing. Overwriting previous data for this ID.",
                    file=sys.stderr,
                )
                # Or potentially add a suffix, e.g., f"{translation_id}_{index}"
            translation_data[translation_id] = text_lines

        except Exception as e:
            print(
                f"Error processing file {text_file_path.name} for {translation_id}: {e}",
                file=sys.stderr,
            )
            skipped_translations.append(
                {
                    "id": translation_id,
                    "path": str(text_file_path),
                    "reason": f"Processing error: {e}",
                }
            )
            # Decide whether to continue or exit on error
            # continue

    # 6. Combine base DataFrame with translation data
    print("\n--- Combining DataFrames ---")
    if translation_data:
        included_translation_ids = list(translation_data.keys())
        translations_df = pd.DataFrame(translation_data)
        df = pd.concat([df, translations_df], axis=1)
        print(f"Added {len(translations_df.columns)} translation columns.")
    else:
        included_translation_ids = []
        print("No translation data was collected.")
    # 7. Save Main Parquet Output
    print("\n--- Saving Parquet File ---")
    if len(df.columns) <= 3:  # Only book, chapter, verse columns exist
        print(
            "Error: No translation data was successfully processed and added.",
            file=sys.stderr,
        )
        print("Please check warnings and input files.", file=sys.stderr)
    else:
        try:
            print(
                f"Writing {len(df)} rows and {len(df.columns)} columns to {hf_main_parquet_file}..."
            )
            df.to_parquet(
                hf_main_parquet_file,
                index=False,
                engine="pyarrow",
                compression="snappy",
            )
            print("Successfully wrote Parquet file.")
        except Exception as e:
            print(
                f"Error writing Parquet file to {hf_main_parquet_file}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    # 8. Prepare and Save Metadata File for Included Translations
    print("\n--- Preparing and Saving Metadata File ---")
    if included_translation_ids:
        # Filter the original metadata_df to include only rows for translations present in the parquet file
        final_metadata_df = metadata_df[
            metadata_df["translationId"].isin(included_translation_ids)
        ].copy()

        # Select relevant columns for the public metadata file (adjust this list as needed)
        relevant_metadata_columns = [
            "languageCode",
            "translationId",
            "languageName",
            "languageNameInEnglish",
            "dialect",
            "homeDomain",
            "title",
            "description",
            "Redistributable",
            "Copyright",
            "UpdateDate",
            "publicationURL",
            "OTbooks",
            "OTchapters",
            "OTverses",
            "NTbooks",
            "NTchapters",
            "NTverses",
            "DCbooks",
            "DCchapters",
            "DCverses",
            "FCBHID",
            "Certified",
            "inScript",
            "swordName",
            "rodCode",
            "textDirection",
            "downloadable",
            "font",
            "shortTitle",
            "PODISBN",
            "script",
            "sourceDate",
            "licence_ID",
            "licence_File",
            "licence_Language",
            "licence_Dialect",
            "licence_Vernacular_Title",
            "licence_Licence_Type",
            "licence_Licence_Version",
            "licence_CC_Licence_Link",
            "licence_Copyright_Holder",
            "licence_Copyright_Years",
            "licence_Translation_by",
            "status_versification",
            "status_inferred_versification",  # Added the new column
        ]
        # Ensure only existing columns are selected
        relevant_metadata_columns = [
            col for col in relevant_metadata_columns if col in final_metadata_df.columns
        ]
        final_metadata_df = final_metadata_df[relevant_metadata_columns]

        try:
            print(
                f"Writing metadata for {len(final_metadata_df)} included translations to {hf_main_parquet_file}..."
            )
            final_metadata_df.to_csv(
                hf_main_parquet_file, index=False, encoding="utf-8"
            )
            print("Successfully wrote metadata CSV file.")
        except Exception as e:
            print(
                f"Error writing metadata CSV file to {hf_main_parquet_file}: {e}",
                file=sys.stderr,
            )
            # Don't exit, the main parquet might still be useful
    else:
        print("Skipping metadata file generation as no translations were included.")

    # 9. Final Report
    print("\n--- Processing Summary ---")
    print(f"Total translations in metadata: {len(metadata_df)}")
    print(
        f"Considered for processing (Redistributable, not private): {len(filtered_df)}"
    )
    processed_count = len(df.columns) - 3  # Subtract book, chapter, verse columns
    print(f"Successfully processed and included: {processed_count}")
    print(f"Skipped translations: {len(skipped_translations)}")

    if skipped_translations:
        print("\nSkipped Translations Report:")
        for item in skipped_translations:
            print(
                f"  - ID: {item['id']}, Reason: {item['reason']}, Path: {item['path']}"
            )

    print("\nPreprocessing finished.")


if __name__ == "__main__":
    main()
