"""Convert eBible corpus .txt files to Parquet format for HuggingFace.

Configuration is via .env — run with --help for details.
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from machine.corpora import create_versification_ref_corpus
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ebible import clean_range_markers

# Columns included in metadata.parquet, in order.
# status_inferred_versification is renamed to inferred_versification on output.
METADATA_SOURCE_COLUMNS = [
    # All ORIGINAL_COLUMNS from ebible_status.csv
    "languageCode", "translationId", "languageName", "languageNameInEnglish", "dialect",
    "homeDomain", "title", "description", "Redistributable", "Copyright", "UpdateDate",
    "publicationURL", "OTbooks", "OTchapters", "OTverses", "NTbooks", "NTchapters",
    "NTverses", "DCbooks", "DCchapters", "DCverses", "FCBHID", "Certified", "inScript",
    "swordName", "rodCode", "textDirection", "downloadable", "font", "shortTitle",
    "PODISBN", "script", "sourceDate",
    # Selected LICENCE_COLUMNS
    "licence_Vernacular_Title", "licence_Licence_Type", "licence_Licence_Version",
    "licence_CC_Licence_Link", "licence_Copyright_Holder", "licence_Copyright_Years",
    "licence_Translation_by",
    # STATUS: versification only
    "status_inferred_versification",
    # ENRICHMENT columns (Phase 1)
    "countryCode",
    "continentCode",
]

def build_vref_list() -> list:
    """Return the canonical list of verse reference strings (e.g. 'GEN 1:1') from machine.corpora."""
    corpus = create_versification_ref_corpus()
    with corpus.get_rows() as rows:
        return [str(row.ref) for row in rows]


def load_and_filter_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load ebible_status.csv, filter to redistributable non-private extracted translations.

    Warns for rows where only one of status_extract_date / status_extract_hash is set.
    Returns rows where both fields are populated and path is not private_corpus.
    """
    df = pd.read_csv(metadata_path, keep_default_na=False, dtype={"Redistributable": str})

    required = ["translationId", "status_extract_path", "Redistributable",
                "status_extract_date", "status_extract_hash"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata CSV missing required columns: {', '.join(missing)}")

    df = df[df["Redistributable"] == "True"].copy()
    df["status_extract_path"] = df["status_extract_path"].astype(str)
    df = df[~df["status_extract_path"].str.contains("private_corpus", case=False, na=False)]
    df = df[df["status_extract_path"].notna() & (df["status_extract_path"] != "")]

    has_date = df["status_extract_date"].notna() & (df["status_extract_date"] != "")
    has_hash = df["status_extract_hash"].notna() & (df["status_extract_hash"] != "")

    for _, row in df[has_date & ~has_hash].iterrows():
        print(f"Warning: {row['translationId']} has status_extract_date but no status_extract_hash",
              file=sys.stderr)
    for _, row in df[~has_date & has_hash].iterrows():
        print(f"Warning: {row['translationId']} has status_extract_hash but no status_extract_date",
              file=sys.stderr)

    return df[has_date & has_hash].copy()


def validate_corpus_files(candidates: pd.DataFrame, ebible_data_dir: Path,
                          vref_length: int,
                          min_lines: int = 400,
                          min_chars: int = 7,
                          _input=input) -> tuple:
    """Clean and validate each candidate corpus file.

    For each file: runs clean_range_markers in-place, then checks:
      1. File exists.
      2. Line count == vref_length.
      3. At least min_lines lines contain >= min_chars characters
         (lines shorter than min_chars are treated as empty for this check).

    Returns (valid_ids, skipped_ids). Prompts the user to continue if there
    are failures; exits with code 1 if they decline.
    """
    valid_ids = []
    issues = []

    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Validating corpus files"):
        tid = row["translationId"]
        path_str = row["status_extract_path"]
        file_path = Path(path_str) if Path(path_str).is_absolute() else ebible_data_dir / path_str

        if not file_path.exists():
            issues.append((tid, f"File not found: {file_path}"))
            continue

        cleaned = clean_range_markers(file_path)
        if cleaned:
            print(f"  Cleaned {cleaned} orphaned <range> marker(s) in {tid}", file=sys.stderr)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f]

        if len(lines) != vref_length:
            issues.append((tid, f"Line count {len(lines)} != expected {vref_length}"))
            continue

        substantive = sum(1 for line in lines if len(line) >= min_chars)
        if substantive < min_lines:
            issues.append((tid, f"Only {substantive} lines with data (minimum {min_lines} required)"))
            continue

        valid_ids.append(tid)

    if issues:
        print(f"\n{len(issues)} file(s) have issues:")
        for tid, reason in issues:
            print(f"  {tid}: {reason}")
        answer = _input(f"\n{len(issues)} file(s) have issues. Continue anyway and skip them? [y/N]: ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            sys.exit(1)

    return valid_ids, [tid for tid, _ in issues]


def load_translation_texts(candidates: pd.DataFrame, valid_ids: list,
                           ebible_data_dir: Path) -> dict:
    """Load the text lines for each valid translation. Returns {translationId: [lines]}."""
    valid_set = set(valid_ids)
    data = {}
    for _, row in tqdm(
        candidates[candidates["translationId"].isin(valid_set)].iterrows(),
        total=len(valid_ids),
        desc="Loading translations",
    ):
        tid = row["translationId"]
        path_str = row["status_extract_path"]
        file_path = Path(path_str) if Path(path_str).is_absolute() else ebible_data_dir / path_str
        with open(file_path, "r", encoding="utf-8") as f:
            data[tid] = [line.rstrip("\r\n") for line in f]
    return data


def build_main_dataframe(vref_list: list, translation_data: dict) -> pd.DataFrame:
    """Build the wide-format main DataFrame.

    First column is 'vref'. Remaining columns are translations sorted
    alphabetically by translationId.
    """
    sorted_ids = sorted(translation_data.keys())
    data = {"vref": vref_list, **{tid: translation_data[tid] for tid in sorted_ids}}
    return pd.DataFrame(data)


def build_metadata_dataframe(full_metadata: pd.DataFrame, included_ids: list) -> pd.DataFrame:
    """Build the metadata DataFrame for included translations only.

    Selects the columns defined in METADATA_SOURCE_COLUMNS (skipping any absent),
    then renames status_inferred_versification -> inferred_versification.
    """
    df = full_metadata[full_metadata["translationId"].isin(set(included_ids))].copy()
    present = [c for c in METADATA_SOURCE_COLUMNS if c in df.columns]
    df = df[present].copy()
    if "status_inferred_versification" in df.columns:
        df = df.rename(columns={"status_inferred_versification": "inferred_versification"})
    return df


def render_readme(template: str, stats: dict) -> str:
    """Replace {{PLACEHOLDER}} markers in template with values from stats dict."""
    result = template
    for key, value in stats.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert eBible corpus .txt files to Parquet format for HuggingFace.\n\n"
            "All paths are configured via .env:\n"
            "  EBIBLE_DATA_DIR              Root of the data repository\n"
            "  METADATA_FILENAME            Filename of ebible_status.csv under metadata/\n"
            "  HUGGINGFACE_OUTPUT_FOLDER    Output directory for all output files\n"
            "  HUGGINGFACE_MAIN_PARQUET_FILENAME     e.g. main.parquet\n"
            "  HUGGINGFACE_METADATA_PARQUET_FILENAME e.g. metadata.parquet\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args()


def main():
    _parse_args()
    load_dotenv()

    ebible_data_dir_str = os.getenv("EBIBLE_DATA_DIR")
    if not ebible_data_dir_str:
        print("Error: EBIBLE_DATA_DIR is not set in .env", file=sys.stderr)
        sys.exit(1)
    ebible_data_dir = Path(ebible_data_dir_str)
    if not ebible_data_dir.is_dir():
        print(f"Error: EBIBLE_DATA_DIR does not exist: {ebible_data_dir}", file=sys.stderr)
        sys.exit(1)

    metadata_path = ebible_data_dir / "metadata" / os.getenv("METADATA_FILENAME", "ebible_status.csv")
    hf_output_dir = ebible_data_dir / os.getenv("HUGGINGFACE_OUTPUT_FOLDER", "huggingface")
    main_parquet_path = hf_output_dir / os.getenv("HUGGINGFACE_MAIN_PARQUET_FILENAME", "main.parquet")
    metadata_parquet_path = hf_output_dir / os.getenv("HUGGINGFACE_METADATA_PARQUET_FILENAME", "metadata.parquet")
    readme_path = hf_output_dir / "README.md"
    template_path = Path(__file__).parent.parent / "assets" / "parquet_README_template.md"

    if not metadata_path.exists():
        print(f"Error: METADATA_FILENAME not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    if not hf_output_dir.is_dir():
        print(f"Error: HUGGINGFACE_OUTPUT_FOLDER does not exist: {hf_output_dir}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Build vref list from machine.corpora
    print("--- Loading vref ---")
    vref_list = build_vref_list()
    print(f"  {len(vref_list)} verse references loaded")

    # Step 2: Load and filter metadata
    print("\n--- Loading metadata ---")
    full_metadata_df = pd.read_csv(metadata_path, keep_default_na=False, dtype={"Redistributable": str})
    total_in_metadata = len(full_metadata_df)
    try:
        candidates = load_and_filter_metadata(metadata_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  {total_in_metadata} total translations, {len(candidates)} candidates (redistributable, extracted)")

    # Step 3: Pre-flight validation
    min_lines = int(os.getenv("MINIMUM_LINES_IN_EXTRACT", "400"))
    min_chars = int(os.getenv("MINIMUM_CHARACTERS_IN_VERSE", "7"))
    print("\n--- Validating corpus files ---")
    valid_ids, skipped_ids = validate_corpus_files(
        candidates, ebible_data_dir, len(vref_list), min_lines, min_chars
    )
    print(f"  {len(valid_ids)} valid, {len(skipped_ids)} skipped")

    if not valid_ids:
        print("Error: No valid translations to include.", file=sys.stderr)
        sys.exit(1)

    # Step 4: Load text and build main.parquet
    print("\n--- Loading translation texts ---")
    translation_data = load_translation_texts(candidates, valid_ids, ebible_data_dir)

    print("\n--- Building main.parquet ---")
    main_df = build_main_dataframe(vref_list, translation_data)
    main_df.to_parquet(main_parquet_path, index=False, engine="pyarrow", compression="snappy")
    print(f"  Written: {main_parquet_path} ({len(main_df)} rows × {len(main_df.columns)} columns)")

    # Step 5: Build metadata.parquet
    print("\n--- Building metadata.parquet ---")
    metadata_df = build_metadata_dataframe(full_metadata_df, valid_ids)
    metadata_df.to_parquet(metadata_parquet_path, index=False, engine="pyarrow", compression="snappy")
    print(f"  Written: {metadata_parquet_path} ({len(metadata_df)} rows)")

    # Step 6: Generate README.md
    print("\n--- Generating README.md ---")
    language_count = metadata_df["languageCode"].nunique() if "languageCode" in metadata_df.columns else "?"

    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        print(f"Warning: README template not found at {template_path}. Writing minimal README.",
              file=sys.stderr)
        template = "# eBible Parallel Corpus\n\nGenerated: {{GENERATED_DATE}}\n"

    readme_content = render_readme(template, {
        "TRANSLATION_COUNT": len(valid_ids),
        "LANGUAGE_COUNT": language_count,
        "VERSE_COUNT": len(vref_list),
        "GENERATED_DATE": date.today().isoformat(),
    })
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"  Written: {readme_path}")

    # Step 7: Summary
    print("\n--- Summary ---")
    print(f"  Translations in metadata:      {total_in_metadata}")
    print(f"  Candidates (redistributable):  {len(candidates)}")
    print(f"  Pre-flight failures (skipped): {len(skipped_ids)}")
    print(f"  Included in main.parquet:      {len(valid_ids)}")
    print(f"  Languages represented:         {language_count}")
    print(f"  Output:                        {hf_output_dir}")


if __name__ == "__main__":
    main()
