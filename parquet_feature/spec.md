# Parquet Export — Specification

## Goals

1. Produce `main.parquet`: a wide-format, verse-aligned parallel corpus covering all redistributable, extracted eBible translations.
2. Produce `metadata.parquet`: per-translation metadata for every translation present in `main.parquet`.
3. Produce `README.md` in the output folder: a HuggingFace dataset card generated from a static template with injected statistics.
4. Add `clean_range_markers(file_path)` to `ebible.py`: removes orphaned `<range>` markers in-place from corpus `.txt` files, called after extraction and again during the Parquet validation pass.

No HuggingFace upload. All output files are always overwritten on each run.

---

## Part 1 — `clean_range_markers(file_path)`

### Location
`ebible_code/ebible.py`, as a standalone importable function.

### Algorithm
Single top-to-bottom pass over the lines of a corpus `.txt` file:

```
for i in range(len(lines)):
    if lines[i] == "<range>":
        prev = lines[i - 1] if i > 0 else ""
        if prev == "":
            lines[i] = ""
```

- Line 0 being `<range>` is treated as orphaned (no preceding line → `prev = ""`).
- Processing top-to-bottom makes cascading automatic: once an orphaned `<range>` is replaced with `""`, the next `<range>` in the chain sees `""` as its predecessor and is also replaced.
- The function is idempotent: running it twice on the same file produces the same result.
- Writes back to the same file path in-place (UTF-8, Unix line endings `\n`).
- Returns the number of replacements made (for logging).

### Insertion point in `ebible.py` pipeline
`extract → clean_range_markers → hash`

The hash recorded in `status_extract_hash` must reflect the cleaned file. `clean_range_markers` is called immediately after a successful extraction, before the hash is computed.

### Backfill
Because existing corpus files predate this function, `corpus_to_parquet.py` also calls `clean_range_markers` on every file during its pre-flight validation pass. Since the function is idempotent, re-running it on already-clean files is safe.

---

## Part 2 — `corpus_to_parquet.py` rewrite

### CLI
Uses `argparse`. No positional arguments and no optional flags except `--help`.
`--help` text must state that `HUGGINGFACE_OUTPUT_FOLDER` (and other paths) are configured via `.env`.

### Environment variables (all loaded from `.env`)
| Variable | Purpose |
|---|---|
| `EBIBLE_DATA_DIR` | Root of the data repository |
| `VREF_FILENAME` | Filename of `vref.txt` under `metadata/` |
| `METADATA_FILENAME` | Filename of `ebible_status.csv` under `metadata/` |
| `HUGGINGFACE_OUTPUT_FOLDER` | Output directory for all three output files |
| `HUGGINGFACE_MAIN_PARQUET_FILENAME` | Filename for `main.parquet` |
| `HUGGINGFACE_METADATA_PARQUET_FILENAME` | Filename for `metadata.parquet` |

### Step 1 — Load vref

Read `vref.txt`. Build a list of 41,899 vref strings (`"GEN 1:1"`, ...). Abort with a clear error if the file is missing or empty.

### Step 2 — Load and filter metadata

Read `ebible_status.csv`. Filter to rows where:
- `Redistributable == "True"`
- `status_extract_path` is non-empty and does not contain `"private_corpus"`
- Both `status_extract_date` and `status_extract_hash` are populated

Warn (do not skip silently) for rows where exactly one of those two fields is populated.

### Step 3 — Pre-flight validation pass

For each candidate translation, with a `tqdm` progress bar:

1. Call `clean_range_markers(path)` in-place (backfill).
2. Check the file exists.
3. Check `len(lines) == 41899`.

Collect all failures into a list. After the pass:

- If no failures: proceed.
- If failures exist: print a formatted report listing every failed translation with its reason, then prompt:
  ```
  N file(s) have issues. Continue anyway and skip them? [y/N]:
  ```
  - If the user enters anything other than `y` or `Y`: exit with code 1.
  - If `y`/`Y`: remove the failing translations from the candidate set and continue.

### Step 4 — Build `main.parquet`

- Load each passing translation file into a dict `{translationId: [line, ...]}`.
- Build a DataFrame:
  - Column 1: `vref` — the 41,899 vref strings from `vref.txt`.
  - Remaining columns: one per translation, **sorted alphabetically by `translationId`**.
  - All translation columns have dtype `object` (string).
- Write to `HUGGINGFACE_OUTPUT_FOLDER / HUGGINGFACE_MAIN_PARQUET_FILENAME` using `engine="pyarrow"`, `compression="snappy"`, `index=False`.
- Always overwrite.

### Step 5 — Build `metadata.parquet`

Include only the translations present in `main.parquet`.

**Included columns** (in this order):

*From ORIGINAL_COLUMNS (all 33):*
`languageCode`, `translationId`, `languageName`, `languageNameInEnglish`, `dialect`,
`homeDomain`, `title`, `description`, `Redistributable`, `Copyright`, `UpdateDate`,
`publicationURL`, `OTbooks`, `OTchapters`, `OTverses`, `NTbooks`, `NTchapters`,
`NTverses`, `DCbooks`, `DCchapters`, `DCverses`, `FCBHID`, `Certified`, `inScript`,
`swordName`, `rodCode`, `textDirection`, `downloadable`, `font`, `shortTitle`,
`PODISBN`, `script`, `sourceDate`

*From LICENCE_COLUMNS (7):*
`licence_Vernacular_Title`, `licence_Licence_Type`, `licence_Licence_Version`,
`licence_CC_Licence_Link`, `licence_Copyright_Holder`, `licence_Copyright_Years`,
`licence_Translation_by`

*From STATUS_COLUMNS (1, renamed):*
`status_inferred_versification` → **`inferred_versification`**

Columns absent from `ebible_status.csv` are silently skipped (forward-compatible).

Write to `HUGGINGFACE_OUTPUT_FOLDER / HUGGINGFACE_METADATA_PARQUET_FILENAME` using `engine="pyarrow"`, `compression="snappy"`, `index=False`.

### Step 6 — Generate `README.md`

Read `assets/README_template.md`. Replace placeholders:

| Placeholder | Value |
|---|---|
| `{{TRANSLATION_COUNT}}` | Number of translation columns in `main.parquet` |
| `{{LANGUAGE_COUNT}}` | Count of unique `languageCode` values among included translations |
| `{{VERSE_COUNT}}` | 41899 (constant) |
| `{{GENERATED_DATE}}` | ISO date of the run (`YYYY-MM-DD`) |
| `{{LICENCE_TABLE}}` | Markdown table of `translationId`, `licence_Licence_Type`, `licence_CC_Licence_Link` for all included translations |

Write to `HUGGINGFACE_OUTPUT_FOLDER / "README.md"`. Always overwrite.

### Step 7 — Print summary

```
--- Summary ---
Translations in metadata:       <N>
Candidates (redistributable):   <N>
Pre-flight failures (skipped):  <N>
Included in main.parquet:       <N>
Languages represented:          <N>
Output:                         <path>
```

---

## Part 3 — Bug fixes in existing code

- `corpus_to_parquet.py` line 397: remove `"status_versification"` from the metadata column list (column does not exist).
- `corpus_to_parquet.py` line 410: metadata was being written to `hf_main_parquet_file` instead of `hf_metadata_parquet_file`. Fixed as part of the rewrite.
- `corpus_to_parquet.py` line 410: metadata was written as CSV. Fixed: both output files are now Parquet.

---

## Verification

Each requirement maps to at least one test. Tests live in `tests/test_parquet.py` and `tests/test_clean_range_markers.py`.

### `clean_range_markers` tests (`tests/test_clean_range_markers.py`)

| Test | What it proves |
|---|---|
| `test_valid_range_preserved` | A `<range>` preceded by text is not modified |
| `test_orphaned_range_replaced` | A `<range>` preceded by `""` becomes `""` |
| `test_cascade` | Two consecutive orphaned `<range>` lines both become `""` |
| `test_first_line_range` | Line 0 being `<range>` is treated as orphaned |
| `test_empty_file` | File with all empty lines is unchanged |
| `test_no_ranges` | File with no `<range>` markers is unchanged |
| `test_mixed` | Valid and orphaned ranges in same file — only orphaned are replaced |
| `test_idempotent` | Running the function twice produces the same result |
| `test_return_count` | Return value equals the number of replacements made |

Tests use `tmp_path` to create temporary files. No corpus data required.

### `corpus_to_parquet` tests (`tests/test_parquet.py`)

These tests use a small synthetic corpus (3 translations, 5 verses) and a matching synthetic `ebible_status.csv`. No real data files required.

| Test | What it proves |
|---|---|
| `test_vref_is_first_column` | Column 0 of `main.parquet` is named `vref` |
| `test_translations_alphabetical` | Translation columns are sorted alphabetically by ID |
| `test_row_count` | `main.parquet` has exactly as many rows as `vref.txt` |
| `test_empty_and_range_preserved` | `""` and `"<range>"` values survive round-trip to Parquet |
| `test_metadata_only_included_translations` | `metadata.parquet` contains only IDs present in `main.parquet` |
| `test_metadata_columns_order` | `metadata.parquet` has exactly the specified columns in order |
| `test_inferred_versification_renamed` | `status_inferred_versification` appears as `inferred_versification` in `metadata.parquet` |
| `test_readme_placeholders_replaced` | `README.md` contains none of the `{{...}}` placeholder strings |
| `test_readme_translation_count` | `README.md` contains the correct translation count |
| `test_missing_file_skipped` | A translation whose file does not exist is absent from both output files |
| `test_wrong_line_count_skipped` | A translation with wrong line count is absent from both output files |
| `test_clean_range_called_in_validation` | Orphaned ranges in an existing file are cleaned before the file is loaded |
| `test_output_overwritten` | Re-running the script overwrites existing output files |
