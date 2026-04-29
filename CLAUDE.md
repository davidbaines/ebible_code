# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install

# Run the main pipeline
poetry run python ebible_code/ebible.py

# Run with common flags
poetry run python ebible_code/ebible.py --filter 'eng'          # filter by regex
poetry run python ebible_code/ebible.py --update-settings       # only regenerate Settings.xml
poetry run python ebible_code/ebible.py --test                  # use TEST_EBIBLE_DATA_DIR
poetry run python ebible_code/ebible.py --force_download        # re-download everything
poetry run python ebible_code/ebible.py --hash-for-changes      # only update wildebeest_hash

# Convert corpus to Parquet for HuggingFace
poetry run python ebible_code/corpus_to_parquet.py

# Run all tests
poetry run pytest

# Run a specific test file
poetry run pytest tests/test_smoke.py

# Run a single test by name
poetry run pytest tests/test_smoke.py::test_number_of_files
```

## Environment Setup

Copy `.env` and set paths before running:

```
EBIBLE_DATA_DIR=<path to local ebible_data repo>
CORPUS_PATH=<path>/corpus          # used by smoke tests
MAX_AGE_DAYS=365                   # how long before re-downloading
LOG_LEVEL=WARNING                  # DEBUG/INFO/WARNING/ERROR/CRITICAL
TEST_EBIBLE_DATA_DIR=<path>        # used with --test flag
VREF_FILENAME=vref.txt
METADATA_FILENAME=ebible_status.csv
HUGGINGFACE_OUTPUT_FOLDER=huggingface
HUGGINGFACE_MAIN_PARQUET_FILENAME=main.parquet
HUGGINGFACE_METADATA_PARQUET_FILENAME=metadata.parquet
```

The data directory (`EBIBLE_DATA_DIR`) must contain these subdirectories (created automatically on first run if you consent):
`corpus`, `downloads`, `logs`, `metadata`, `private_corpus`, `private_projects`, `projects`

## Architecture

### Pipeline overview

`ebible.py` is the main entry point. It runs a multi-stage pipeline tracked by a status CSV:

1. Download `translations.csv` from eBible.org
2. Load/create `metadata/ebible_status.csv` (the central ledger for all pipeline state)
3. Scan existing folders to populate missing status entries
4. Filter translations (by redistributability, verse count, regex, or test set)
5. Determine which actions are needed based on status dates and `--max-age-days`
6. Download ZIP files → unzip to Paratext project folders
7. Rename USFM files, generate `.vrs` files, write `Settings.xml`
8. Extract licence info from `copr.htm`
9. Extract verse-aligned text to `corpus/<translationId>.txt`
10. Calculate xxhash of extracted files

### Key modules

- **`ebible_code/ebible.py`** — orchestrates the whole pipeline; contains all major pipeline stages as top-level functions (`download_required_files`, `unzip_and_process_files`, `extract_and_finalize_texts`, etc.)

- **`ebible_code/settings_file.py`** — two responsibilities:
  1. `generate_vrs_from_project()`: reads a Paratext project's USFM files and writes a `<projectId>.vrs` file recording max verse numbers per chapter
  2. `write_settings_file()` / `get_versification_with_scoring()`: scores the project's `.vrs` against all standard versifications (from `sil-machine`) and writes `Settings.xml` with the best-matching versification number
  
- **`ebible_code/rename_usfm.py`** — renames `*.usfm` files to the Paratext `NNBBBISO.SFM` convention (e.g. `41MATeng.SFM`); OT book numbers are used as-is, NT/DC numbers are incremented by 1

- **`ebible_code/corpus_to_parquet.py`** — reads `ebible_status.csv` and all redistributable corpus `.txt` files, validates line counts against `vref.txt` (41,899 lines), and writes a wide-format Parquet file (one column per translation)

### Data model

`ebible_status.csv` has three column groups defined in `ebible.py`:
- `ORIGINAL_COLUMNS` — sourced from eBible.org's `translations.csv` (language codes, verse counts, flags)
- `LICENCE_COLUMNS` — parsed from `copr.htm` in each project folder
- `STATUS_COLUMNS` — pipeline state (download/unzip/extract dates and paths, hashes, last error)

### Corpus format

Every corpus file (`<translationId>.txt`) has exactly 41,899 lines, one per verse reference in `assets/vref.txt`. Line N in every file corresponds to the same verse (GEN 1:1 = line 1). Empty lines = missing/untranslated verse. Lines containing `<range>` = verse is part of a range starting at an earlier line.

### External dependency: sil-machine

`sil-machine` (`from machine.corpora import ...`) provides `ParatextTextCorpus`, `UsfmFileTextCorpus`, `create_versification_ref_corpus`, `extract_scripture_corpus`, and the `Versification`/`VersificationType` classes. These are used to read USFM files, align text to a reference versification, and load standard `.vrs` files.

### Tests

- `tests/test_smoke.py` — integration tests that scan the whole corpus directory; requires `CORPUS_PATH` set in `.env` pointing to a populated corpus. These are CPU-intensive and skip gracefully if the corpus path doesn't exist.
- `tests/test_settings_file.py` — unit tests for versification detection; requires `EBIBLE_DATA_DIR/projects/` to contain specific project folders (skips individual test cases if the folder is absent).
