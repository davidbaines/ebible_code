# Phase 1 Todo — Country and Continent Data

## Step 1 — Prepare `assets/country_continent.csv`

- [x] Download the Gist CSV: https://gist.github.com/stevewithington/20a69c0b6d2ff846ea5d35e5fc47f26c
- [x] Inspect columns; confirm `CountryCode` and `ContinentCode` column names
- [x] Trim to keep only `CountryCode` and `ContinentCode` columns, uppercase values
- [x] Save as `assets/country_continent.csv` and commit

## Step 1b — Create `assets/country_code_overrides.csv`

- [x] Create `assets/country_code_overrides.csv` with columns `raw_code`, `iso_code`, `notes`
- [x] Add initial entry: `RP → PH` (legacy Philippines code used by ebible.org)
- [x] Commit alongside other assets

## Step 2 — Write `ebible_code/generate_language_country_continent.py`

- [x] Fetch and parse `https://ebible.org/Scriptures/` HTML (use `requests` + `beautifulsoup4`)
- [x] Extract `(translationId, countryCode)` pairs from `details.php?id=` and `country.php?c=` href patterns
- [x] Skip header row and rows with missing patterns
- [x] Fix: use second `<table>` (data table) not first (nav table)
- [x] Fix: use `keep_default_na=False` to prevent pandas converting "NA" (Namibia) to NaN
- [x] Read `assets/country_continent.csv`, join on `countryCode` to get `continentCode`
- [x] Warn to stderr for any `countryCode` not found in mapping
- [x] Write `assets/language_country_continent.csv` with columns: `translationId`, `countryCode`, `continentCode`
- [x] Run the script; verified spec checks 1–6 (1545 rows, correct columns, no duplicates)
- [x] Apply `assets/country_code_overrides.csv` before continent lookup (silent when file absent)
- [x] Re-run script; `RP` now stored as `PH`/`AS`, no warnings fire for it
- [x] Verify spec check 7: `RP` translation maps to `PH`/`AS`

## Step 3 — Modify `ebible.py`

- [x] Add `ENRICHMENT_COLUMNS = ["countryCode", "continentCode"]` after `LICENCE_COLUMNS`
- [x] Update `ALL_STATUS_COLUMNS` to append `ENRICHMENT_COLUMNS`
- [x] Write `enrich_with_country_data(df, assets_dir)` function (idempotent, warns for missing ids)
- [x] Call `enrich_with_country_data()` in `main()` after `initialize_or_load_status()`, before first `to_csv()`

## Step 4 — Modify `corpus_to_parquet.py`

- [x] Add `"countryCode"` and `"continentCode"` to `METADATA_SOURCE_COLUMNS`

## Step 5 — Tests

- [x] Write `tests/test_phase1.py` with all unit tests listed in spec
- [x] All 14 unit tests pass: `poetry run pytest tests/test_phase1.py`
- [x] No regressions in `test_clean_range_markers.py` or other non-integration tests
- [x] Add `test_override_applied` — `RP` → `PH`/`AS` after override
- [x] Add `test_override_missing_file_ok` — script runs without error when overrides file absent
- [x] Add `test_override_unknown_after_override_warns` — unresolved code still warns
- [x] All 18 tests pass

## Step 6 — Final commit

- [x] Commit `assets/country_continent.csv`
- [x] Commit `assets/language_country_continent.csv`
- [x] Commit `ebible_code/generate_language_country_continent.py`
- [x] Commit changes to `ebible_code/ebible.py`
- [x] Commit changes to `ebible_code/corpus_to_parquet.py`
- [x] Commit `tests/test_phase1.py`
- [x] Commit `assets/country_code_overrides.csv`
- [x] Commit updated `generate_language_country_continent.py` (with override logic)
- [x] Commit updated `tests/test_phase1.py` (with override tests)
- [x] Re-commit `assets/language_country_continent.csv` (regenerated with `RP` fixed)

---

# Phase 2 Todo — Glottolog Language Family Data

## Step 1 — Create `assets/ATTRIBUTION.md`

- [x] Write attribution for Glottolog 5.3 (CC BY 4.0, full Hammarström et al. citation, URL, version)
- [x] Write attribution for country-continent Gist (Steve Withington)
- [x] Write attribution for eBible.org Scriptures table
- [x] Commit `assets/ATTRIBUTION.md`

## Step 2 — Create `assets/macrolanguage_overrides.csv`

- [x] Create file with header: `ebible_language_code,glottolog_lookup_code,notes`
- [x] Check eBible `languageCode` values — eBible already uses specific ISO 639-3 codes; no macrolanguage codes found
- [x] File created with header only (no data rows needed)
- [x] Commit `assets/macrolanguage_overrides.csv`

## Step 3 — Write `ebible_code/get_glottolog_families.py`

- [x] Download Glottolog 5.3 zip and inspect `languoid.csv` structure
- [x] Implement `load_macrolanguage_overrides(path)` — returns `{}` if file absent, else `{ebible_code: glottolog_code}`
- [x] Implement `load_languoids(zip_url)` — downloads zip, reads `languoid.csv` with `keep_default_na=False, dtype=str`
- [x] Implement `build_family_records(df, overrides)` — traces ancestor paths for all language-level rows with ISO codes
- [x] Handle isolates: `family_id` empty → `family_name = "Isolate"`, `classification` = language name only
- [x] Handle first-occurrence-wins for duplicate ISO codes
- [x] Implement `write_glottolog_families(records, output_path)` — writes CSV with four columns
- [x] Wire up `main()` calling all of the above

## Step 4 — Run and verify

- [x] Run `poetry run python ebible_code/get_glottolog_families.py`
- [x] Verify: row count = 7,859 (≥ 7,500)
- [x] Verify: no duplicate `languageCode` values
- [x] Verify spot-checks: `eng` (Indo-European/Germanic/...), `eus` (Isolate), `fra` (Indo-European/.../Romance/...), `arq` (Afro-Asiatic)
- [x] Verify: no warning fires for `arq` (eBible uses specific code, not macrolanguage `ara`)
- [x] Commit `assets/glottolog_families.csv`

## Step 5 — Tests

- [x] Write `tests/test_phase2.py` with all tests listed in spec (15 tests)
- [x] All 15 tests pass: `poetry run pytest tests/test_phase2.py`
- [x] No regressions: all 42 tests pass across test_phase1.py, test_phase2.py, test_clean_range_markers.py

## Step 6 — Commit

- [x] Commit `ebible_code/get_glottolog_families.py`
- [x] Commit `assets/macrolanguage_overrides.csv`
- [x] Commit `assets/ATTRIBUTION.md`
- [x] Commit `tests/test_phase2.py`

---

# Phase 3 Todo — Dataloader Script

## Step 1 — Read parquet files and understand their schemas

- [ ] Read `main.parquet` schema: confirm vref column name, sample translationId column names
- [ ] Read `metadata.parquet` schema: confirm all available columns
- [ ] Note row/column counts for use in tests

## Step 2 — Write core filter logic

- [ ] Implement `parse_filter_args(filters)` — parses `[COLUMN, [OPERATOR], VALUE...]` tuples into filter specs
- [ ] Implement `apply_filters(metadata_df, filter_specs)` — applies AND-combined filters; raises on unknown column
- [ ] Implement `load_custom_filter(file, join_col, metadata_df)` — joins user CSV, validates join column
- [ ] Support all four operators: `is`, `contains`, `not`, `in`
- [ ] Test with sample metadata DataFrame (no parquet I/O needed)

## Step 3 — Write `filter` subcommand

- [ ] Load `metadata.parquet` from `--repo` path
- [ ] Apply all filters and custom filters
- [ ] Print sorted translationId list + summary count
- [ ] Implement `--repo` resolution (local path vs HuggingFace dataset ID)

## Step 4 — Write `load` subcommand

- [ ] Load filtered translationId list → select those columns from `main.parquet`
- [ ] Build text table: `vref` + selected translation columns; empty string for NaN
- [ ] Build metadata table using `--metadata-columns` (default set)
- [ ] Write text table to `--output` in `--output-format` (csv, parquet)
- [ ] Write metadata table to `--metadata-output` (unless `--no-metadata`)
- [ ] Print summary to stderr (translation count, non-empty verse counts)

## Step 5 — Write `split` subcommand

- [ ] Parse `splits.csv` with omission semantics:
  - translationId only → full Bible
  - + book → whole book
  - + book + chapter → whole chapter
  - + all four → specific verse
- [ ] Map parsed split specs to vref rows from `assets/vref.txt`
- [ ] For each split: produce text table (same columns, same rows; out-of-split verses = `""`)
- [ ] For each split: produce metadata table
- [ ] Write split files to `--output-dir`
- [ ] Summary: note translations excluded by filter that appear in splits.csv

## Step 6 — HuggingFace Dataset output

- [ ] Implement `--output-format huggingface`: flat `Dataset` (no splits) or `DatasetDict` (with splits)
- [ ] Implement `--output-format pandas`: return DataFrame (used programmatically, not from CLI)

## Step 7 — Tests

- [ ] Write `tests/test_phase3.py` with all 21 tests from spec
- [ ] All tests pass: `poetry run pytest tests/test_phase3.py`
- [ ] No regressions across test_phase1.py, test_phase2.py, test_clean_range_markers.py

## Step 8 — README update

- [ ] Add `dataloader.py` section to `assets/parquet_README_template.md` documenting:
  - Installation: `pip install pandas pyarrow datasets`
  - Basic usage examples for `filter`, `load`, `split` subcommands
  - `--custom_filter` example with `glottolog_families.csv`
  - Link to `assets/ATTRIBUTION.md`

## Step 9 — Commit

- [ ] Commit `ebible_code/dataloader.py`
- [ ] Commit `tests/test_phase3.py`
- [ ] Commit updated `assets/parquet_README_template.md`
