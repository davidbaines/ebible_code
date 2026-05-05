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
