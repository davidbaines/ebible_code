# Phase 1 Todo — Country and Continent Data

## Step 1 — Prepare `assets/country_continent.csv`

- [x] Download the Gist CSV: https://gist.github.com/stevewithington/20a69c0b6d2ff846ea5d35e5fc47f26c
- [x] Inspect columns; confirm `CountryCode` and `ContinentCode` column names
- [x] Trim to keep only `CountryCode` and `ContinentCode` columns, uppercase values
- [x] Save as `assets/country_continent.csv` and commit

## Step 2 — Write `ebible_code/generate_language_country_continent.py`

- [x] Fetch and parse `https://ebible.org/Scriptures/` HTML (use `requests` + `beautifulsoup4`)
- [x] Extract `(translationId, countryCode)` pairs from `details.php?id=` and `country.php?c=` href patterns
- [x] Skip header row and rows with missing patterns
- [x] Read `assets/country_continent.csv`, join on `countryCode` to get `continentCode`
- [x] Warn to stderr for any `countryCode` not found in mapping (one known: `RP`, legacy Philippines code)
- [x] Write `assets/language_country_continent.csv` with columns: `translationId`, `countryCode`, `continentCode`
- [x] Run the script; verified spec checks 1–6 (1545 rows, correct columns, no duplicates, 1 known bad continent from `RP`)
- [x] Fix: use `keep_default_na=False` to prevent pandas converting "NA" (Namibia) to NaN
- [x] Fix: use second `<table>` (data table) not first (nav table)

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

## Step 6 — Final commit

- [ ] Commit `assets/country_continent.csv`
- [ ] Commit `assets/language_country_continent.csv`
- [ ] Commit `ebible_code/generate_language_country_continent.py`
- [ ] Commit changes to `ebible_code/ebible.py`
- [ ] Commit changes to `ebible_code/corpus_to_parquet.py`
- [ ] Commit `tests/test_phase1.py`
