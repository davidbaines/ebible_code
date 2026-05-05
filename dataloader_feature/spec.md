# Phase 1 Spec — Country and Continent Data

## Goal

Add `countryCode` (ISO 3166-1 alpha-2) and `continentCode` (two-letter code) to `ebible_status.csv` and `metadata.parquet` for every translation, by scraping `ebible.org/Scriptures` and joining against a committed country→continent reference file.

## Deliverables

| File | Type | Description |
|---|---|---|
| `assets/country_continent.csv` | Static reference (committed) | Country → continent mapping, sourced from GitHub Gist |
| `assets/country_code_overrides.csv` | Static reference (committed) | Maps non-standard/legacy country codes to ISO 3166-1 alpha-2 codes |
| `assets/language_country_continent.csv` | Generated reference (committed) | translationId → countryCode → continentCode |
| `ebible_code/generate_language_country_continent.py` | New script | Scrapes eBible, joins tables, writes the mapping CSV |
| `ebible_code/ebible.py` | Modified | Adds ENRICHMENT_COLUMNS; populates countryCode/continentCode at pipeline runtime |
| `ebible_code/corpus_to_parquet.py` | Modified | Includes countryCode/continentCode in metadata.parquet |
| `tests/test_phase1.py` | New tests | Verifies each piece works |

---

## Data Sources

### 1. ebible.org/Scriptures page

URL: `https://ebible.org/Scriptures/`

The page contains an HTML table. Each data row has:
- **Territory cell** (col 1): `<a href="country.php?c=AL">` → `countryCode = AL`
- **Language cell** (col 2): `<a href="details.php?id=engPEV">` → `translationId = engPEV`

The `id` in `details.php?id=XXX` is the eBible `translationId` as stored in `ebible_status.csv`. It equals the ISO 639-3 `languageCode` only for single-translation languages (e.g. `rup`); `engPEV` confirms these are distinct.

### 2. Country → continent mapping

Source: `https://gist.github.com/stevewithington/20a69c0b6d2ff846ea5d35e5fc47f26c`

Downloaded once and committed as `assets/country_continent.csv`. Columns kept: `CountryCode` (2-letter, uppercase), `ContinentCode` (2-letter, uppercase). Valid continent codes: `AF`, `AN`, `AS`, `EU`, `NA`, `OC`, `SA`.

### 3. Country code overrides

`assets/country_code_overrides.csv` — committed static file, manually maintained. Maps non-standard or legacy country codes (as used by ebible.org) to their correct ISO 3166-1 alpha-2 equivalents, so they resolve against the continent mapping.

| Column | Example | Notes |
|---|---|---|
| `raw_code` | `RP` | Code as scraped from ebible.org |
| `iso_code` | `PH` | Correct ISO 3166-1 alpha-2 code |
| `notes` | `Legacy Philippines code` | Human-readable reason; kept for auditing |

The override is applied **before** the continent lookup. After applying overrides, the stored `countryCode` is the corrected ISO code (not the raw scraped value). Any raw code still unresolved after overrides triggers a warning — new unknown codes are still surfaced.

---

## New Script: `ebible_code/generate_language_country_continent.py`

### Inputs
- Live HTTP fetch of `https://ebible.org/Scriptures/`
- `assets/country_continent.csv` (must exist before running)
- `assets/country_code_overrides.csv` (applied if present; missing file is not an error)

### Algorithm

1. HTTP GET `https://ebible.org/Scriptures/`; parse HTML with `beautifulsoup4` (already a project dependency via `sil-machine`).
2. Locate the main `<table>`. For each `<tr>` (skip the header row):
   - Extract `countryCode` from the `href` of the first `<a>` in the Territory cell matching `country.php?c=XX`.
   - Extract `translationId` from the `href` of the first `<a>` in the Language cell (col 2) matching `details.php?id=XXX`.
   - Skip rows where either pattern is absent.
3. Load `assets/country_code_overrides.csv` (if present) and apply: replace any scraped `countryCode` that matches a `raw_code` entry with its `iso_code`.
4. Read `assets/country_continent.csv`; build a dict `countryCode → continentCode`.
5. Join: for each `(translationId, countryCode)` pair, look up `continentCode`.
6. Warn to stderr for any `countryCode` still not found in the continent mapping after overrides.
7. Write `assets/language_country_continent.csv`.

### Output schema

| Column | Example | Constraint |
|---|---|---|
| `translationId` | `engPEV` | Matches key in `ebible_status.csv` |
| `countryCode` | `GB` | ISO 3166-1 alpha-2, always uppercase |
| `continentCode` | `EU` | One of: AF, AN, AS, EU, NA, OC, SA |

No duplicate `translationId` rows. One translation may map to exactly one country.

---

## Changes to `ebible.py`

### 1. New column group

Add after `LICENCE_COLUMNS`:

```python
ENRICHMENT_COLUMNS = ["countryCode", "continentCode"]
```

Update the combined list:

```python
ALL_STATUS_COLUMNS = ORIGINAL_COLUMNS + LICENCE_COLUMNS + STATUS_COLUMNS + ENRICHMENT_COLUMNS
```

### 2. New function: `enrich_with_country_data(df, assets_dir)`

```
Inputs:  df (status DataFrame), assets_dir (Path)
Returns: df with countryCode and continentCode populated where absent
```

Behaviour:
- Load `assets_dir / "language_country_continent.csv"`. If the file does not exist, log a warning and return `df` unchanged.
- For each row in `df` where `countryCode` is NaN, look up `translationId` in the mapping and fill both `countryCode` and `continentCode`.
- Log a warning for any `translationId` present in `df` but absent from the mapping.
- Do not overwrite existing non-null values (idempotent).

### 3. Call site in `main()`

Call `enrich_with_country_data(status_df, assets_dir)` after loading/creating `status_df` and before the first `status_df.to_csv()` call.

---

## Changes to `corpus_to_parquet.py`

Add to `METADATA_SOURCE_COLUMNS` after the `status_inferred_versification` entry:

```python
# ENRICHMENT columns (Phase 1)
"countryCode",
"continentCode",
```

The existing guard `present = [c for c in METADATA_SOURCE_COLUMNS if c in df.columns]` already handles the case where the columns are absent, so no other change is needed.

---

## Verification

### Script output (`generate_language_country_continent.py`)

1. `assets/language_country_continent.csv` is created with exactly three columns: `translationId`, `countryCode`, `continentCode`.
2. Row count ≥ 700 (eBible has 700+ translations).
3. All `countryCode` values match `^[A-Z]{2}$`.
4. All `continentCode` values are in `{AF, AN, AS, EU, NA, OC, SA}`.
5. No duplicate `translationId` values.
6. Spot-check: `engBBE` maps to `GB` / `EU` (note: `engKJV` is not listed on the Scriptures page).
7. Known override applied: any translation with raw code `RP` stores `countryCode = PH`, `continentCode = AS`.

### `ebible.py` pipeline

7. After running the pipeline: `ebible_status.csv` has `countryCode` and `continentCode` columns.
8. A known redistributable translation (e.g. `engKJV`) has non-null `countryCode`.
9. Re-running the pipeline does not overwrite populated values with NaN (idempotent).

### `corpus_to_parquet.py`

10. `metadata.parquet` contains `countryCode` and `continentCode` columns.
11. Values in `metadata.parquet` match those in `ebible_status.csv` for the same `translationId`.

### Tests (`tests/test_phase1.py`)

| Test | What it checks |
|---|---|
| `test_scrape_parses_country_code` | Mock HTML → correct `countryCode` extracted |
| `test_scrape_parses_translation_id` | Mock HTML → correct `translationId` extracted |
| `test_scrape_skips_headerrow` | Header row is not included in output |
| `test_continent_join` | Given sample mapping CSV, join produces correct `continentCode` |
| `test_missing_country_warns` | Unknown `countryCode` triggers a stderr warning |
| `test_output_csv_columns` | Output CSV has exactly the three expected columns |
| `test_no_duplicate_translation_ids` | Output CSV has no duplicate `translationId` |
| `test_enrich_fills_missing` | `enrich_with_country_data()` fills NaN rows correctly |
| `test_enrich_is_idempotent` | Re-running enrich does not overwrite existing values |
| `test_enrich_missing_file` | Function returns df unchanged when CSV is absent |
| `test_metadata_parquet_includes_columns` | (integration) `metadata.parquet` has both new columns when source data is present |
| `test_override_applied` | `RP` raw code is replaced by `PH` before continent lookup; result is `PH`/`AS` |
| `test_override_missing_file_ok` | Script runs without error when overrides file is absent |
| `test_override_unknown_after_override_warns` | Code not in overrides and not in continent map still triggers warning |
