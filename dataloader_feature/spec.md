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
| `translationId` | `tblNT` | eBible translationId the override applies to |
| `ebible_country_code` | `RP` | Code as scraped from ebible.org for this translation |
| `country_code` | `PH` | Correct ISO 3166-1 alpha-2 replacement |
| `notes` | `Legacy Philippines code` | Human-readable reason; kept for auditing |

The match key is `(translationId, ebible_country_code)` — both must match for the override to apply. This prevents a bad code from silently correcting a different translation that happens to use the same raw code legitimately. Any raw code still unresolved after overrides triggers a warning — new unknown codes are always surfaced.

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

---

# Phase 2 Spec — Glottolog Language Family Data

## Goal

Produce `assets/glottolog_families.csv`, a committed reference file that maps every Glottolog language's ISO 639-3 code to its top-level family name and full ancestor-path classification. This file is a join asset for the Phase 3 dataloader; no pipeline columns are added to `ebible_status.csv` or `metadata.parquet`.

## Deliverables

| File | Type | Description |
|---|---|---|
| `assets/glottolog_families.csv` | Generated reference (committed) | languageCode → glottocode, family_name, classification |
| `assets/macrolanguage_overrides.csv` | Static reference (committed) | Maps eBible macrolanguage codes to Glottolog lookup codes |
| `assets/ATTRIBUTION.md` | Static reference (committed) | Credits for all data sources in `assets/` |
| `ebible_code/get_glottolog_families.py` | New script | Downloads Glottolog data and writes `glottolog_families.csv` |
| `tests/test_phase2.py` | New tests | Unit tests for all script logic |

---

## Data Sources

### Glottolog 5.3

URL: `https://cdstar.eva.mpg.de/bitstreams/EAEA0-608B-9919-A962-0/glottolog_languoid.csv.zip`

Licence: CC BY 4.0. May be committed to git with attribution.

Citation: Hammarström, Harald & Forkel, Robert & Haspelmath, Martin & Bank, Sebastian. 2024. *Glottolog 5.3*. Leipzig: Max Planck Institute for Evolutionary Anthropology. Available at https://glottolog.org

The zip contains `languoid.csv`, a self-referential table (~27,000 rows). Relevant columns:

| Column | Description |
|---|---|
| `id` | Glottocode (unique identifier for this languoid) |
| `name` | Human-readable name |
| `level` | `language` \| `dialect` \| `family` |
| `iso639P3code` | ISO 639-3 code (non-empty only for language-level entries) |
| `family_id` | Glottocode of top-level ancestor (empty for top-level families and isolates) |
| `parent_id` | Glottocode of immediate parent (empty for root nodes) |

### macrolanguage_overrides.csv

Static file, manually maintained. Used only inside `get_glottolog_families.py` to redirect any eBible macrolanguage codes to their Glottolog-resolvable specific codes.

| Column | Example | Notes |
|---|---|---|
| `ebible_language_code` | `ara` | Code as it appears in eBible's `languageCode` column |
| `glottolog_lookup_code` | `arb` | ISO 639-3 code to use when looking up in Glottolog |
| `notes` | `Arabic macrolanguage → Standard Arabic` | Human-readable reason |

Note: eBible already uses specific ISO 639-3 codes for Arabic and Chinese variants (e.g. `arq`, `arz`, `cmn`), so this file may start empty. It is provided as a safety valve.

---

## New Script: `ebible_code/get_glottolog_families.py`

### Inputs
- Glottolog 5.3 `languoid.csv.zip` (downloaded from URL above)
- `assets/macrolanguage_overrides.csv` (optional; missing file is not an error)

### Algorithm

1. HTTP GET the Glottolog CDStar URL; open the zip in-memory (no temp file written).
2. Read `languoid.csv` with `dtype=str, keep_default_na=False` — prevents `"NA"` (and other short strings) being silently coerced to NaN.
3. Load `assets/macrolanguage_overrides.csv` if present; build dict `{ebible_language_code: glottolog_lookup_code}`. Return `{}` if file absent.
4. Build two indexes from the DataFrame:
   - `id_to_row`: `{glottocode: Series}` over all rows
   - `iso_to_row`: `{iso639P3code: Series}` over language-level rows with non-empty ISO code (first occurrence wins on duplicates)
5. For each language-level row with a non-empty `iso639P3code`:
   a. If the ISO code is in the macrolanguage overrides dict, use the override code for lookup (emit a warning to stderr; write the original code as `languageCode` in the output).
   b. Trace the ancestor chain upward via `parent_id` until a row with no `parent_id` or a `parent_id` not in `id_to_row` is reached.
   c. Compute `family_name`:
      - If the language itself has an empty `family_id`: it is a top-level isolate → `family_name = "Isolate"`.
      - Otherwise: look up `family_id` in `id_to_row` → use that row's `name`.
   d. Compute `classification`: names of all nodes in the ancestor chain from root down to (and including) the language itself, joined with `/`.
      - For isolates: single component (just the language name).
6. Collect one row per ISO code (first occurrence wins).
7. Write `assets/glottolog_families.csv`.

### Output schema

| Column | Example | Constraint |
|---|---|---|
| `languageCode` | `eng` | ISO 639-3 code, lowercase; join key into eBible `languageCode` |
| `glottocode` | `stan1293` | Glottolog identifier for this language |
| `family_name` | `Indo-European` | Non-empty string; `"Isolate"` for isolates |
| `classification` | `Indo-European/Germanic/West Germanic/High German/German` | Slash-separated path from root to language |

One row per ISO 639-3 code. No duplicate `languageCode` values.

### What is NOT changed

- `ebible.py` — no new columns added to `ebible_status.csv`
- `corpus_to_parquet.py` — no new columns added to `metadata.parquet`
- `ebible_status.csv` — Glottolog data is a separate join-time asset, not pipeline state

---

## Verification

### Script output (`get_glottolog_families.py`)

1. `assets/glottolog_families.csv` is created with exactly four columns: `languageCode`, `glottocode`, `family_name`, `classification`.
2. Row count ≥ 7,500 (Glottolog 5.3 has 7,859 language-level entries with ISO codes).
3. No duplicate `languageCode` values.
4. All `languageCode` values match `^[a-z]{3}$`.
5. All `family_name` values are non-empty strings.
6. All `classification` values are non-empty strings.
7. Spot-check: `eng` → `glottocode = stan1293`, `family_name = Indo-European`, `classification` starts with `Indo-European` and contains `Germanic`.
8. Spot-check: `eus` (Basque) → `family_name = Isolate`.
9. Spot-check: `fra` (French) → `family_name = Indo-European`, `classification` contains `Romance`.
10. Spot-check: `arq` (Algerian Arabic) → `family_name = Afro-Asiatic`.
11. No warning fired for `arq` (eBible uses the specific code, not the macrolanguage `ara`).

### Tests (`tests/test_phase2.py`)

| Test | What it checks |
|---|---|
| `test_family_name_extracted` | Language with known family_id → correct family_name looked up |
| `test_classification_path` | Full ancestor chain traced correctly; slash-separated path matches expected |
| `test_isolate_family_name` | Language with empty family_id → `family_name = "Isolate"` |
| `test_isolate_classification_single_component` | Isolate classification has no `/` separator |
| `test_no_duplicate_language_codes` | Output DataFrame has no duplicate `languageCode` |
| `test_output_csv_columns` | Output has exactly four expected columns in correct order |
| `test_macrolanguage_override_applied` | Override dict redirects lookup; original code used as `languageCode` in output |
| `test_macrolanguage_override_missing_file_ok` | Missing overrides file → `{}` returned, no exception |
| `test_missing_iso_code_skipped` | Rows with empty `iso639P3code` do not appear in output |
| `test_first_occurrence_wins_on_duplicate_iso` | When two Glottolog rows share an ISO code, first row's data is kept |
| `test_keep_default_na_false` | ISO code `"NA"` is not parsed as NaN and survives into output |

---

# Phase 3 Spec — Dataloader Script

## Goal

A user-facing CLI utility (`dataloader.py`) that reads eBible corpus parquet files, applies flexible filtering and optional train/test/val splits, and writes the results in formats suited for ML use. The script lives at the top level of the HuggingFace dataset repo (`DavidCBaines/ebible_corpus`) so users who download the dataset can run it locally alongside the data.

## Deliverables

| File | Type | Description |
|---|---|---|
| `ebible_code/dataloader.py` | New script | CLI utility; deployed to HuggingFace repo top level |
| `tests/test_phase3.py` | New tests | Unit tests for filter, load, and split logic |

---

## Data Model

Two source parquet files (co-located with `dataloader.py` in the HuggingFace repo, or loaded remotely via `datasets.load_dataset()`):

| File | Shape | Notes |
|---|---|---|
| `main.parquet` | 41,899 rows × (1 `vref` + N translation columns) | Wide/parallel format; empty string for untranslated verses |
| `metadata.parquet` | N rows × M columns | One row per translation; includes `translationId`, `languageCode`, `countryCode`, `continentCode`, `Redistributable`, etc. |

---

## Subcommand Architecture

Three subcommands share a common parent parser that carries all filtering arguments. The subcommands build on each other: `filter` inspects, `load` produces output files, `split` also applies train/test/val partitioning.

```
python dataloader.py filter [filter-args]
python dataloader.py load   [filter-args] [load-args]
python dataloader.py split  [filter-args] [load-args] [split-args]
```

---

## Shared Filter Arguments (parent parser)

| Argument | Default | Description |
|---|---|---|
| `--repo REPO` | `DavidCBaines/ebible_corpus` | Local directory path or HuggingFace dataset ID |
| `--filter COLUMN [OPERATOR] VALUE ...` | — | Filter on any metadata column; repeatable; AND-combined |
| `--custom_filter FILE` | — | Join a user-supplied CSV onto metadata before filtering; repeatable |
| `--join-on COLUMN` | — | Join key for the preceding `--custom_filter`; scoped per file |

### Filter operators

| Operator | Meaning | Example |
|---|---|---|
| `is` | Exact match (default if operator omitted) | `--filter Redistributable True` |
| `contains` | Substring match | `--filter classification contains Germanic` |
| `not` | Not equal | `--filter translationId not engKJV` |
| `in` | One of a list | `--filter continentCode in EU AS` |

Multiple `--filter` flags are AND-combined. Unknown column names raise a clear error. `--join-on` column absent from the custom CSV raises a clear error.

---

## `filter` Subcommand

Prints matching translationIds and a one-line summary count. Writes no output files.

```bash
python dataloader.py filter \
  --filter Redistributable True \
  --filter continentCode EU
```

Output: sorted list of matching translationIds followed by `N translations matched.`

---

## `load` Subcommand

Loads verse text for matched translations and writes two output files: a text table and (optionally) a metadata table.

### Additional arguments

| Argument | Default | Description |
|---|---|---|
| `--output FILE` | stdout | Output path for text table |
| `--output-format csv\|parquet\|huggingface\|pandas` | `csv` | Output format |
| `--metadata-output FILE` | Derived from `--output` | Path for metadata table (e.g. `out.csv` → `out_metadata.csv`) |
| `--no-metadata` | off | Suppress metadata table |
| `--metadata-columns COL [COL ...]` | see below | Columns in metadata table |

Default `--metadata-columns`: `translationId languageCode countryCode continentCode Redistributable`

### Output shape

**Text table** — one row per vref, one column per selected translation:

| vref | engBBE | fra | deu |
|---|---|---|---|
| GEN 1:1 | In the beginning... | Au commencement... | Im Anfang... |
| GEN 1:2 | … | … | |

- Column headers are `translationId` values.
- All columns the same length (41,899 rows).
- Missing/untranslated verses are empty strings, not NaN.

**Metadata table** — one row per selected translation:

| translationId | languageCode | countryCode | continentCode | Redistributable |
|---|---|---|---|---|
| engBBE | eng | GB | EU | True |
| fra | fra | FR | EU | True |

- `translationId` is the join key matching text table column names.
- Columns controlled by `--metadata-columns`.

---

## `split` Subcommand

Applies user-defined train/test/val splits. Inherits all `load` arguments, plus:

| Argument | Description |
|---|---|
| `--splits FILE` | Path to `splits.csv` |
| `--output-dir DIR` | Directory for split output files |

### Splits CSV format

Columns: `translationId`, `book`, `chapter`, `verse`, `split`

### Omission semantics

| Columns present | Scope |
|---|---|
| `translationId` only | Entire Bible for that translation |
| `translationId` + `book` | Whole book |
| `translationId` + `book` + `chapter` | Whole chapter |
| All four | Specific verse |

### Split output files

For `--output-dir out/` with splits `train`, `test`, `val`:

```
out/train.csv         out/train_metadata.csv
out/test.csv          out/test_metadata.csv
out/val.csv           out/val_metadata.csv
```

All split text tables have the same column set and the same row count; verses not in a given split are empty strings (alignment preserved).

### Filtering and splits interaction

1. Filtering runs **first**; only matched translations are loaded.
2. Splits are applied to the filtered set.
3. Summary printed to stderr:
   - Number of translations matched by filters.
   - For each translation: number of non-empty verses in output.
   - Translations named in `splits.csv` but excluded by filters (with reason).
   - Translations that matched filters but have zero non-empty verses.

---

## Loading Strategy

When `--repo` contains `/` (e.g. `DavidCBaines/ebible_corpus`): load via `datasets.load_dataset(repo)`.

When `--repo` is a local path: read `main.parquet` and `metadata.parquet` directly from that directory.

When running from the HuggingFace repo directory with no `--repo` flag: defaults to loading from the current directory.

---

## HuggingFace Dataset Output (`--output-format huggingface`)

- No `--splits`: returns a flat `Dataset`.
- With `--splits`: returns a `DatasetDict` with split names as keys.

---

## Verification

1. `filter` with `--filter Redistributable True` prints only redistributable translationIds.
2. `filter` with `--filter classification contains Germanic` (via `--custom_filter assets/glottolog_families.csv --join-on languageCode`) prints only Germanic-family translations.
3. `filter` with `--filter continentCode in EU AS` returns correct subset.
4. Multiple `--filter` flags are AND-combined (intersection, not union).
5. Unknown column in `--filter` raises a clear error message (not a Python traceback).
6. `--join-on COLUMN` pointing to absent column raises a clear error.
7. `load` text table has 41,899 rows and N+1 columns (vref + translations).
8. `load` text table has empty strings (not NaN/None) for untranslated verses.
9. `load` metadata table has N rows and the columns specified by `--metadata-columns`.
10. `--no-metadata` produces text table only.
11. `--metadata-columns translationId languageCode` produces metadata with exactly those two columns.
12. `split` output files all have 41,899 rows; verses not in a split are empty strings.
13. Omission: a splits row with only `translationId` assigns the full Bible to that split.
14. Omission: a row with `translationId + book` assigns the whole book.
15. Filtering runs before splits: a translation excluded by `--filter` is absent from all split files, and named in the summary.
16. `--output-format parquet` writes valid parquet that can be read back with pandas.
17. `--output-format huggingface` returns `Dataset` (no splits) or `DatasetDict` (with splits).

---

## Tests (`tests/test_phase3.py`)

| Test | What it checks |
|---|---|
| `test_filter_exact` | `is` operator returns exact matches only |
| `test_filter_contains` | `contains` operator returns substring matches |
| `test_filter_not` | `not` operator excludes the specified value |
| `test_filter_in` | `in` operator matches any of the listed values |
| `test_filter_and_combined` | Two `--filter` flags → AND of both conditions |
| `test_filter_unknown_column_raises` | Clear error on column not in metadata |
| `test_custom_filter_join` | `--custom_filter FILE --join-on COLUMN` works correctly |
| `test_custom_filter_bad_column_raises` | Clear error when join column absent from CSV |
| `test_load_text_table_shape` | Text table has correct row and column counts |
| `test_load_empty_strings_not_nan` | Missing verses are `""`, not NaN |
| `test_load_vref_first_column` | First column of text table is `vref` |
| `test_metadata_table_shape` | Metadata table has N rows × M columns |
| `test_metadata_columns_flag` | `--metadata-columns` controls output columns exactly |
| `test_no_metadata_flag` | `--no-metadata` suppresses metadata output |
| `test_split_assigns_correctly` | Verses go to the split named in splits.csv |
| `test_split_omission_translation_only` | Row with only `translationId` → whole Bible in that split |
| `test_split_omission_book_only` | Row with `translationId + book` → whole book |
| `test_split_filter_first` | Translation excluded by filter absent from all split files |
| `test_split_summary_warns_filtered` | Summary mentions translations removed by filter |
| `test_split_alignment_preserved` | All split text tables same row count; absent verses are `""` |
| `test_output_format_parquet` | Parquet output round-trips correctly through pandas |
