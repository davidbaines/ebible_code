# Process to follow

## Outline of the feature
The next feature is a utiliy script loading the data from huggingface. This is for those using the data to be able to create slices that
are useful for machine learning. Train, Test and Validation set creation will be one use for the script. 

## Interview stage
To create the spec for the features interview me in depth about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one by one.  Ask about requirements, edge cases, user experience, data models, and failure modes. Do not write a plan document or code until we are in agreement about how to proceed. 

Key features will be the ability to filter the data in a very flexible way.  In particular being able to filter Bible translations according to their language, language family, country, and even continent. Additional data containing tables that link iso codes to language families, and linking countries to continents will be required. Another simple table to match the country to the continent will be required and a script to create that is part of the scope of this work. The ebible.py code will need to get the country for each translationId from the https://ebible.org/Scriptures/ page. Unfortunately the information isn't included in the translations.csv file, that page contains a table that should be simple to parse.

## File creation
After the interview phase, and before you start work on this project, create these two files in the `dataloader_feature` folder (the folder that contains this file):
1. `spec.md` — a complete spec with goals, implementation details, and a verification section describing exactly how you'll prove each piece works.
2. `todo.md` — a running to-do list you'll edit as you work. Break complex tasks into verifiable sub-tasks.
3. Store tests in tests/ to verify everything you build. Loop on them until each passes.

## Long running phase once the plan is ready - follow these steps while you work:
 (a) Consult spec.md before every change.
 (b) Mark each completed task in todo.md with [x] once it is completed. 
 (c) Run tests after every meaningful commit, 
 (d) Every 20 iterations or so, call a fresh sub-agent with "Review spec.md and the current implementation for gaps" and loop on the sub-agent's feedback until alignment is reached.

Do not ask me for clarification on anything you can resolve by reading the spec and running the tests. Start with the spec.

____________________________________

Design decisions so far:

Phase 1 — Country and continent data in the pipeline

Scrape country info from ebible.org/Scriptures. Match it with continent data and add both to ebible_status.csv. 
Have corpus_to_parquet.py include those in metadata.parquet.

Phase 2 — ISO 639-3 enrichment + country→continent mapping
Join reference tables at corpus_to_parquet.py time (not into ebible_status.csv), adding the enriched columns to metadata.parquet.

Phase 3 — Dataloader script
Utility that loads from HuggingFace, filters, and produces output suitable for ML use.

1. "language family" refers to traditional linguistic families (Indo-European, Niger-Congo, Austronesian).
2. "country" will be stored as an ISO 3166 code only. Continent will be stored as a two letter code only. We will list only one country per translationId
3. We will create the country to continent mapping from the glottolog data.
4. The dataloader script will provide flexible output options from the dataloader script. Including HuggingFace Dataset object, a pandas DataFrame or csv files according to the users requests.
5. The flexible filtering will allow users to create train/test/val splits. 

Notes:
https://ebible.org/Scriptures/ contains a table whose first column is 'Territory' and then the 'Language', 'Language (English)' and 'Vernacular Title'. We should be able to find one of those three in translations.csv in order to match it up with the Territory field which contains the name of the country. 
Here's the first three rows of the table, showing the column headings.
<table border="1" padding="2"><tbody><tr><td><b>Territory</b></td><td><a href="index.php?sort=l">Language</a></td><td><a href="index.php?sort=e">Language (English)</a></td><td><a href="index.php?sort=v">Vernacular Title</a></td><td><a href="index.php?sort=t">English Title</a></td></tr><tr class="redist"><td><a href="country.php?c=AL"><img src="/flags/al.png"> Albania</a></td><td><a href="details.php?id=rup" target="_blank" class="liberation_sans redist">Armãneashti/Arumanisht</a></td><td><a href="details.php?id=rup" target="_blank" class="liberation_sans redist">Armãneashti/Arumanisht</a></td><td><a href="details.php?id=rup" target="_blank" class="liberation_sans redist">Biblija tu limba Rrãmãnã</a></td><td><a href="details.php?id=rup" target="_blank">Aromanian Bible</a></td></tr>
<tr class="restricted"><td><a href="country.php?c=DZ"><img src="/flags/dz.png"> Algeria</a></td><td><a href="details.php?id=arq" target="_blank" class="amiri restricted">Arabic, Algerian Spoken</a></td><td><a href="details.php?id=arq" target="_blank" class="amiri restricted">Arabic, Algerian Spoken</a></td><td><a href="details.php?id=arq" target="_blank" class="amiri restricted">العهد الجديد باللهجة الجزائرية</a></td><td><a href="details.php?id=arq" target="_blank">Arabic, Algerian Spoken: Arabe Algerian NT</a></td></tr>

Data sources
This github gist contains a list of countries by continent: https://gist.github.com/stevewithington/20a69c0b6d2ff846ea5d35e5fc47f26c#file-country-and-continent-codes-list-csv-csv

One of the motivations for creating this dataset was to be able to reproduce Sami Liedes experiment as described in his blog: https://samiliedes.wordpress.com/author/samiliedes/  He trained a model with the Bible in ~50 languages, omitting Genesis, or the whole OT from 3 of them. Then the model 'knows' the content of the Bible having seen it in many languages. It can then translate Genesis into the languages where it hasn't trained on Genesis. This data is being prepared in order to be able to repeat variations of that experiment.

Train, Test and Val splits can be defined by the user of the data. The user should provide a CSV file (e.g. splits.csv) with columns 'translationId' 'book' 'chapter' 'verse' 'split' where the user could list which translations and books are in each split. Omitting values in the verse column would imply the whole chapter, omitting values in the chapter column would imply the whole book and omiting books would imply the whole Bible. Or they could have very precise selections down to the verse level. This would allow a carefully controlled mix of verses across train, test and val that so that there isn't cross contamination. 

Glottolog (https://glottolog.org), maintained by MPI-EVA and widely used in NLP research. It maps ISO 639-3 codes to full family trees (e.g. eng → Germanic → Indo-European). We wil use Glottolog as the language family source? Glottolog uses a 'Glottocode' to identify a 'languoid' which may be a language, dialect or language family. Glottocodes representing languages are assosciated with ISO 639-3 code, so for those there's an easy connection to the ebible data. The data is stored on Github: https://github.com/glottolog/glottolog The README includes various ways to access and use the data: "This repository is the place where Glottolog data is curated. So it's the right place to open issues about errors you identified and to propose changes. Since the format of the data here is tailored towards maintainability - and not towards accessibility - you might want to use the Python package pyglottolog (https://github.com/glottolog/pyglottolog)to access it programmatically. glottolog.org - the Glottolog website - may be the most convenient place to inspect and browse the latest released version of Glottolog data. It also provides access to various download formats, tailored towards various re-use scenarios.
glottolog as CLDF dataset is probably the best option for accessing all of Glottolog's languoid data. Due to the format being CLDF, it can be used from all kinds of programming environments such as spreadsheet programs, programming languages like R or python, or the UNIX shell. A description of the files in this datasets is available in the README."

Reference tables should be stored in the assets/ folder. They should be versioned in git alongside the code?

Glottolog implementation approach will be a One-time generation script → committed CSV — A scripts/generate_language_families.py script uses pyglottolog (or Glottolog's CLDF download) once to produce assets/language_families.csv with ~1,000 rows (one per language in the corpus). That CSV is committed to git. corpus_to_parquet.py just reads the CSV — no Glottolog dependency at runtime. Re-run the script only when Glottolog data needs updating.

--custom-filter design discussion

A single general mechanism is better than multiple named table overrides. 

Convenience flags for the common dimensions:
--language CODE [CODE ...]    # filters on languageCode
--family FAMILY [FAMILY ...]  # filters on language_family
--country CODE [CODE ...]     # filters on country_iso2
--continent CODE [CODE ...]   # filters on continent_code

General filter flag (new — any column in metadata):
--filter COLUMN VALUE [VALUE ...]

Can be repeated. AND-combined with convenience flags and with other --filter calls. 

--custom_filter flag :
--custom_filter FILE --filter COLUMN VALUE [VALUE ...]

Joins a user-provided CSV onto metadata before filtering. Join key is translationId. 
This allows a user to bring any custom grouping — their own region taxonomy, project grouping, experimental cohort — and then filter on it with --custom-filter.
--custom_filter my_groups.csv --filter group 1 2
--filter Redistributable True
--filter languageCode eng fra spa

python dataloader.py filter --repo REPO --custom_filter my_groups.csv --filter experiment_group "South Asia"

--filter syntax should be --filter COLUMN VALUE1 VALUE2 (space-separated) both for --filter and --custom_filter

____________________________________

## Settled decisions — Phase 2

### Phase 2 — Glottolog language family enrichment

1. **Hierarchy level**: Two fields — `family_name` (top-level ancestor, e.g. `Indo-European`) and `classification` (full slash-separated path from root to language, e.g. `Indo-European/Germanic/West Germanic/High German/German`). No fixed "subfamily" column, because the tree depth is inconsistent across families (Germanic is 3 levels below Indo-European; Romance is 7). Users filter on `classification contains Germanic` instead.

2. **Column names**: `languageCode`, `glottocode`, `family_name`, `classification`. The file is a standalone join asset (`assets/glottolog_families.csv`) — these columns do NOT appear in `ebible_status.csv` or `metadata.parquet`. Users join this file at filter time via `--custom_filter assets/glottolog_families.csv --join-on languageCode`.

3. **Missing entries**: Rows with empty `iso639P3code` in Glottolog are silently skipped. eBible `languageCode` values that have no Glottolog match are absent from the output file; a `--filter` on `family_name` for an absent language simply returns no rows for it (not an error).

4. **Script location**: `ebible_code/get_glottolog_families.py`, following the Phase 1 pattern.

5. **Isolates**: Languages whose `family_id` is empty in Glottolog are assigned `family_name = "Isolate"`. The `classification` for an isolate is just the language name (single component, no slash).

6. **Macrolanguage overrides**: eBible already uses specific ISO 639-3 codes (e.g. `arq` for Algerian Arabic, `arz` for Egyptian Arabic) rather than macrolanguage umbrella codes. The `assets/macrolanguage_overrides.csv` file is provided as a safety valve for any edge cases, keyed on `ebible_language_code → glottolog_lookup_code`. The override is applied inside `get_glottolog_families.py` only; it does not modify `ebible_status.csv`.

7. **Licence and attribution**: Glottolog 5.3 is CC BY 4.0. The generated `assets/glottolog_families.csv` may be committed to git with proper attribution in `assets/ATTRIBUTION.md`.

## Settled decisions — Phase 3

### Phase 3 — Dataloader script

1. **Script location**: `ebible_code/dataloader.py` in this dev repo; deployed as top-level `dataloader.py` in the HuggingFace dataset repo (`DavidCBaines/ebible_corpus`), so users who download the dataset can run it directly alongside the data files.

2. **HuggingFace dataset repo**: `DavidCBaines/ebible_corpus`. This is the default value for `--repo`. Note: HuggingFace dataset loading scripts (the `DatasetBuilder` subclass mechanism) were deprecated and removed in `datasets` v4.0.0 (July 2025). The parquet-native approach is now standard; `dataloader.py` is a utility script documented in the dataset README.

3. **Splits omission semantics**: `translationId` only → full Bible; + `book` → whole book; + `book + chapter` → whole chapter; + all four columns → specific verse. All output columns the same length; verses not in a split are empty strings (vref alignment preserved).

4. **Splits + filtering interaction**: Filtering runs first; splits are applied only to the filtered set. Translations named in `splits.csv` but excluded by filters are listed in the summary. Translations that survived filters but have zero non-empty verses are also noted.

5. **Output shape**: Wide/parallel by default — one row per vref (41,899 rows), one column per selected translation (translationId as column header), first column = `vref`. Two output files: text table + metadata table. `--no-metadata` suppresses the metadata table.

6. **Metadata flag**: `--metadata-columns COLUMN [COLUMN ...]` controls which columns appear in the metadata table. Default: `translationId languageCode countryCode continentCode Redistributable`.

7. **HuggingFace Dataset output**: `--output-format huggingface` returns a flat `Dataset` (no splits) or a `DatasetDict` with split names as keys (with `--splits`).

8. **Subcommand structure**: `filter` (inspect), `load` (produce output files), `split` (produce split output files). Shared parent parser carries all filter arguments across all three subcommands.

____________________________________

## Design summary — Phase 1: country and continent data

**Goal**: Add `countryCode` (ISO 3166-1 alpha-2) and `continentCode` (two-letter code, e.g. `EU`, `AF`) to `ebible_status.csv` and `metadata.parquet`.

### Mapping file

A static reference file `assets/language_country_continent.csv` will be committed to git and regenerated only when the upstream ebible.org data changes. It has three columns:

| Column | Example | Notes |
|---|---|---|
| `translationId` | `engPEV` | Matches the key in `ebible_status.csv` |
| `countryCode` | `GB` | ISO 3166-1 alpha-2, extracted from `country.php?c=XX` hrefs |
| `continentCode` | `EU` | Two-letter code only; no English names stored |

The file is keyed at **translationId level**, not languageCode level. This intentionally duplicates the country/continent values across translations that share a language, in exchange for a simple, join-free table design.

### How the file is generated

A one-off script (`ebible_code/generate_language_country_continent.py`) will:

1. Scrape the table at `https://ebible.org/Scriptures/` and parse two href patterns per row:
   - Territory cell: `country.php?c=AL` → `countryCode = AL`
   - Language cell: `details.php?id=engPEV` → `translationId = engPEV`
   
   Note: the `id` in `details.php?id=XXX` is the ebible `translationId` as it appears in `ebible_status.csv`. It happens to equal the ISO 639-3 `languageCode` only when there is one translation for that language; `engPEV` proves these are distinct fields.

2. Download (or read from `assets/`) the country→continent mapping CSV (source: https://gist.github.com/stevewithington/20a69c0b6d2ff846ea5d35e5fc47f26c) and extract `countryCode → continentCode`.

3. Join the two tables on `countryCode` and write `assets/language_country_continent.csv`.

### Pipeline integration

- **`ebible.py`**: after the existing extract/finalize stage, join `ebible_status.csv` with `assets/language_country_continent.csv` on `translationId` and populate the `countryCode` and `continentCode` status columns.
- **`corpus_to_parquet.py`**: include `countryCode` and `continentCode` in `metadata.parquet`.

---

## Design summary — Phase 2: Glottolog language family data

**Goal**: Produce `assets/glottolog_families.csv` — a committed join asset mapping every Glottolog language's ISO 639-3 code to its top-level family name and full classification path. The Phase 3 dataloader uses this file at filter time; no pipeline columns are added.

### Output file

`assets/glottolog_families.csv` — four columns:

| Column | Example | Notes |
|---|---|---|
| `languageCode` | `eng` | ISO 639-3 code; join key into eBible `languageCode` |
| `glottocode` | `stan1293` | Glottolog unique identifier |
| `family_name` | `Indo-European` | Name of the top-level ancestor; `"Isolate"` for language isolates |
| `classification` | `Indo-European/Germanic/West Germanic/High German/German` | Full path from root to language, slash-separated |

One row per ISO 639-3 code. Glottolog rows with empty `iso639P3code` are excluded. If two Glottolog entries share an ISO code, the first occurrence wins.

### How the file is generated

`ebible_code/get_glottolog_families.py`:

1. Downloads Glottolog 5.3 `languoid.csv.zip` from the versioned CDStar URL and opens it in-memory.
2. Reads `languoid.csv` with `keep_default_na=False` and `dtype=str` (prevents `"NA"` being read as NaN).
3. Loads `assets/macrolanguage_overrides.csv` if present (no error if absent).
4. Builds an index `id → row` over all rows and `iso639P3code → row` over language-level rows.
5. For each language-level row with a non-empty ISO code, applies any macrolanguage override, then traces the ancestor chain via `parent_id` to build `family_name` and `classification`.
6. Languages whose `family_id` is empty are isolates: `family_name = "Isolate"`, `classification` = just the language name.
7. Writes `assets/glottolog_families.csv`.

### What is NOT changed

- `ebible.py` — no new columns in `ebible_status.csv`
- `corpus_to_parquet.py` — no new columns in `metadata.parquet`

### Using the data at filter time (Phase 3 preview)

```bash
python dataloader.py filter \
  --custom_filter assets/glottolog_families.csv --join-on languageCode \
  --filter family_name is Indo-European \
  --filter classification contains Germanic
```

The `--join-on COLUMN` argument is scoped to its `--custom_filter` file: each `--custom_filter` can specify a different join key.

### Attribution

`assets/ATTRIBUTION.md` covers all data sources:
- Glottolog 5.3 (CC BY 4.0) — Hammarström et al. 2024
- Country-continent mapping Gist — Steve Withington (public domain / open)
- eBible.org translations table — sourced per individual translation licences

## Design summary — Phase 3: dataloader script

**Goal**: A CLI utility (`dataloader.py`) that reads eBible corpus parquet files, applies flexible filtering, and writes parallel text + metadata output suitable for ML use. Deployed as a top-level file in the HuggingFace dataset repo so users can run it alongside the downloaded data.

### HuggingFace context

The HuggingFace dataset loading script mechanism (`DatasetBuilder` subclass, invoked by `load_dataset()`) was deprecated and removed in `datasets` v4.0.0 (July 2025). The parquet-native approach is now standard. `dataloader.py` is a utility script documented in the dataset README, not a `load_dataset()` hook. Parquet files remain directly usable via `datasets.load_dataset("DavidCBaines/ebible_corpus")` without any custom script.

### Subcommands

```bash
python dataloader.py filter [filter-args]           # inspect: print matching translationIds
python dataloader.py load   [filter-args] [load-args]    # produce output files
python dataloader.py split  [filter-args] [load-args] [split-args]  # produce split files
```

### Two output files (from `load` and `split`)

**Text table** — one row per vref, one column per selected translation:

```
vref, engBBE, fra, deu
GEN 1:1, In the beginning..., Au commencement..., Im Anfang...
GEN 1:2, ...
```

**Metadata table** — one row per selected translation:

```
translationId, languageCode, countryCode, continentCode, Redistributable
engBBE, eng, GB, EU, True
fra, fra, FR, EU, True
```

`translationId` is the join key between the two files (column headers in text table ↔ `translationId` column in metadata table). `vref` is only in the text table.

### Key flags

| Flag | Default | Notes |
|---|---|---|
| `--repo` | `DavidCBaines/ebible_corpus` | Local path or HuggingFace dataset ID |
| `--filter COLUMN [OPERATOR] VALUE ...` | — | Operators: `is`, `contains`, `not`, `in`; repeatable; AND-combined |
| `--custom_filter FILE` | — | Join user CSV; paired with `--join-on COLUMN` |
| `--output FILE` | stdout | Text table output path |
| `--output-format` | `csv` | `csv`, `parquet`, `huggingface`, `pandas` |
| `--no-metadata` | off | Suppress metadata table |
| `--metadata-columns COL ...` | 5 defaults | Columns in metadata table |
| `--splits FILE` | — | splits.csv for train/test/val partitioning |
| `--output-dir DIR` | — | Directory for split output files |

### Splits semantics

All output columns are the same length; verses not in a split are empty strings. Filter runs before splits. Summary printed to stderr after every run.
