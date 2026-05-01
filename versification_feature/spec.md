# Versification Estimation Feature — Spec

## Goals

1. Replace `get_versification_with_scoring() -> str` with `estimate_versification() -> VersificationType`, including `UNKNOWN` (0) as a valid return value.
2. Pre-compute module-level caches (standard verse data, invariant chapters) so work is never repeated across calls.
3. Restructure `write_settings_file()` to accept versification as a parameter rather than computing it internally.
4. Update pipeline callers in `ebible.py` to store the integer versification value in the status CSV, with correct UNKNOWN → ENGLISH fallback written to Settings.xml.
5. Provide `analyse_versification.py` to help the user choose `VERSIFICATION_UNKNOWN_THRESHOLD` empirically.
6. Add `describe_versification_match() -> VersificationMatchReport` for explainability and threshold calibration.
---

## Definitions

- **Differentiating chapter**: a `(book, chapter)` pair where not all standard versifications agree on the verse count. Only these chapters can distinguish one versification from another.
- **Invariant chapter**: a `(book, chapter)` pair where all standard versifications agree on the verse count. These chapters contribute nothing to versification scoring.
- **Indistinguishable**: a project whose `.vrs` file contains no differentiating chapters (e.g. a very short NT-only translation that happens to cover only invariant chapters). Returned as `ENGLISH` but flagged in the match report. "Indistinguishable" is shorthand for: the Bible only contains chapters and verses whose verse counts are identical in all versification systems.
- **UNKNOWN**: a project whose best similarity score across all standards is below `VERSIFICATION_UNKNOWN_THRESHOLD`. Stored as integer `0` in the status CSV; Settings.xml receives `ENGLISH` (4) as a safe fallback.

---

## Standard Versifications

Loaded at module import via `VersificationType` (all members except `UNKNOWN`):

| VersificationType    | Integer value |
|----------------------|---------------|
| `ORIGINAL`           | 1             |
| `SEPTUAGINT`         | 2             |
| `VULGATE`            | 3             |
| `ENGLISH`            | 4             |
| `RUSSIAN_PROTESTANT` | 5             |
| `RUSSIAN_ORTHODOX`   | 6             |

---

## Module-Level Pre-Computation

`populate_standard_versifications()` (called at module import) populates:

| Name | Type | Description |
|------|------|-------------|
| `LOADED_VRS_OBJECTS` | `Dict[str, Versification]` | Keyed by versification name (existing) |
| `STANDARD_VERSE_DATA` | `Dict[VersificationType, Dict[Tuple[str, int], int]]` | Pre-extracted `{(book, chapter): max_verse}` per standard (**new**) |
| `INVARIANT_CHAPTERS` | `Set[Tuple[str, int]]` | Chapters where all standards agree on verse count (**new**) |
| `VALID_VRS_NUM_STRINGS` | `List[str]` | Valid integer strings for validation (existing) |

`VRS_NAME_TO_NUM_STRING` is replaced by direct use of `VersificationType.value`.

---

## API

### `estimate_versification(project_path: Path) -> VersificationType`

1. Locate `<project_path>/<project_name>.vrs`. If missing or unloadable, log a warning and return `VersificationType.ENGLISH`.
2. Extract project verse data via `get_verse_data_from_vrs_obj()`.
3. Determine the project's differentiating chapters: non-invariant `(book, chapter)` pairs that are present in the project's verse data, after first excluding **spurious placeholder chapters** — entries where the project has `max_verse=1` but at least one standard versification has `max_verse > 1`. These placeholders arise because `Versification.load()` fills `(chapter=1, verse=1)` for all 66 canonical books not actually present in the `.vrs` file; excluding them prevents NT-only projects from falsely matching standards (e.g. SEPTUAGINT) that happen to assign 1 verse to certain OT chapters.
4. If there are no differentiating chapters: return `VersificationType.ENGLISH` (indistinguishable case; caller can detect this via `describe_versification_match()`).
5. For each standard versification, compute:
   ```
   score = (number of differentiating chapters where project verse count == standard verse count)
           / (total number of project differentiating chapters)
   ```
6. If `best_score < VERSIFICATION_UNKNOWN_THRESHOLD`: return `VersificationType.UNKNOWN`.
7. Otherwise return the `VersificationType` with the highest score. Tie-break: prefer `ENGLISH`.

**Threshold**: read `VERSIFICATION_UNKNOWN_THRESHOLD` from environment (float, 0.0–1.0). Default `0.0` (UNKNOWN is never returned until the user sets a value after running `analyse_versification.py`).

---

### `write_settings_file(project_folder, language_code, versification, language_name_in_english, full_name) -> bool`

Full signature:
```python
def write_settings_file(
    project_folder: Path,
    language_code: str,
    versification: VersificationType,
    language_name_in_english: str = "",  # from CSV column: languageNameInEnglish
    full_name: str = "",                 # from CSV column: title
) -> bool:
```

- The caller is responsible for substituting `ENGLISH` for `UNKNOWN` before calling.
- Writes Settings.xml using `versification.value` as the integer.
- Returns `True` on success, `False` on failure.
- Does **not** read the existing `Settings.xml`.
- Does **not** return old or new value dicts.

CSV column → XML element mapping:

| CSV column | XML element | Notes |
|---|---|---|
| `languageNameInEnglish` | `<Language>` | English name of the language |
| `title` | `<FullName>` | Full vernacular title |
| `translationId` | `<Name>` | Derived from `project_folder.name` |
| *(hardcoded)* | `<Encoding>` | Always `65001` (UTF-8) |
| `languageCode` | `<LanguageIsoCode>` | Appended with `:::` |

Settings.xml format:
```xml
<ScriptureText>
  <Language>{language_name_in_english}</Language>
  <Encoding>65001</Encoding>
  <FullName>{title}</FullName>
  <Name>{project_folder.name}</Name>
  <Versification>{versification.value}</Versification>
  <LanguageIsoCode>{language_code}:::</LanguageIsoCode>
  <Naming BookNameForm="41MAT" PostPart="{file_prefix}.SFM" PrePart="" />
</ScriptureText>
```

---

### `VersificationMatchReport` and `describe_versification_match()`

> **Lives in `analyse_versification.py`**, not `settings_file.py`. Not imported by `ebible.py`.

```python
@dataclass
class VersificationMatchReport:
    project_name: str
    best_match: VersificationType           # what estimate_versification() returned
    best_score: float                       # fraction of differentiating chapters matched
    scores: dict[VersificationType, float]  # score for every standard
    mismatch_counts: dict[VersificationType, int]  # mismatched differentiating chapters per standard
    total_differentiating_chapters: int     # chapters that can distinguish versifications
    total_project_chapters: int             # total chapters in the project
    status: str                             # "matched" | "indistinguishable" | "unknown"
    notes: str                              # human-readable explanation (see below)
```

`describe_versification_match(project_path: Path) -> VersificationMatchReport` — runs the full scoring logic using the same internals exported from `settings_file.py` as `estimate_versification()`, so results are always consistent.

**`notes` generation rules:**

- `"matched"`: `"Best match: {name} ({score:.1%} of differentiating chapters match)"`
- `"indistinguishable"`: `"All {n} project chapters are invariant across versifications. Cannot distinguish; defaulting to English. 'Indistinguishable' means the Bible only contains chapters and verses whose verse counts are identical in all versification systems."`
- `"unknown"`: `"Best score {score:.1%} is below threshold {threshold:.1%}. No versification matched well. Settings.xml will use English (4) as a fallback. Mismatch counts per standard: {per_standard_summary}"`

---

## Module Boundaries

| Module | Contains | Imported by |
|---|---|---|
| `settings_file.py` | `estimate_versification()`, `write_settings_file()`, `generate_vrs_from_project()`, scoring internals, module-level caches | `ebible.py` |
| `analyse_versification.py` | `VersificationMatchReport`, `describe_versification_match()`, CSV/PNG output, stdout summary | standalone script only |

---

## Pipeline Changes (`ebible.py`)

### `unzip_and_process_files()`

**Replace:**
```python
settings_path, vrs_num, _, _ = write_settings_file(project_dir, lang_code)
df.loc[index, 'status_inferred_versification'] = vrs_num
if settings_path: df.loc[index, 'status_settings_xml_date'] = TODAY_STR
```

**With:**
```python
vrs_type = estimate_versification(project_dir)
df.loc[index, 'status_inferred_versification'] = vrs_type.value
xml_vrs_type = VersificationType.ENGLISH if vrs_type == VersificationType.UNKNOWN else vrs_type
success = write_settings_file(
    project_dir, lang_code, xml_vrs_type,
    language_name_in_english=row.get('languageNameInEnglish', ''),
    full_name=row.get('title', ''),
)
if success:
    df.loc[index, 'status_settings_xml_date'] = TODAY_STR
```

### `update_all_settings()`

Same pattern. Old versification value is read from `status_inferred_versification` in the status CSV *before* calling `estimate_versification()`, replacing the old `old_vals` mechanism.

### UNKNOWN Reporting

After the extraction phase, count rows where `status_inferred_versification == 0`. If any:

```
WARNING: {n} translation(s) have UNKNOWN versification (value 0).
Settings.xml for these projects uses English (4) as a fallback.
Review {status_path}, filtering for status_inferred_versification == 0.
Run ebible_code/analyse_versification.py for detailed mismatch analysis.
```

---

## `analyse_versification.py`

Standalone script. Not imported by `ebible.py`.

**Inputs**: reads `EBIBLE_DATA_DIR` (and optionally `TEST_EBIBLE_DATA_DIR`) from `.env`. Scans all subdirectories of `projects/` and `private_projects/` that contain a `.vrs` file.

**Processing**: calls `describe_versification_match()` for each qualifying project folder.

**Outputs**:

1. `{metadata_folder}/analyse_versification.csv` — one row per project, all `VersificationMatchReport` fields flattened (scores and mismatch_counts expanded to one column per versification).
2. `{metadata_folder}/versification_scores_histogram.png` — histogram of `best_score` values across all projects, with a vertical dashed line at the current `VERSIFICATION_UNKNOWN_THRESHOLD`.
3. Stdout summary:
   - Total projects analysed
   - Score distribution in bands: 0.0–0.1, 0.1–0.2, … 0.9–1.0 (count per band)
   - Count of indistinguishable projects
   - Count of projects currently below threshold

---

## `.env` Additions

```dotenv
# Threshold below which a translation's best versification match score is treated as UNKNOWN (VersificationType 0).
# Range: 0.0 (never return UNKNOWN) to 1.0 (always return UNKNOWN).
# Run `poetry run python ebible_code/analyse_versification.py` to inspect the score
# distribution and choose a suitable value. Start by examining the histogram.
VERSIFICATION_UNKNOWN_THRESHOLD=0.0
```

---

## Removals

| Item | Reason |
|------|--------|
| `get_versification_with_scoring()` | Replaced by `estimate_versification()` |
| `add_settings_file()` | Not called by any production code |
| `WEIGHT_BOOK`, `WEIGHT_CHAPTER`, `WEIGHT_VERSE_COUNT` constants | Scoring logic now lives entirely in `estimate_versification()` |
| `VRS_NAME_TO_NUM_STRING` lookup in `write_settings_file()` | `VersificationType.value` used directly |
| `old_vals` / `new_vals` return from `write_settings_file()` | Old values read from status CSV instead |

---

## Verification

### V1 — Module-level pre-computation

**How**: unit test that imports `settings_file` and asserts:
- `STANDARD_VERSE_DATA` has exactly 6 keys, one per non-UNKNOWN `VersificationType`
- `INVARIANT_CHAPTERS` is non-empty (expected: several hundred chapters)
- Every key in `STANDARD_VERSE_DATA` is a `VersificationType` instance

### V2 — `estimate_versification()` with synthetic fixtures

Fixtures in `tests/fixtures/versification/` — hand-crafted `.vrs` files with controlled verse counts:

| Fixture file | Content | Expected return | Expected `status` |
|---|---|---|---|
| `nt_only_invariant.vrs` | NT chapters only, all invariant | `ENGLISH` | `indistinguishable` |
| `english_pattern.vrs` | Differentiating chapters matching English counts | `ENGLISH` | `matched` |
| `vulgate_pattern.vrs` | Differentiating chapters matching Vulgate counts | `VULGATE` | `matched` |
| `russian_orthodox_pattern.vrs` | Differentiating chapters matching Russian Orthodox counts | `RUSSIAN_ORTHODOX` | `matched` |
| `high_mismatch.vrs` | Differentiating chapters with verse counts matching no standard | `UNKNOWN` (threshold=0.3) | `unknown` |

**How**: parametrized pytest, creates a temp dir for each fixture, calls `estimate_versification()`, asserts return value and (via `describe_versification_match()`) the `status` field.

### V3 — `estimate_versification()` with real project data

Parametrized, skipped if project folder absent:

| Project folder | Expected `VersificationType` |
|---|---|
| `eng-web-c` | `ENGLISH` |
| `eng-vul` | `VULGATE` |
| `rus-synod-usfm-from-textus-rec` | `RUSSIAN_ORTHODOX` |
| `rus-vrt` | `RUSSIAN_PROTESTANT` |
| `abt-maprik` | `ENGLISH` |
| `eng-uk-lxx2012` | `SEPTUAGINT` |

**How**: extend existing parametrized test in `test_settings_file.py`.

> **Note**: Tests force `VERSIFICATION_UNKNOWN_THRESHOLD=0.0` via `monkeypatch` to assert raw scoring algorithm output. With the calibrated production threshold of `0.8`, low-confidence projects like `eng-uk-lxx2012` (score ≈ 0.596) return `UNKNOWN` instead of `SEPTUAGINT`.

> **Note**: The `status_inferred_versification` column in the status CSV is stored as `float64` by pandas when the column contains `NaN` for unprocessed rows. This is expected behaviour. Integer comparisons still work correctly (`4.0 == 4`, `0.0 == 0`); the spec's use of "integer" refers to the conceptual value, not the pandas dtype.

### V4 — `write_settings_file()` unit tests

- Call with each `VersificationType` value 1–6; parse resulting XML; assert `<Versification>` tag contains the correct integer.
- Call with a non-existent `project_folder`; assert returns `False`.

**How**: unit test using `tmp_path` pytest fixture.

### V5 — Pipeline integration

**How**: run `poetry run python ebible_code/ebible.py --test --filter <known_project>`; inspect status CSV; assert `status_inferred_versification` equals the expected integer.

### V6 — `analyse_versification.py`

**How**: run script against `TEST_EBIBLE_DATA_DIR`; assert:
- CSV file created with expected columns (`project_name`, `status`, `best_score`, one score column per versification)
- PNG file created and non-zero bytes
- Stdout contains at least one score band line and a total count
