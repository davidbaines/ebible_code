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

---

## Phase 7: Improved Scoring System

### Motivation

The current scoring uses only differentiating chapters (chapters where standards disagree) as the
denominator. A project with 2 matching differentiating chapters scores 100% identically to one with
300. The new system scores against all non-spurious project chapters, making `score_ENGLISH = 97.3`
mean "97.3% of this project's chapters have the correct verse count for the ENGLISH versification" —
interpretable without domain knowledge.

The distinction between invariant and differentiating chapters is retained as a reporting column
(`total_differentiating_chapters`) so readers can assess how much discriminating signal was
available, but it no longer controls the denominator.

---

### Changes to `compute_versification_scores()`

**New behaviour:** for each non-spurious project chapter `(book, chapter)` with verse count `v`,
compare `v` against `STANDARD_VERSE_DATA[VT][(book, chapter)]` for every standard `VT`. Count
chapters where they differ as `mismatch_counts[VT]`.

Spurious chapter filter (unchanged): exclude `(book, chapter)` where
`project_verse_data[(book, chapter)] == 1` AND `any(std.get((book, chapter), 0) > 1 for std in
STANDARD_VERSE_DATA.values())`.

**Updated return dict:**

| Key | Type | Description |
|-----|------|-------------|
| `scores` | `Dict[VersificationType, float]` | `(total − mismatches) / total` per standard, 0.0–1.0 |
| `mismatch_counts` | `Dict[VersificationType, int]` | Mismatched chapters across **all** non-spurious chapters |
| `project_differentiating_chapters` | `Set[Tuple[str, int]]` | Non-spurious non-invariant chapters (for reporting only) |
| `total_project_chapters` | `int` | Non-spurious chapter count |

---

### Changes to `estimate_versification()`

**Tie-break preference order** (replaces ENGLISH-only preference):

| Rank | VersificationType |
|------|------------------|
| 1 | `ENGLISH` |
| 2 | `ORIGINAL` |
| 3 | `RUSSIAN_PROTESTANT` |
| 4 | `RUSSIAN_ORTHODOX` |
| 5 | `SEPTUAGINT` |
| 6 | `VULGATE` |

**UNKNOWN threshold**: read from env var `VERSIFICATION_MATCH_THRESHOLD` (replaces
`VERSIFICATION_UNKNOWN_THRESHOLD`). Basis: `max(scores.values())` (fraction of all project chapters
matching the best standard). Default `0.0` (UNKNOWN never returned until threshold is set).

**Updated algorithm:**

1. Locate `.vrs`; if missing or unloadable, log warning and return `VersificationType.ENGLISH`.
2. Extract verse data via `get_verse_data_from_vrs_obj()`.
3. Compute spurious chapters and remove them from verse data.
4. If no chapters remain, return `VersificationType.ENGLISH`.
5. Call `compute_versification_scores()`.
6. If `best_score < VERSIFICATION_MATCH_THRESHOLD`: return `VersificationType.ENGLISH`.
7. Collect all `VersificationType`s with `score == best_score`; apply preference order; return winner.

---

### Changes to `analyse_versification.csv`

**Column order:**

```
project_name | best_match | status | matching_chapters | total_project_chapters |
total_differentiating_chapters | mismatch_ORIGINAL | mismatch_SEPTUAGINT | mismatch_VULGATE |
mismatch_ENGLISH | mismatch_RUSSIAN_PROTESTANT | mismatch_RUSSIAN_ORTHODOX |
score_ORIGINAL | score_SEPTUAGINT | score_VULGATE | score_ENGLISH |
score_RUSSIAN_PROTESTANT | score_RUSSIAN_ORTHODOX | notes
```

**Column definitions:**

| Column | Type | Definition |
|--------|------|-----------|
| `matching_chapters` | int | `total_project_chapters − mismatch_[winning_vt]` |
| `mismatch_*` | int | Chapters where project verse count ≠ standard verse count (all non-spurious chapters) |
| `score_*` | float (1 d.p.) | `(total_project_chapters − mismatch_VT) / total_project_chapters × 100` |
| `total_differentiating_chapters` | int | Non-spurious, non-invariant chapters (context only; not used in scoring) |

**Status values:**

| Value | Meaning |
|-------|---------|
| `"matched"` | One versification scored strictly higher than all others |
| `"tied"` | Two or more versifications tied at top score; preference order determined winner |
| `"unknown"` | Best score < `VERSIFICATION_MATCH_THRESHOLD`; `best_match` written as `ENGLISH` as fallback |

Note: the former `"indistinguishable"` status (0 differentiating chapters) is subsumed into
`"tied"` — all versifications tie at the same score (all invariant chapters match all standards
equally), and ENGLISH wins by preference.

---

### Changes to `VersificationMatchReport`

```python
@dataclass
class VersificationMatchReport:
    project_name: str
    best_match: VersificationType       # what estimate_versification() returned
    matching_chapters: int              # total_project_chapters − mismatch[winning_vt]  (replaces best_score)
    scores: dict                        # Dict[VersificationType, float]  0.0–1.0
    mismatch_counts: dict               # Dict[VersificationType, int]  all-chapter mismatches
    total_differentiating_chapters: int
    total_project_chapters: int
    status: str                         # "matched" | "tied" | "unknown"
    notes: str
```

---

### Changes to `describe_versification_match()` — notes generation

- **`"matched"`**: `"Best match: {name} ({score:.1f}% of project chapters match)"`
- **`"tied"` (differentiating chapters exist)**: `"Tied at {score:.1f}%: {tied_names}. {winner} chosen as most common."`
- **`"tied"` (0 differentiating chapters)**: `"All {n} project chapters are the same in all versifications; all versifications match equally ({score:.1f}%). ENGLISH chosen by preference."`
- **`"unknown"`**: `"Best score {score:.1f}% is below threshold {threshold:.1f}%. No versification matched well enough. Settings.xml will use English (4) as a fallback. Mismatch counts per standard: {per_standard_summary}"`

---

### `.env` change

```dotenv
# Threshold below which a translation's best versification match score is treated as UNKNOWN.
# Score = fraction of all project chapters (excluding spurious placeholders) that match the
# best-fitting standard versification. Range: 0.0–1.0.
# Run `poetry run python ebible_code/analyse_versification.py` to inspect the distribution.
VERSIFICATION_MATCH_THRESHOLD=0.0
```

Remove `VERSIFICATION_UNKNOWN_THRESHOLD`.

---

### Verification

#### V7a — `compute_versification_scores()` unit test

Synthetic verse data with known invariant and differentiating chapters. Assert:
- `mismatch_counts` reflects all-chapter mismatches (not differentiating-only)
- `scores` equal `(total − mismatches) / total`
- `total_project_chapters` excludes spurious chapters

#### V7b — Tie-break preference order

Construct synthetic `.vrs` where VULGATE and RUSSIAN_PROTESTANT tie; assert `estimate_versification()` returns `RUSSIAN_PROTESTANT` (ranked higher in preference order).

#### V7c — Updated V2 synthetic fixture tests

Update `high_mismatch.vrs` expected return: with `VERSIFICATION_MATCH_THRESHOLD=0.0`, returns
whichever versification scores highest (not `UNKNOWN`). With threshold set above that best score,
`describe_versification_match()` returns status `"unknown"`.

Update `nt_only_invariant.vrs` expected status: `"tied"` (was `"indistinguishable"`).

#### V7d — V3 real-data tests

All six projects should return the same `VersificationType` as before (scoring changes absolute
values but not ranking for projects with strong differentiating signal). Update
`monkeypatch.setenv` key from `VERSIFICATION_UNKNOWN_THRESHOLD` to `VERSIFICATION_MATCH_THRESHOLD`.

#### V7e — CSV output test (extends V6)

Assert new column headers present and in correct order. Assert `score_*` values are floats with
1 decimal place. Assert `mismatch_*` values ≥ old differentiating-only mismatch values (all-chapter
count is always ≥ differentiating-only count). Assert `status` values are only
`"matched"` | `"tied"` | `"unknown"`.

---

## Phase 8: Threshold Calibration (requires user input)

Run `poetry run python ebible_code/analyse_versification.py` against full data. Review histogram
(vertical line at `VERSIFICATION_MATCH_THRESHOLD`). Identify natural breakpoint in the new
all-chapters score distribution. Set `VERSIFICATION_MATCH_THRESHOLD` in `.env`.
