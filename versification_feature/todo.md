# Versification Feature — To-Do

Consult spec.md before every change. Mark tasks [x] as soon as they are done.
Run `poetry run pytest` after every meaningful commit.

---

## Phase 1a: Safe additions to `settings_file.py` (keep old functions — build stays green)

- [x] 1.1 Add `STANDARD_VERSE_DATA: Dict[VersificationType, Dict[Tuple[str, int], int]]` module-level dict
- [x] 1.2 Add `INVARIANT_CHAPTERS: Set[Tuple[str, int]]` module-level set
- [x] 1.3 Update `populate_standard_versifications()` to compute and populate both new caches
      - Invariant = present in ALL 6 standards AND all agree on verse count (DC-only chapters are NOT invariant)
- [x] 1.4 Add `compute_versification_scores(project_verse_data) -> dict` (shared internal function)
      - Returns: scores, mismatch_counts, project_differentiating_chapters, total_project_chapters
      - Exported so `analyse_versification.py` can import it for `describe_versification_match()`
- [x] 1.5 Add `estimate_versification(project_path: Path) -> VersificationType`
      - Calls compute_versification_scores(); handles missing .vrs → ENGLISH; indistinguishable → ENGLISH
      - Reads VERSIFICATION_UNKNOWN_THRESHOLD from env (strict <); default 0.0
      - Tie-breaks: collect all max-scoring types, prefer ENGLISH
- [x] 1.6 Run `poetry run pytest` — all existing tests still pass (old functions still present)
      - Note: 2 pre-existing failures in get_versification_with_scoring() (abt-maprik, eng-uk-lxx2012); will be fixed in 1b

## Phase 1b: Atomic replacement (one commit — keeps build green throughout)

- [x] 1.7 Rewrite `write_settings_file()` to new signature (returns bool; new XML fields)
- [x] 1.8 Update `ebible.py`: call `estimate_versification()` + new `write_settings_file()` signature
      — pass `languageNameInEnglish` and `title` from row to write_settings_file()
- [x] 1.9 Update `test_settings_file.py`: import `estimate_versification`; assert VersificationType enum
- [x] 1.10 Delete `get_versification_with_scoring()`, `add_settings_file()`, `calculate_similarity_score_for_settings()`
- [x] 1.11 Delete `WEIGHT_BOOK`, `WEIGHT_CHAPTER`, `WEIGHT_VERSE_COUNT` and `VRS_NAME_TO_NUM_STRING`
- [x] 1.12 Run `poetry run pytest` — all tests pass (48 passed, 3 skipped)
      - Also fixed: spurious single-verse placeholder filter in compute_versification_scores()

## Phase 2: Pipeline changes (`ebible.py`)

- [x] 2.1 Update import: add `estimate_versification`; add `VersificationType` from machine.scripture
- [x] 2.2 Update `unzip_and_process_files()`: call `estimate_versification()`, store `.value` in status CSV, use ENGLISH fallback for XML; pass `languageNameInEnglish` and `title` to `write_settings_file()`
- [x] 2.3 Update `update_all_settings()`: same pattern; pass `languageNameInEnglish` and `title` from status CSV row; read old versification from status CSV before overwriting (removes need for old_vals)
- [x] 2.4 Add UNKNOWN count reporting after extraction phase (log warning + filter instructions)
- [x] 2.5 Add `VERSIFICATION_UNKNOWN_THRESHOLD=0.0` to `.env` with comment (see spec §.env Additions)

## Phase 3: Synthetic test fixtures and unit tests

- [x] 3.1 Create `tests/fixtures/versification/` directory
- [x] 3.2 Create `nt_only_invariant.vrs` — NT chapters, all invariant across standards
- [x] 3.3 Create `english_pattern.vrs` — differentiating chapters with English verse counts
- [x] 3.4 Create `vulgate_pattern.vrs` — differentiating chapters with Vulgate verse counts
- [x] 3.5 Create `russian_orthodox_pattern.vrs` — differentiating chapters with Russian Orthodox verse counts
- [x] 3.6 Create `high_mismatch.vrs` — differentiating chapters with verse counts matching no standard
- [x] 3.7 Write unit tests for module-level pre-computation (V1 in spec)
- [x] 3.8 Write parametrized unit tests for `estimate_versification()` using synthetic fixtures (V2 in spec)
- [x] 3.9 Write unit tests for `write_settings_file()`:
      - All VersificationType values 1–6: parse XML, assert correct <Versification> integer
      - Check <Language>, <FullName>, <Name> fields populated correctly
      - Invalid path returns False (V4 in spec)
- [x] 3.10 Update `test_settings_file.py` real-data parametrized tests to assert `VersificationType` enum instead of string (V3 in spec)
- [x] 3.11 Run `poetry run pytest` — all tests pass (67 passed, 3 skipped)

## Phase 4: Analysis script

- [ ] 4.1 Create `ebible_code/analyse_versification.py`
- [x] 4.2 Define `VersificationMatchReport` dataclass here (not in settings_file.py)
- [x] 4.3 Implement `describe_versification_match(project_path) -> VersificationMatchReport`
      — imports scoring internals from settings_file.py; consistent with estimate_versification()
      — generates `notes` string per spec §notes generation rules
- [x] 4.4 Implement project scanning (projects/ and private_projects/ from EBIBLE_DATA_DIR)
- [x] 4.5 Implement CSV output (flattened VersificationMatchReport; one score column per VersificationType)
- [x] 4.6 Implement histogram PNG (matplotlib; vertical line at current threshold)
- [x] 4.7 Implement stdout summary (total, score bands 0.0–1.0 in 0.1 steps, indistinguishable count, below-threshold count)
- [x] 4.8 Write test: run against TEST_EBIBLE_DATA_DIR, assert CSV + PNG created and non-empty (V6 in spec)
- [x] 4.9 Run `poetry run pytest` — all tests pass
      - Fixed typo "Versionification", inflated chapter count in indistinguishable notes, spec.md spurious-chapter filter documentation, and added describe_versification_match() status assertions to V2 tests

## Phase 5: Threshold calibration (requires user input)

- [x] 5.1 Run `poetry run python ebible_code/analyse_versification.py` against full data
- [x] 5.2 Review histogram and CSV with user; identify natural breakpoint in score distribution
      - Clear gap at 0.1–0.3 (zero projects); 94% score ≥ 0.9; low-scoring tail are partial Bibles
- [x] 5.3 Set `VERSIFICATION_UNKNOWN_THRESHOLD` in `.env` to chosen value
      - Set to 0.8; flags 34 projects as UNKNOWN; 1441 matched; 21 indistinguishable
- [x] 5.4 Re-run tests with threshold set; confirm UNKNOWN cases are correctly identified

## Phase 6: Final validation

- [x] 6.1 Run `poetry run pytest` — all tests pass (67 passed, 3 skipped)
- [x] 6.2 Run `poetry run python ebible_code/ebible.py --test` — no regressions
- [x] 6.3 Run `poetry run python ebible_code/ebible.py --update-settings` on a subset — confirmed correct versifications in Settings.xml and status CSV
      - Note: `--filter` does not gate `update_all_settings()`; the function iterates all rows in the status CSV regardless of filter flag
- [x] 6.4 Advisor review complete — no blockers; spec.md updated with spurious-chapter filter docs, V3 threshold note, and float64 note
