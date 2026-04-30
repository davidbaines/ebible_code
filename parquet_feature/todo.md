# Parquet Export — To-Do

Consult spec.md before every change. Check off tasks as completed. Run tests after each section.

---

## Phase 1 — `clean_range_markers`

- [x] Add `clean_range_markers(file_path: Path) -> int` to `ebible_code/ebible.py`
  - [x] Top-to-bottom pass: replace `<range>` with `""` when preceded by `""`
  - [x] Handle line-0 edge case (no predecessor → treated as orphaned)
  - [x] Write back in-place, UTF-8, Unix line endings
  - [x] Return count of replacements made
- [x] Insert call in `ebible.py` pipeline: extract → `clean_range_markers` → hash
- [x] Write `tests/test_clean_range_markers.py` with all 9 cases from spec
- [x] Run tests — all pass (9/9)

## Phase 2 — Assets

- [x] Create `assets/parquet_README_template.md` with placeholders defined in spec
  - [x] YAML frontmatter for HuggingFace
  - [x] Static prose sections (description, structure, fields, licence, citation)

## Phase 3 — `corpus_to_parquet.py` rewrite

- [x] Add `argparse` with `--help` text referencing `.env`
- [x] Fix bug: remove non-existent `status_versification` column reference
- [x] Step 1: load vref, build list of 41,899 `"GEN 1:1"` format strings
- [x] Step 2: load and filter `ebible_status.csv` (redistributable, not private, both extract fields set)
- [x] Step 3: pre-flight validation pass
  - [x] Call `clean_range_markers` on each file (backfill)
  - [x] Check file exists
  - [x] Check line count == 41,899
  - [x] tqdm progress bar
  - [x] Collect all failures, print report, prompt user
- [x] Step 4: build `main.parquet`
  - [x] `vref` as column 1
  - [x] Translation columns alphabetical by `translationId`
  - [x] Write with pyarrow / snappy
- [x] Step 5: build `metadata.parquet`
  - [x] Correct column list and order from spec
  - [x] Rename `status_inferred_versification` → `inferred_versification`
  - [x] Only included translations
  - [x] Fix bug: write to `hf_metadata_parquet_file` (not `hf_main_parquet_file`)
  - [x] Write as Parquet (not CSV)
- [x] Step 6: generate `README.md`
  - [x] Read template from `assets/README_template.md`
  - [x] Inject all 5 placeholders
  - [x] Write to output folder
- [x] Step 7: print summary

## Phase 4 — Tests for `corpus_to_parquet`

- [x] Create `tests/test_parquet.py` with all 13 cases from spec
  - [x] Synthetic 3-translation / 5-verse fixture (no real data required)
  - [x] All tests pass (19/19 — spec cases plus additional coverage)

## Phase 5 — Review

- [x] Run full test suite — 28 new tests pass, 2 pre-existing failures in test_settings_file.py
      (unrelated to this feature, tracked in issues.md)
- [x] Spawn sub-agent: "Review spec.md and the current implementation for gaps"
- [x] Address gaps found:
  - [x] Log clean_range_markers return value in corpus_to_parquet validation pass
  - [x] Fix BibTeX year field in README_template.md (add {{GENERATED_YEAR}} placeholder)
  - [x] Fix spec.md verification table test name (test_metadata_columns → test_metadata_columns_order)
  - [x] Add GENERATED_YEAR to render_readme call and test coverage
- [x] Resolve open items in issues.md with user

## Phase 6 — Issues 1 and 6

- [x] Issue 1: Remove LICENCE_TABLE from README
  - [x] Remove `{{LICENCE_TABLE}}` placeholder from `assets/parquet_README_template.md`
  - [x] Replace with prose pointing users to `metadata.parquet`
  - [x] Remove `_make_licence_table()` and `LICENCE_TABLE` from `corpus_to_parquet.py`
  - [x] Update tests to remove `LICENCE_TABLE` from fixtures
  - [x] Run tests — 28/28 pass

- [x] Issue 6: Rename template file and update references
  - [x] Rename `assets/README_template.md` → `assets/parquet_README_template.md` (done by user)
  - [x] Update `corpus_to_parquet.py` template path reference
  - [x] Update `tests/test_parquet.py` template path reference
  - [x] Run tests — 28/28 pass

- [x] Issue 7 (revert): Revert `{{GENERATED_YEAR}}` back to `{{GENERATED_DATE}}` per "leave as-is" decision
  - [x] Revert `assets/parquet_README_template.md` citation year placeholder
  - [x] Remove `GENERATED_YEAR` from render_readme call in `corpus_to_parquet.py`
  - [x] Update test fixture
