# Open Issues

Issues and questions that arose during implementation. Resolve before final release.

---

## 1. LICENCE_TABLE size in README.md

**Issue:** The `{{LICENCE_TABLE}}` placeholder is replaced with a Markdown table of all included translations (potentially 700+ rows). This makes `README.md` very large and may render poorly on HuggingFace.

**Options:**
- Keep the full table (current behaviour).
- Replace with a summary table grouped by licence type (e.g. "Creative Commons 4.0: 450 translations").
- Remove the table entirely and point users to `metadata.parquet`.
- Cap the table at N rows with a "... and N more" note.

**Resolution needed from:** user.

---

## 2. Pre-existing versification test failures

**Issue:** `tests/test_settings_file.py` has 2 pre-existing failures unrelated to this feature:
- `test_get_versification[abt-maprik-English]` — scores as Septuagint instead of English
- `test_get_versification[eng-uk-lxx2012-Septuagint]` — scores as Russian Protestant instead of Septuagint

These existed before any changes on this branch. They are tracked here for completeness.

**Resolution needed from:** separate investigation of `get_versification_with_scoring`.

---

## 3. Double file reads during validation + load

**Issue:** `validate_corpus_files` reads each file to count lines; `load_translation_texts` re-reads the same files to load content. Each valid file is therefore read twice, which doubles I/O for large corpora.

**Impact:** Low — sequential reads of flat text files are fast. Not a correctness issue.

**Options:**
- Accept the double read (current, simpler code).
- Combine validation and loading into a single pass (more complex, harder to test separately).

**Resolution needed from:** performance testing on real corpus.

---

## 4. `sys.path.insert` in `corpus_to_parquet.py`

**Issue:** `corpus_to_parquet.py` inserts `ebible_code/` into `sys.path` at module level to import `clean_range_markers` from `ebible`. This is a workaround for the flat (non-package) layout of `ebible_code/`. If the project is ever reorganised into a proper Python package, this should be replaced with a relative import.

**Resolution needed from:** future package restructure (not urgent).

---

## 5. Interactive prompt incompatible with non-interactive runs

**Issue:** `validate_corpus_files` uses `input()` to ask whether to continue when issues are found. This hangs if `corpus_to_parquet.py` is called from a CI pipeline or a script.

**Current mitigation:** The `_input` parameter allows tests (and future automation) to inject a non-interactive responder.

**Options:**
- Add a `--non-interactive` / `--skip-issues` CLI flag that defaults to "continue" without prompting.
- Document that automated callers should pipe `y` to stdin.

**Resolution needed from:** user (when/if automation is needed).

---

## 6. `README_template.md` citation year

**Issue:** The citation block in `assets/README_template.md` uses `{{{GENERATED_DATE}}}` for the year, which produces the full ISO date (`2026-04-30`) rather than just the year. The BibTeX `year` field conventionally takes a 4-digit year.

**Options:**
- Add a separate `{{GENERATED_YEAR}}` placeholder.
- Extract the year from `GENERATED_DATE` in the render step.
- Leave as-is (non-standard but harmless).

**Resolution needed from:** user preference.
