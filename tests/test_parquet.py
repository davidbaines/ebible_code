"""Tests for corpus_to_parquet.py."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))
from corpus_to_parquet import (
    METADATA_SOURCE_COLUMNS,
    VREF_LENGTH,
    build_main_dataframe,
    build_metadata_dataframe,
    render_readme,
    validate_corpus_files,
    load_and_filter_metadata,
    load_translation_texts,
    _make_licence_table,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VREF_5 = ["GEN 1:1", "GEN 1:2", "GEN 1:3", "GEN 1:4", "GEN 1:5"]

TRANS_A = ["In the beginning", "<range>", "The earth was", "", "And God said"]
TRANS_B = ["Au commencement", "", "<range>", "Et Dieu dit", "La lumière"]
TRANS_C = ["", "<range>", "", "Text C4", "Text C5"]


def make_corpus_files(tmp_path, translations: dict, line_count: int = 5) -> Path:
    """Write corpus txt files; returns ebible_data_dir."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for tid, lines in translations.items():
        p = corpus_dir / f"{tid}.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def make_metadata_df(tmp_path, translations: dict, extra_cols: dict = None) -> Path:
    """Write a minimal ebible_status.csv and return its path."""
    rows = []
    for tid, corpus_path in translations.items():
        row = {
            "translationId": tid,
            "languageCode": tid[:3],
            "languageName": f"Language {tid}",
            "languageNameInEnglish": f"Language {tid} EN",
            "dialect": "",
            "homeDomain": "",
            "title": f"Title {tid}",
            "description": "",
            "Redistributable": "True",
            "Copyright": "",
            "UpdateDate": "2024-01-01",
            "publicationURL": "",
            "OTbooks": 0,
            "OTchapters": 0,
            "OTverses": 0,
            "NTbooks": 1,
            "NTchapters": 1,
            "NTverses": 5,
            "DCbooks": 0,
            "DCchapters": 0,
            "DCverses": 0,
            "FCBHID": "",
            "Certified": "",
            "inScript": "True",
            "swordName": "",
            "rodCode": "",
            "textDirection": "ltr",
            "downloadable": "True",
            "font": "",
            "shortTitle": tid,
            "PODISBN": "",
            "script": "Latin",
            "sourceDate": "2024-01-01",
            "licence_Vernacular_Title": "",
            "licence_Licence_Type": "Creative Commons",
            "licence_Licence_Version": "4.0",
            "licence_CC_Licence_Link": "https://creativecommons.org/licenses/by/4.0/",
            "licence_Copyright_Holder": "Holder",
            "licence_Copyright_Years": "2024",
            "licence_Translation_by": "Org",
            "status_inferred_versification": "4",
            "status_extract_path": str(corpus_path),
            "status_extract_date": "2024-01-01",
            "status_extract_hash": "abc123",
        }
        if extra_cols:
            row.update(extra_cols.get(tid, {}))
        rows.append(row)

    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(exist_ok=True)
    csv_path = meta_dir / "ebible_status.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# build_main_dataframe
# ---------------------------------------------------------------------------

def test_vref_is_first_column():
    data = {"aaa-A": TRANS_A, "bbb-B": TRANS_B}
    df = build_main_dataframe(VREF_5, data)
    assert df.columns[0] == "vref"


def test_translations_alphabetical():
    data = {"zzz-Z": TRANS_B, "aaa-A": TRANS_A, "mmm-M": TRANS_C}
    df = build_main_dataframe(VREF_5, data)
    translation_cols = list(df.columns[1:])
    assert translation_cols == sorted(translation_cols)


def test_row_count():
    data = {"aaa-A": TRANS_A}
    df = build_main_dataframe(VREF_5, data)
    assert len(df) == len(VREF_5)


def test_empty_and_range_preserved(tmp_path):
    data = {"aaa-A": TRANS_A}
    df = build_main_dataframe(VREF_5, data)
    # Row index 1: second line of TRANS_A is "<range>"
    assert df.iloc[1]["aaa-A"] == "<range>"
    # Row index 3: fourth line is ""
    assert df.iloc[3]["aaa-A"] == ""


def test_vref_values():
    data = {"aaa-A": TRANS_A}
    df = build_main_dataframe(VREF_5, data)
    assert list(df["vref"]) == VREF_5


# ---------------------------------------------------------------------------
# build_metadata_dataframe
# ---------------------------------------------------------------------------

def _make_full_meta(included_ids):
    rows = []
    for tid in included_ids + ["excluded-X"]:
        rows.append({
            "translationId": tid,
            "languageCode": tid[:3],
            "status_inferred_versification": "4",
            "licence_Licence_Type": "CC",
        })
    return pd.DataFrame(rows)


def test_metadata_only_included_translations():
    full = _make_full_meta(["aaa-A", "bbb-B"])
    meta = build_metadata_dataframe(full, ["aaa-A", "bbb-B"])
    assert set(meta["translationId"]) == {"aaa-A", "bbb-B"}
    assert "excluded-X" not in meta["translationId"].values


def test_inferred_versification_renamed():
    full = _make_full_meta(["aaa-A"])
    meta = build_metadata_dataframe(full, ["aaa-A"])
    assert "inferred_versification" in meta.columns
    assert "status_inferred_versification" not in meta.columns


def test_metadata_absent_columns_skipped():
    """Columns in METADATA_SOURCE_COLUMNS absent from the CSV are silently skipped."""
    full = pd.DataFrame([{"translationId": "aaa-A", "languageCode": "aaa",
                          "status_inferred_versification": "4"}])
    meta = build_metadata_dataframe(full, ["aaa-A"])
    assert "translationId" in meta.columns
    assert "languageCode" in meta.columns
    # Columns not present in source are absent (no KeyError)
    assert "OTbooks" not in meta.columns


# ---------------------------------------------------------------------------
# render_readme
# ---------------------------------------------------------------------------

def test_readme_placeholders_replaced():
    template = "Count: {{TRANSLATION_COUNT}} Date: {{GENERATED_DATE}}"
    result = render_readme(template, {"TRANSLATION_COUNT": 42, "GENERATED_DATE": "2024-01-01"})
    assert "{{" not in result
    assert "42" in result
    assert "2024-01-01" in result


def test_readme_translation_count():
    template = "Translations: {{TRANSLATION_COUNT}}"
    result = render_readme(template, {"TRANSLATION_COUNT": 123})
    assert "123" in result


def test_readme_all_placeholders_replaced():
    template_path = Path(__file__).parent.parent / "assets" / "README_template.md"
    if not template_path.exists():
        pytest.skip("README template not found")
    template = template_path.read_text(encoding="utf-8")
    result = render_readme(template, {
        "TRANSLATION_COUNT": 10,
        "LANGUAGE_COUNT": 5,
        "VERSE_COUNT": VREF_LENGTH,
        "GENERATED_DATE": "2024-01-01",
        "GENERATED_YEAR": 2024,
        "LICENCE_TABLE": "| id | type |\n|---|---|\n| aaa | CC |",
    })
    assert "{{" not in result


# ---------------------------------------------------------------------------
# validate_corpus_files (file-system tests using tmp_path)
# ---------------------------------------------------------------------------

def _make_candidates(corpus_paths: dict) -> pd.DataFrame:
    rows = [{"translationId": tid, "status_extract_path": str(p)}
            for tid, p in corpus_paths.items()]
    return pd.DataFrame(rows)


def test_valid_files_pass(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    p = corpus_dir / "aaa-A.txt"
    p.write_text("\n".join(["line"] * VREF_LENGTH) + "\n", encoding="utf-8")
    candidates = _make_candidates({"aaa-A": p})
    valid, skipped = validate_corpus_files(candidates, tmp_path, VREF_LENGTH, _input=lambda _: "y")
    assert "aaa-A" in valid
    assert skipped == []


def test_missing_file_skipped(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    missing = corpus_dir / "missing.txt"
    candidates = _make_candidates({"missing-X": missing})
    valid, skipped = validate_corpus_files(candidates, tmp_path, VREF_LENGTH, _input=lambda _: "y")
    assert valid == []
    assert "missing-X" in skipped


def test_wrong_line_count_skipped(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    p = corpus_dir / "bad-B.txt"
    p.write_text("\n".join(["line"] * 100) + "\n", encoding="utf-8")
    candidates = _make_candidates({"bad-B": p})
    valid, skipped = validate_corpus_files(candidates, tmp_path, VREF_LENGTH, _input=lambda _: "y")
    assert valid == []
    assert "bad-B" in skipped


def test_user_abort_on_issues(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    missing = corpus_dir / "gone.txt"
    candidates = _make_candidates({"gone-X": missing})
    with pytest.raises(SystemExit):
        validate_corpus_files(candidates, tmp_path, VREF_LENGTH, _input=lambda _: "n")


def test_clean_range_called_in_validation(tmp_path):
    """Orphaned <range> markers in an existing file are cleaned during validation."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    p = corpus_dir / "aaa-A.txt"
    lines = [""] + ["<range>"] + ["line"] * (VREF_LENGTH - 2)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    candidates = _make_candidates({"aaa-A": p})
    validate_corpus_files(candidates, tmp_path, VREF_LENGTH, _input=lambda _: "y")
    result = [line.rstrip("\r\n") for line in p.read_text(encoding="utf-8").splitlines()]
    assert result[1] == ""  # orphaned <range> was cleaned


# ---------------------------------------------------------------------------
# Integration: output files are overwritten and contain correct data
# ---------------------------------------------------------------------------

def test_output_overwritten(tmp_path):
    """Re-running build_main_dataframe with different data produces a different result."""
    df1 = build_main_dataframe(VREF_5, {"aaa-A": TRANS_A})
    df2 = build_main_dataframe(VREF_5, {"aaa-A": TRANS_B})
    assert not df1.equals(df2)


def test_metadata_columns_order(tmp_path):
    """Columns in metadata output follow METADATA_SOURCE_COLUMNS order (renamed)."""
    rows = [{c: "val" for c in METADATA_SOURCE_COLUMNS}]
    rows[0]["translationId"] = "aaa-A"
    full = pd.DataFrame(rows)
    meta = build_metadata_dataframe(full, ["aaa-A"])
    expected_cols = [
        "inferred_versification" if c == "status_inferred_versification" else c
        for c in METADATA_SOURCE_COLUMNS
    ]
    assert list(meta.columns) == expected_cols


def test_load_and_filter_metadata_excludes_private(tmp_path):
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    csv_path = meta_dir / "ebible_status.csv"
    rows = [
        {"translationId": "pub-A", "Redistributable": "True",
         "status_extract_path": "/corpus/pub-A.txt",
         "status_extract_date": "2024-01-01", "status_extract_hash": "abc"},
        {"translationId": "prv-B", "Redistributable": "True",
         "status_extract_path": "/private_corpus/prv-B.txt",
         "status_extract_date": "2024-01-01", "status_extract_hash": "def"},
        {"translationId": "notredist-C", "Redistributable": "False",
         "status_extract_path": "/corpus/notredist-C.txt",
         "status_extract_date": "2024-01-01", "status_extract_hash": "ghi"},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    result = load_and_filter_metadata(csv_path)
    assert list(result["translationId"]) == ["pub-A"]
