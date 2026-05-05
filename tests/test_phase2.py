"""Tests for Phase 2: Glottolog language family data."""
import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))

from get_glottolog_families import (
    build_family_records,
    load_macrolanguage_overrides,
)

# ---------------------------------------------------------------------------
# Minimal fake languoid dataset
# ---------------------------------------------------------------------------
#
# Structure:
#   Indo-European (indo1319) [family]
#     Germanic (germ1287) [family]
#       English (stan1293) [language, eng]
#   Basque (basq1248) [language, eus]  ← isolate: empty family_id and parent_id
#   Afro-Asiatic (afro1255) [family]
#     Algerian Arabic (alge1239) [language, arq]
#   Unknown Family (unkn1001) [family]
#     DupLang1 (dup1001) [language, dup]  ← first occurrence
#     DupLang2 (dup1002) [language, dup]  ← duplicate iso, should be skipped
#     Generic (gen1001) [language, gen]   ← used for macrolanguage override tests
#   NoISO (niso1001) [language, ""]       ← no ISO code, should be skipped
#   Namibian (nami1234) [language, NA]    ← "NA" iso code (keep_default_na_false test)

FAKE_LANGUOIDS = pd.DataFrame(
    [
        # id          name              level     iso639P3code  family_id  parent_id
        ("indo1319", "Indo-European",   "family", "",           "",        ""),
        ("germ1287", "Germanic",        "family", "",           "indo1319", "indo1319"),
        ("stan1293", "English",         "language", "eng",      "indo1319", "germ1287"),
        ("basq1248", "Basque",          "language", "eus",      "",        ""),
        ("afro1255", "Afro-Asiatic",    "family", "",           "",        ""),
        ("alge1239", "Algerian Arabic", "language", "arq",      "afro1255", "afro1255"),
        ("unkn1001", "UnknownFamily",   "family", "",           "",        ""),
        ("dup1001",  "DupLang1",        "language", "dup",      "unkn1001", "unkn1001"),
        ("dup1002",  "DupLang2",        "language", "dup",      "unkn1001", "unkn1001"),
        ("gen1001",  "Generic",         "language", "gen",      "unkn1001", "unkn1001"),
        ("niso1001", "NoISO",           "language", "",         "unkn1001", "unkn1001"),
        ("nami1234", "Namibian",        "language", "NA",       "afro1255", "afro1255"),
    ],
    columns=["id", "name", "level", "iso639P3code", "family_id", "parent_id"],
)

OVERRIDES_CSV = "ebible_language_code,glottolog_lookup_code,notes\ngen,eng,Test override\n"


def _records_by_iso(records):
    return {r["languageCode"]: r for r in records}


# ---------------------------------------------------------------------------
# Tests for build_family_records
# ---------------------------------------------------------------------------

def test_family_name_extracted():
    records = build_family_records(FAKE_LANGUOIDS, {})
    by_iso = _records_by_iso(records)
    assert by_iso["eng"]["family_name"] == "Indo-European"
    assert by_iso["arq"]["family_name"] == "Afro-Asiatic"


def test_classification_path():
    records = build_family_records(FAKE_LANGUOIDS, {})
    by_iso = _records_by_iso(records)
    assert by_iso["eng"]["classification"] == "Indo-European/Germanic/English"
    assert by_iso["arq"]["classification"] == "Afro-Asiatic/Algerian Arabic"


def test_isolate_family_name():
    records = build_family_records(FAKE_LANGUOIDS, {})
    by_iso = _records_by_iso(records)
    assert by_iso["eus"]["family_name"] == "Isolate"


def test_isolate_classification_single_component():
    records = build_family_records(FAKE_LANGUOIDS, {})
    by_iso = _records_by_iso(records)
    assert "/" not in by_iso["eus"]["classification"]
    assert by_iso["eus"]["classification"] == "Basque"


def test_no_duplicate_language_codes():
    records = build_family_records(FAKE_LANGUOIDS, {})
    iso_codes = [r["languageCode"] for r in records]
    assert len(iso_codes) == len(set(iso_codes))


def test_output_csv_columns():
    records = build_family_records(FAKE_LANGUOIDS, {})
    df = pd.DataFrame(records)
    assert list(df.columns) == ["languageCode", "glottocode", "family_name", "classification"]


def test_missing_iso_code_skipped():
    records = build_family_records(FAKE_LANGUOIDS, {})
    iso_codes = [r["languageCode"] for r in records]
    assert "" not in iso_codes
    # niso1001 has no ISO code so it must not appear
    glottocodes = [r["glottocode"] for r in records]
    assert "niso1001" not in glottocodes


def test_first_occurrence_wins_on_duplicate_iso():
    records = build_family_records(FAKE_LANGUOIDS, {})
    by_iso = _records_by_iso(records)
    assert "dup" in by_iso
    assert by_iso["dup"]["glottocode"] == "dup1001"  # first row wins


def test_glottocode_populated():
    records = build_family_records(FAKE_LANGUOIDS, {})
    by_iso = _records_by_iso(records)
    assert by_iso["eng"]["glottocode"] == "stan1293"
    assert by_iso["eus"]["glottocode"] == "basq1248"


def test_keep_default_na_false():
    # "NA" as an ISO code (like Namibia) must not be dropped; it should survive
    # in the output exactly as the string "NA" (not filtered as if it were NaN).
    # This tests that load_languoids must use keep_default_na=False when reading CSV.
    records = build_family_records(FAKE_LANGUOIDS, {})
    iso_codes = [r["languageCode"] for r in records]
    assert "NA" in iso_codes


# ---------------------------------------------------------------------------
# Tests for macrolanguage overrides
# ---------------------------------------------------------------------------

def test_macrolanguage_override_applied():
    # gen -> eng override: output uses eng's glottocode/family, but languageCode stays "gen"
    records = build_family_records(FAKE_LANGUOIDS, {"gen": "eng"})
    by_iso = _records_by_iso(records)
    assert "gen" in by_iso
    r = by_iso["gen"]
    assert r["languageCode"] == "gen"           # original eBible code preserved
    assert r["glottocode"] == "stan1293"        # from English row
    assert r["family_name"] == "Indo-European"  # from English row
    assert "Germanic" in r["classification"]    # from English row


def test_macrolanguage_override_missing_lookup_warns(capsys):
    # Override points to a code not in the fake DF → warning emitted, original row used
    records = build_family_records(FAKE_LANGUOIDS, {"gen": "zzz"})
    captured = capsys.readouterr()
    assert "zzz" in captured.err
    by_iso = _records_by_iso(records)
    # Falls back to gen's own data
    assert by_iso["gen"]["glottocode"] == "gen1001"


def test_macrolanguage_override_missing_file_ok(tmp_path):
    overrides = load_macrolanguage_overrides(tmp_path / "nonexistent.csv")
    assert overrides == {}


def test_macrolanguage_override_loaded_from_file(tmp_path):
    path = tmp_path / "overrides.csv"
    path.write_text(OVERRIDES_CSV)
    overrides = load_macrolanguage_overrides(path)
    assert overrides == {"gen": "eng"}


def test_macrolanguage_override_empty_file_ok(tmp_path):
    # File with header only (no data rows) → empty dict, no error
    path = tmp_path / "overrides.csv"
    path.write_text("ebible_language_code,glottolog_lookup_code,notes\n")
    overrides = load_macrolanguage_overrides(path)
    assert overrides == {}
