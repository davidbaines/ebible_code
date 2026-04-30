"""
Unit tests for versification estimation:
  V1 — module-level pre-computation (STANDARD_VERSE_DATA, INVARIANT_CHAPTERS)
  V2 — estimate_versification() with synthetic fixtures
  V4 — write_settings_file() unit tests
"""

import shutil
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from machine.scripture import VersificationType
from ebible_code.settings_file import (
    INVARIANT_CHAPTERS,
    STANDARD_VERSE_DATA,
    estimate_versification,
    write_settings_file,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "versification"


# ---------------------------------------------------------------------------
# V1 — Module-level pre-computation
# ---------------------------------------------------------------------------

def test_standard_verse_data_has_six_keys():
    assert len(STANDARD_VERSE_DATA) == 6
    for vt in STANDARD_VERSE_DATA:
        assert isinstance(vt, VersificationType)
        assert vt != VersificationType.UNKNOWN


def test_standard_verse_data_keys_are_versification_types():
    expected = {
        VersificationType.ORIGINAL,
        VersificationType.SEPTUAGINT,
        VersificationType.VULGATE,
        VersificationType.ENGLISH,
        VersificationType.RUSSIAN_PROTESTANT,
        VersificationType.RUSSIAN_ORTHODOX,
    }
    assert set(STANDARD_VERSE_DATA.keys()) == expected


def test_invariant_chapters_is_nonempty():
    assert len(INVARIANT_CHAPTERS) > 100, (
        f"Expected several hundred invariant chapters, got {len(INVARIANT_CHAPTERS)}"
    )


def test_invariant_chapters_all_present_in_every_standard():
    for bc in INVARIANT_CHAPTERS:
        verse_counts = [STANDARD_VERSE_DATA[vt].get(bc) for vt in STANDARD_VERSE_DATA]
        assert all(v is not None for v in verse_counts), (
            f"Invariant chapter {bc} is missing from at least one standard versification"
        )
        assert len(set(verse_counts)) == 1, (
            f"Invariant chapter {bc} has different verse counts across standards: {verse_counts}"
        )


# ---------------------------------------------------------------------------
# V2 — estimate_versification() with synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def project_from_fixture(tmp_path):
    """Returns a factory: project_from_fixture(fixture_name) → project Path."""
    def _factory(fixture_name: str) -> Path:
        proj_name = fixture_name.removesuffix(".vrs")
        proj_dir = tmp_path / proj_name
        proj_dir.mkdir()
        shutil.copy(FIXTURES_DIR / fixture_name, proj_dir / f"{proj_name}.vrs")
        return proj_dir
    return _factory


synthetic_cases = [
    ("nt_only_invariant.vrs",        VersificationType.ENGLISH),
    ("english_pattern.vrs",          VersificationType.ENGLISH),
    ("vulgate_pattern.vrs",          VersificationType.VULGATE),
    ("russian_orthodox_pattern.vrs", VersificationType.RUSSIAN_ORTHODOX),
]

@pytest.mark.parametrize("fixture_name, expected", synthetic_cases)
def test_estimate_versification_fixture(fixture_name, expected, project_from_fixture, monkeypatch):
    monkeypatch.setenv("VERSIFICATION_UNKNOWN_THRESHOLD", "0.0")
    proj_dir = project_from_fixture(fixture_name)
    actual = estimate_versification(proj_dir)
    assert actual == expected, (
        f"Fixture '{fixture_name}': expected {expected.name}, got {actual.name}"
    )


def test_estimate_versification_high_mismatch_returns_unknown(project_from_fixture, monkeypatch):
    monkeypatch.setenv("VERSIFICATION_UNKNOWN_THRESHOLD", "0.3")
    proj_dir = project_from_fixture("high_mismatch.vrs")
    actual = estimate_versification(proj_dir)
    assert actual == VersificationType.UNKNOWN, (
        f"Expected UNKNOWN for high_mismatch fixture at threshold=0.3, got {actual.name}"
    )


def test_estimate_versification_missing_vrs_returns_english(tmp_path):
    proj_dir = tmp_path / "no_vrs_project"
    proj_dir.mkdir()
    # No .vrs file present
    result = estimate_versification(proj_dir)
    assert result == VersificationType.ENGLISH


def test_estimate_versification_nonexistent_folder(tmp_path):
    proj_dir = tmp_path / "nonexistent"
    # Folder doesn't exist
    result = estimate_versification(proj_dir)
    assert result == VersificationType.ENGLISH


# ---------------------------------------------------------------------------
# V4 — write_settings_file() unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vrs_type", [
    VersificationType.ORIGINAL,
    VersificationType.SEPTUAGINT,
    VersificationType.VULGATE,
    VersificationType.ENGLISH,
    VersificationType.RUSSIAN_PROTESTANT,
    VersificationType.RUSSIAN_ORTHODOX,
])
def test_write_settings_file_versification_integer(tmp_path, vrs_type):
    proj_dir = tmp_path / "test_proj"
    proj_dir.mkdir()
    success = write_settings_file(proj_dir, "eng", vrs_type)
    assert success is True
    settings_file = proj_dir / "Settings.xml"
    assert settings_file.exists()
    root = ET.parse(settings_file).getroot()
    assert root.findtext("Versification") == str(vrs_type.value), (
        f"Expected {vrs_type.value} for {vrs_type.name}"
    )


def test_write_settings_file_xml_fields(tmp_path):
    proj_dir = tmp_path / "myproject"
    proj_dir.mkdir()
    success = write_settings_file(
        proj_dir, "fra", VersificationType.VULGATE,
        language_name_in_english="French",
        full_name="La Sainte Bible",
    )
    assert success is True
    root = ET.parse(proj_dir / "Settings.xml").getroot()
    assert root.findtext("Language") == "French"
    assert root.findtext("Encoding") == "65001"
    assert root.findtext("FullName") == "La Sainte Bible"
    assert root.findtext("Name") == "myproject"
    assert root.findtext("Versification") == str(VersificationType.VULGATE.value)
    assert root.findtext("LanguageIsoCode") == "fra:::"
    naming = root.find("Naming")
    assert naming is not None
    assert naming.get("BookNameForm") == "41MAT"
    assert naming.get("PostPart") == "myp.SFM"
    assert naming.get("PrePart") == ""


def test_write_settings_file_returns_false_for_nonexistent_folder(tmp_path):
    proj_dir = tmp_path / "does_not_exist"
    result = write_settings_file(proj_dir, "eng", VersificationType.ENGLISH)
    assert result is False
