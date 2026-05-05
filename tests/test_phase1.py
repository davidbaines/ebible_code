"""Tests for Phase 1: country and continent data enrichment."""
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))

from generate_language_country_continent import (
    apply_overrides,
    build_mapping,
    fetch_scriptures_table,
    load_continent_map,
    load_overrides,
)
from ebible import enrich_with_country_data

# ---------------------------------------------------------------------------
# Minimal fake HTML matching the real ebible.org/Scriptures/ structure
# ---------------------------------------------------------------------------

FAKE_HTML = """
<html><body>
<table><tr><td>nav</td></tr></table>
<table>
  <tr><td><b>Territory</b></td><td>Language</td></tr>
  <tr class="redist">
    <td><a href="country.php?c=GB"><img> United Kingdom</a></td>
    <td><a href="details.php?id=engBBE" class="redist">Basic English</a></td>
    <td><a href="details.php?id=engBBE">Basic English</a></td>
    <td><a href="details.php?id=engBBE">Bible in Basic English</a></td>
    <td><a href="details.php?id=engBBE">Bible in Basic English</a></td>
  </tr>
  <tr class="redist">
    <td><a href="country.php?c=AU"><img> Australia</a></td>
    <td><a href="details.php?id=engPEV" class="redist">Plain English</a></td>
    <td><a href="details.php?id=engPEV">Plain English</a></td>
    <td><a href="details.php?id=engPEV">Plain English Version</a></td>
    <td><a href="details.php?id=engPEV">Plain English Version</a></td>
  </tr>
  <tr class="restricted">
    <td><a href="country.php?c=NA"><img> Namibia</a></td>
    <td><a href="details.php?id=afrnbg" class="restricted">Afrikaans</a></td>
    <td><a href="details.php?id=afrnbg">Afrikaans</a></td>
    <td></td><td></td>
  </tr>
  <tr>
    <td>No link here</td>
    <td><a href="details.php?id=orphan">Orphan</a></td>
  </tr>
</table>
</body></html>
"""

FAKE_CONTINENT_CSV = (
    "CountryCode,ContinentCode\n"
    "GB,EU\n"
    "AU,OC\n"
    "NA,AF\n"   # Namibia — tests that pandas NA-as-NaN bug is handled
    "AN,NA\n"   # Netherlands Antilles — continent code NA (North America)
)


# ---------------------------------------------------------------------------
# Tests for fetch_scriptures_table
# ---------------------------------------------------------------------------

def test_scrape_parses_country_code():
    with patch("generate_language_country_continent.requests.get") as mock_get:
        mock_get.return_value.text = FAKE_HTML
        mock_get.return_value.raise_for_status = lambda: None
        pairs = fetch_scriptures_table("http://fake/")
    country_codes = [p[1] for p in pairs]
    assert "GB" in country_codes
    assert "AU" in country_codes
    assert "NA" in country_codes


def test_scrape_parses_translation_id():
    with patch("generate_language_country_continent.requests.get") as mock_get:
        mock_get.return_value.text = FAKE_HTML
        mock_get.return_value.raise_for_status = lambda: None
        pairs = fetch_scriptures_table("http://fake/")
    translation_ids = [p[0] for p in pairs]
    assert "engBBE" in translation_ids
    assert "engPEV" in translation_ids
    assert "afrnbg" in translation_ids


def test_scrape_skips_header_row():
    with patch("generate_language_country_continent.requests.get") as mock_get:
        mock_get.return_value.text = FAKE_HTML
        mock_get.return_value.raise_for_status = lambda: None
        pairs = fetch_scriptures_table("http://fake/")
    # Header row has no country.php?c= link — should not appear
    translation_ids = [p[0] for p in pairs]
    assert "Language" not in translation_ids
    assert "Territory" not in translation_ids


def test_scrape_skips_row_missing_country_link():
    with patch("generate_language_country_continent.requests.get") as mock_get:
        mock_get.return_value.text = FAKE_HTML
        mock_get.return_value.raise_for_status = lambda: None
        pairs = fetch_scriptures_table("http://fake/")
    # Row with "No link here" has no country link — "orphan" should not appear
    translation_ids = [p[0] for p in pairs]
    assert "orphan" not in translation_ids


# ---------------------------------------------------------------------------
# Tests for load_continent_map
# ---------------------------------------------------------------------------

def test_continent_join(tmp_path):
    path = tmp_path / "cc_test.csv"
    path.write_text(FAKE_CONTINENT_CSV)
    mapping = load_continent_map(path)
    assert mapping["GB"] == "EU"
    assert mapping["AU"] == "OC"
    assert mapping["NA"] == "AF"   # Namibia must not be parsed as NaN
    assert mapping["AN"] == "NA"   # continent code NA (North America) also survives


def test_continent_map_first_occurrence_wins(tmp_path):
    csv = "CountryCode,ContinentCode\nRU,EU\nRU,AS\n"
    path = tmp_path / "cc_dup_test.csv"
    path.write_text(csv)
    mapping = load_continent_map(path)
    assert mapping["RU"] == "EU"


# ---------------------------------------------------------------------------
# Tests for build_mapping
# ---------------------------------------------------------------------------

def test_missing_country_warns(capsys):
    pairs = [("engXXX", "ZZ")]
    mapping = build_mapping(pairs, continent_map={"GB": "EU"})
    captured = capsys.readouterr()
    assert "ZZ" in captured.err
    assert mapping.at[0, "continentCode"] == ""


def test_output_csv_columns():
    pairs = [("engBBE", "GB"), ("engPEV", "AU")]
    continent_map = {"GB": "EU", "AU": "OC"}
    df = build_mapping(pairs, continent_map)
    assert list(df.columns) == ["translationId", "countryCode", "continentCode"]


def test_no_duplicate_translation_ids():
    pairs = [("engBBE", "GB"), ("engPEV", "AU"), ("afrnbg", "NA")]
    continent_map = {"GB": "EU", "AU": "OC", "NA": "AF"}
    df = build_mapping(pairs, continent_map)
    assert df["translationId"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# Tests for enrich_with_country_data
# ---------------------------------------------------------------------------

MAPPING_CSV = (
    "translationId,countryCode,continentCode\n"
    "engBBE,GB,EU\n"
    "engPEV,AU,OC\n"
    "afrnbg,NA,AF\n"
)


@pytest.fixture
def tmp_assets(tmp_path):
    (tmp_path / "language_country_continent.csv").write_text(MAPPING_CSV)
    return tmp_path


def _make_status(rows):
    return pd.DataFrame(rows, columns=["translationId", "countryCode", "continentCode"])


def test_enrich_fills_missing(tmp_assets):
    df = _make_status([
        ("engBBE", float("nan"), float("nan")),
        ("engPEV", float("nan"), float("nan")),
    ])
    result = enrich_with_country_data(df, tmp_assets)
    assert result.loc[result["translationId"] == "engBBE", "countryCode"].iloc[0] == "GB"
    assert result.loc[result["translationId"] == "engBBE", "continentCode"].iloc[0] == "EU"
    assert result.loc[result["translationId"] == "engPEV", "countryCode"].iloc[0] == "AU"


def test_enrich_namibia_na_code(tmp_assets):
    df = _make_status([("afrnbg", float("nan"), float("nan"))])
    result = enrich_with_country_data(df, tmp_assets)
    assert result.loc[result["translationId"] == "afrnbg", "countryCode"].iloc[0] == "NA"
    assert result.loc[result["translationId"] == "afrnbg", "continentCode"].iloc[0] == "AF"


def test_enrich_is_idempotent(tmp_assets):
    df = _make_status([("engBBE", "GB", "EU")])
    result = enrich_with_country_data(df, tmp_assets)
    assert result.loc[result["translationId"] == "engBBE", "countryCode"].iloc[0] == "GB"
    assert result.loc[result["translationId"] == "engBBE", "continentCode"].iloc[0] == "EU"


def test_enrich_missing_file(tmp_path):
    df = _make_status([("engBBE", float("nan"), float("nan"))])
    result = enrich_with_country_data(df, tmp_path)  # tmp_path has no CSV
    assert pd.isna(result.loc[result["translationId"] == "engBBE", "countryCode"].iloc[0])


def test_enrich_unknown_id_warns(tmp_assets, capsys):
    df = _make_status([("unknownXXX", float("nan"), float("nan"))])
    enrich_with_country_data(df, tmp_assets)
    # Warning goes to the logger, not stderr — just check it doesn't raise


# ---------------------------------------------------------------------------
# Tests for country code overrides
# ---------------------------------------------------------------------------

OVERRIDES_CSV = "raw_code,iso_code,notes\nRP,PH,Legacy Philippines code\n"


def test_override_applied(tmp_path):
    path = tmp_path / "country_code_overrides.csv"
    path.write_text(OVERRIDES_CSV)
    overrides = load_overrides(path)
    pairs = [("filRP", "RP"), ("engBBE", "GB")]
    result = apply_overrides(pairs, overrides)
    assert result[0] == ("filRP", "PH")
    assert result[1] == ("engBBE", "GB")


def test_override_full_pipeline(tmp_path):
    # RP → PH → AS in a complete end-to-end join
    overrides_path = tmp_path / "country_code_overrides.csv"
    overrides_path.write_text(OVERRIDES_CSV)
    continent_path = tmp_path / "cc.csv"
    continent_path.write_text("CountryCode,ContinentCode\nPH,AS\nGB,EU\n")

    overrides = load_overrides(overrides_path)
    pairs = [("filRP", "RP"), ("engBBE", "GB")]
    pairs = apply_overrides(pairs, overrides)
    continent_map = load_continent_map(continent_path)
    df = build_mapping(pairs, continent_map)

    rp_row = df[df["translationId"] == "filRP"].iloc[0]
    assert rp_row["countryCode"] == "PH"
    assert rp_row["continentCode"] == "AS"


def test_override_missing_file_ok(tmp_path):
    # No overrides file — load_overrides returns empty dict, no exception
    overrides = load_overrides(tmp_path / "nonexistent.csv")
    assert overrides == {}
    pairs = [("engBBE", "GB")]
    result = apply_overrides(pairs, overrides)
    assert result == [("engBBE", "GB")]


def test_override_unknown_after_override_warns(capsys):
    # Code not in overrides and not in continent map still triggers warning in build_mapping
    pairs = [("xyzUNK", "ZZ")]
    df = build_mapping(pairs, continent_map={"GB": "EU"})
    captured = capsys.readouterr()
    assert "ZZ" in captured.err
    assert df.at[0, "continentCode"] == ""
