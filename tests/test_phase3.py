"""Tests for Phase 3: Dataloader script."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))

from dataloader import (
    CustomFilterSpec,
    FilterSpec,
    apply_custom_filter,
    apply_filter,
    apply_filters,
    apply_split,
    build_metadata_table,
    build_text_table,
    compute_split_masks,
    parse_filter_tokens,
    parse_splits_csv,
    translations_excluded_by_filter,
    write_output,
)

# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

METADATA_DF = pd.DataFrame(
    {
        "translationId": ["engBBE", "fra1910", "deuLUT", "spaRV", "afrAFR"],
        "languageCode": ["eng", "fra", "deu", "spa", "afr"],
        "countryCode": ["GB", "FR", "DE", "ES", "ZA"],
        "continentCode": ["EU", "EU", "EU", "EU", "AF"],
        "Redistributable": ["True", "True", "False", "True", "True"],
    }
)

VREFS = [f"GEN 1:{i}" for i in range(1, 6)] + [f"EXO 1:{i}" for i in range(1, 6)]

MAIN_DF = pd.DataFrame(
    {
        "vref":   VREFS,
        "engBBE": ["eng1", "eng2", "",     "eng4", "eng5", "eng6", "eng7", "eng8", "",     "eng10"],
        "fra1910":["fra1", "",     "fra3", "fra4", "fra5", "fra6", "",     "fra8", "fra9", "fra10"],
        "deuLUT": ["deu1", "deu2", "deu3", "",     "deu5", "deu6", "deu7", "",     "deu9", "deu10"],
        "spaRV":  ["spa1", "spa2", "spa3", "spa4", "",     "spa6", "spa7", "spa8", "spa9", ""],
        "afrAFR": ["afr1", "",     "afr3", "afr4", "afr5", "",     "afr7", "afr8", "afr9", "afr10"],
    }
)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

def test_filter_exact():
    spec = parse_filter_tokens(["Redistributable", "True"])
    result = apply_filter(METADATA_DF, spec)
    assert set(result["translationId"]) == {"engBBE", "fra1910", "spaRV", "afrAFR"}
    assert "deuLUT" not in result["translationId"].values


def test_filter_contains():
    spec = parse_filter_tokens(["continentCode", "contains", "EU"])
    result = apply_filter(METADATA_DF, spec)
    assert set(result["translationId"]) == {"engBBE", "fra1910", "deuLUT", "spaRV"}


def test_filter_not():
    spec = parse_filter_tokens(["languageCode", "not", "eng"])
    result = apply_filter(METADATA_DF, spec)
    assert "engBBE" not in result["translationId"].values
    assert len(result) == 4


def test_filter_in():
    spec = parse_filter_tokens(["continentCode", "in", "EU", "AF"])
    result = apply_filter(METADATA_DF, spec)
    assert len(result) == 5


def test_filter_and_combined():
    spec1 = parse_filter_tokens(["continentCode", "EU"])
    spec2 = parse_filter_tokens(["Redistributable", "True"])
    result = apply_filters(METADATA_DF, [spec1, spec2])
    assert set(result["translationId"]) == {"engBBE", "fra1910", "spaRV"}


def test_filter_unknown_column_raises():
    spec = parse_filter_tokens(["noSuchColumn", "someValue"])
    with pytest.raises(ValueError, match="Unknown filter column"):
        apply_filter(METADATA_DF, spec)


def test_custom_filter_join(tmp_path):
    glotto = tmp_path / "glotto.csv"
    glotto.write_text(
        "languageCode,family_name\neng,Indo-European\nfra,Indo-European\n"
    )
    spec = CustomFilterSpec(file=str(glotto), join_on="languageCode")
    result = apply_custom_filter(METADATA_DF, spec)
    assert set(result["translationId"]) == {"engBBE", "fra1910"}
    assert "family_name" in result.columns


def test_custom_filter_bad_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("wrongCol,value\neng,something\n")
    spec = CustomFilterSpec(file=str(csv_path), join_on="languageCode")
    with pytest.raises(ValueError, match="Join column"):
        apply_custom_filter(METADATA_DF, spec)


# ---------------------------------------------------------------------------
# Load / table-building tests
# ---------------------------------------------------------------------------

def test_load_text_table_shape():
    result = build_text_table(MAIN_DF, ["engBBE", "fra1910"])
    assert result.shape == (10, 3)  # 10 vrefs × (vref + 2 translations)


def test_load_empty_strings_not_nan():
    result = build_text_table(MAIN_DF, ["engBBE"])
    assert result["engBBE"].isna().sum() == 0
    assert (result["engBBE"] == "").any()


def test_load_vref_first_column():
    result = build_text_table(MAIN_DF, ["engBBE", "fra1910"])
    assert result.columns[0] == "vref"


def test_metadata_table_shape():
    cols = ["translationId", "languageCode", "countryCode", "continentCode", "Redistributable"]
    result = build_metadata_table(METADATA_DF, ["engBBE", "fra1910", "deuLUT"], cols)
    assert result.shape == (3, 5)


def test_metadata_columns_flag():
    result = build_metadata_table(METADATA_DF, ["engBBE", "fra1910"], ["translationId", "languageCode"])
    assert list(result.columns) == ["translationId", "languageCode"]
    assert result.shape == (2, 2)


def test_no_metadata_flag():
    from dataloader import build_parser
    parser = build_parser()
    args = parser.parse_args(["load", "--no-metadata"])
    assert args.no_metadata is True


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

SPLITS_CSV = (
    "translationId,book,chapter,verse,split\n"
    "engBBE,GEN,,,train\n"
    "fra1910,EXO,,,test\n"
    "engBBE,EXO,,,test\n"
)


def _write_splits(content: str, tmp_path: Path) -> pd.DataFrame:
    p = tmp_path / "splits.csv"
    p.write_text(content)
    return parse_splits_csv(p)


def test_split_assigns_correctly(tmp_path):
    splits_df = _write_splits(SPLITS_CSV, tmp_path)
    tids = ["engBBE", "fra1910"]
    text_table = build_text_table(MAIN_DF, tids)
    masks = compute_split_masks(text_table["vref"], splits_df, tids)

    # train: engBBE GEN rows preserved; engBBE EXO rows zeroed
    train_text = apply_split(text_table, masks["train"])
    exo_mask = text_table["vref"].str.startswith("EXO")
    assert (train_text.loc[exo_mask, "engBBE"] == "").all()

    # test: engBBE EXO rows preserved; engBBE GEN rows zeroed
    test_text = apply_split(text_table, masks["test"])
    gen_mask = text_table["vref"].str.startswith("GEN")
    assert (test_text.loc[gen_mask, "engBBE"] == "").all()


def test_split_omission_translation_only(tmp_path):
    content = "translationId,split\nengBBE,train\n"
    splits_df = _write_splits(content, tmp_path)
    tids = ["engBBE"]
    text_table = build_text_table(MAIN_DF, tids)
    masks = compute_split_masks(text_table["vref"], splits_df, tids)
    train_text = apply_split(text_table, masks["train"])
    # No rows should be zeroed — full Bible assigned to train
    pd.testing.assert_series_equal(train_text["engBBE"], text_table["engBBE"])


def test_split_omission_book_only(tmp_path):
    content = "translationId,book,split\nengBBE,GEN,train\n"
    splits_df = _write_splits(content, tmp_path)
    tids = ["engBBE"]
    text_table = build_text_table(MAIN_DF, tids)
    masks = compute_split_masks(text_table["vref"], splits_df, tids)
    train_text = apply_split(text_table, masks["train"])
    exo_mask = text_table["vref"].str.startswith("EXO")
    gen_mask = text_table["vref"].str.startswith("GEN")
    assert (train_text.loc[exo_mask, "engBBE"] == "").all()
    pd.testing.assert_series_equal(
        train_text.loc[gen_mask, "engBBE"].reset_index(drop=True),
        text_table.loc[gen_mask, "engBBE"].reset_index(drop=True),
    )


def test_split_filter_first():
    # deuLUT has Redistributable=False; after filtering it should be absent
    filter_spec = parse_filter_tokens(["Redistributable", "True"])
    filtered = apply_filter(METADATA_DF, filter_spec)
    filtered_tids = set(filtered["translationId"].tolist())
    assert "deuLUT" not in filtered_tids
    # Text table built from filtered set should not have deuLUT column
    text_table = build_text_table(MAIN_DF, list(filtered_tids))
    assert "deuLUT" not in text_table.columns


def test_split_summary_warns_filtered(capsys, tmp_path):
    content = "translationId,split\nengBBE,train\ndeuLUT,test\n"
    splits_df = _write_splits(content, tmp_path)
    # Simulate filtered set that excludes deuLUT
    translation_ids = ["engBBE"]
    excluded = translations_excluded_by_filter(splits_df, translation_ids)
    assert "deuLUT" in excluded
    for tid in sorted(excluded):
        print(f"WARNING: {tid} in splits.csv but excluded by filters", file=sys.stderr)
    captured = capsys.readouterr()
    assert "deuLUT" in captured.err


def test_split_alignment_preserved(tmp_path):
    splits_df = _write_splits(SPLITS_CSV, tmp_path)
    tids = ["engBBE", "fra1910"]
    text_table = build_text_table(MAIN_DF, tids)
    masks = compute_split_masks(text_table["vref"], splits_df, tids)
    for split_name, tid_masks in masks.items():
        split_text = apply_split(text_table, tid_masks)
        assert split_text.shape[0] == len(VREFS)


def test_output_format_parquet(tmp_path):
    tids = ["engBBE", "fra1910"]
    text_table = build_text_table(MAIN_DF, tids)
    out_path = str(tmp_path / "out.parquet")
    write_output(text_table, out_path, "parquet")
    recovered = pd.read_parquet(out_path)
    pd.testing.assert_frame_equal(
        text_table.reset_index(drop=True),
        recovered.reset_index(drop=True),
    )
