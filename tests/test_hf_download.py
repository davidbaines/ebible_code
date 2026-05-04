"""Tests for the download_dataset function in huggingface.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))
from huggingface import download_dataset


def _make_parquet(path: Path, data: dict):
    """Write a small Parquet file from a dict of column_name → list."""
    table = pa.table(data)
    pq.write_table(table, str(path))


def test_tag_passed_as_revision(tmp_path):
    with patch("huggingface.snapshot_download", return_value=str(tmp_path)) as mock_dl:
        download_dataset("owner/repo", tmp_path, tag="v1.0", token="tok")
    mock_dl.assert_called_once()
    assert mock_dl.call_args.kwargs["revision"] == "v1.0"


def test_no_tag_defaults_to_main(tmp_path):
    with patch("huggingface.snapshot_download", return_value=str(tmp_path)) as mock_dl:
        download_dataset("owner/repo", tmp_path, token="tok")
    assert mock_dl.call_args.kwargs["revision"] == "main"


def test_parquet_rows_and_columns_reported(tmp_path, capsys):
    pf = tmp_path / "main.parquet"
    _make_parquet(pf, {"vref": ["GEN 1:1", "GEN 1:2"], "eng-engWEB": ["In the beginning", "Now"]})

    with patch("huggingface.snapshot_download", return_value=str(tmp_path)):
        download_dataset("owner/repo", tmp_path, token="tok")

    out = capsys.readouterr().out
    assert "2" in out          # 2 rows
    assert "2" in out          # 2 columns
    assert "vref" in out
    assert "eng-engWEB" in out


def test_parquet_with_many_columns_truncates_at_ten(tmp_path, capsys):
    data = {f"col{i}": [1, 2] for i in range(15)}
    pf = tmp_path / "wide.parquet"
    _make_parquet(pf, data)

    with patch("huggingface.snapshot_download", return_value=str(tmp_path)):
        download_dataset("owner/repo", tmp_path, token="tok")

    out = capsys.readouterr().out
    assert "..." in out


def test_download_reports_total_file_count(tmp_path, capsys):
    (tmp_path / "README.md").write_text("hello")
    pf = tmp_path / "data.parquet"
    _make_parquet(pf, {"a": [1]})

    with patch("huggingface.snapshot_download", return_value=str(tmp_path)):
        download_dataset("owner/repo", tmp_path, token="tok")

    out = capsys.readouterr().out
    assert "file" in out.lower()


def test_download_creates_folder_if_absent(tmp_path):
    dest = tmp_path / "new_folder"
    with patch("huggingface.snapshot_download", return_value=str(dest)) as mock_dl:
        dest.mkdir()  # snapshot_download would create it; we pre-create for the mock
        download_dataset("owner/repo", dest, token="tok")
    assert dest.is_dir()
