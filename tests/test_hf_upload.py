"""Tests for the upload_dataset function in huggingface.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))
from huggingface import upload_dataset

try:
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    from huggingface_hub.utils import RepositoryNotFoundError


def _make_api(repo_exists=True):
    api = MagicMock()
    if not repo_exists:
        api.repo_info.side_effect = RepositoryNotFoundError("not found")
    refs = MagicMock()
    refs.tags = []
    api.list_repo_refs.return_value = refs
    return api


# ── Repo existence ────────────────────────────────────────────────────────────

def test_missing_repo_user_confirms_creates(tmp_path):
    api = _make_api(repo_exists=False)
    (tmp_path / "data.parquet").write_bytes(b"x")

    inputs = iter(["y", "y"])  # create repo, then confirm files
    with patch("huggingface.HfApi", return_value=api):
        result = upload_dataset("owner/repo", tmp_path, token="tok", input_fn=lambda _: next(inputs))

    api.create_repo.assert_called_once_with(repo_id="owner/repo", repo_type="dataset", exist_ok=True)
    assert result is True


def test_missing_repo_user_declines_clean_exit(tmp_path):
    api = _make_api(repo_exists=False)

    with patch("huggingface.HfApi", return_value=api):
        result = upload_dataset("owner/repo", tmp_path, token="tok", input_fn=lambda _: "n")

    api.create_repo.assert_not_called()
    assert result is False


# ── File discovery ────────────────────────────────────────────────────────────

def test_auto_discovery_only_uploads_correct_extensions(tmp_path):
    (tmp_path / "data.parquet").write_bytes(b"x")
    (tmp_path / "meta.csv").write_bytes(b"x")
    (tmp_path / "README.md").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_bytes(b"x")
    (tmp_path / "ignore.json").write_bytes(b"x")

    api = _make_api()

    with patch("huggingface.HfApi", return_value=api):
        upload_dataset("owner/repo", tmp_path, token="tok", input_fn=lambda _: "y")

    ops = api.create_commit.call_args.kwargs["operations"]
    uploaded_names = {op.path_in_repo for op in ops}
    assert uploaded_names == {"data.parquet", "meta.csv", "README.md"}


def test_files_flag_uploads_only_specified_files(tmp_path):
    (tmp_path / "main.parquet").write_bytes(b"x")
    (tmp_path / "meta.parquet").write_bytes(b"x")

    api = _make_api()

    with patch("huggingface.HfApi", return_value=api):
        upload_dataset(
            "owner/repo", tmp_path, files=["main.parquet"], token="tok", input_fn=lambda _: "y"
        )

    ops = api.create_commit.call_args.kwargs["operations"]
    assert [op.path_in_repo for op in ops] == ["main.parquet"]


def test_files_flag_missing_file_exits(tmp_path):
    api = _make_api()
    with patch("huggingface.HfApi", return_value=api):
        with pytest.raises(SystemExit) as exc:
            upload_dataset(
                "owner/repo", tmp_path, files=["missing.parquet"], token="tok",
                input_fn=lambda _: "y"
            )
    assert exc.value.code == 1


# ── File confirmation ─────────────────────────────────────────────────────────

def test_user_declines_file_confirmation_clean_exit(tmp_path):
    (tmp_path / "data.parquet").write_bytes(b"x")
    api = _make_api()

    with patch("huggingface.HfApi", return_value=api):
        result = upload_dataset("owner/repo", tmp_path, token="tok", input_fn=lambda _: "n")

    api.create_commit.assert_not_called()
    assert result is False


# ── Version tag warnings ──────────────────────────────────────────────────────

def _api_with_existing_tags(*tag_names):
    api = _make_api()
    tags = [MagicMock(name=t) for t in tag_names]
    for tag, name in zip(tags, tag_names):
        tag.name = name
    api.list_repo_refs.return_value.tags = tags
    return api


def test_version_tag_warning_fires_when_new_is_older(tmp_path, capsys):
    (tmp_path / "data.parquet").write_bytes(b"x")
    api = _api_with_existing_tags("v2.0")

    responses = iter(["y", "n"])  # confirm files, then decline tag warning
    with patch("huggingface.HfApi", return_value=api):
        result = upload_dataset(
            "owner/repo", tmp_path, tags=["v1.0"], token="tok",
            input_fn=lambda _: next(responses)
        )

    assert result is False


def test_version_tag_no_warning_when_new_is_newer(tmp_path):
    (tmp_path / "data.parquet").write_bytes(b"x")
    api = _api_with_existing_tags("v1.0")

    call_count = [0]
    def inputs(prompt):
        call_count[0] += 1
        return "y"

    with patch("huggingface.HfApi", return_value=api):
        result = upload_dataset(
            "owner/repo", tmp_path, tags=["v2.0"], token="tok", input_fn=inputs
        )

    # Only one prompt (file confirmation), no version warning
    assert call_count[0] == 1
    assert result is True


def test_user_declines_tag_warning_aborts(tmp_path):
    (tmp_path / "data.parquet").write_bytes(b"x")
    api = _api_with_existing_tags("v3.0")

    responses = iter(["y", "n"])
    with patch("huggingface.HfApi", return_value=api):
        result = upload_dataset(
            "owner/repo", tmp_path, tags=["v2.0"], token="tok",
            input_fn=lambda _: next(responses)
        )

    api.create_tag.assert_not_called()
    assert result is False


# ── Upload and tagging ────────────────────────────────────────────────────────

def test_all_files_uploaded_in_one_commit(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"x")
    (tmp_path / "b.parquet").write_bytes(b"x")
    api = _make_api()

    with patch("huggingface.HfApi", return_value=api):
        upload_dataset("owner/repo", tmp_path, token="tok", input_fn=lambda _: "y")

    assert api.create_commit.call_count == 1
    ops = api.create_commit.call_args.kwargs["operations"]
    assert len(ops) == 2


def test_tags_applied_after_upload(tmp_path):
    (tmp_path / "data.parquet").write_bytes(b"x")
    api = _make_api()

    with patch("huggingface.HfApi", return_value=api):
        upload_dataset(
            "owner/repo", tmp_path, tags=["v2.0", "latest"], token="tok",
            input_fn=lambda _: "y"
        )

    assert api.create_tag.call_count == 2
    tag_calls = [c.kwargs["tag"] for c in api.create_tag.call_args_list]
    assert "v2.0" in tag_calls
    assert "latest" in tag_calls
