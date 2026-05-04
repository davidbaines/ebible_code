"""Tests for the CLI argument parser in huggingface.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))
from huggingface import _build_parser


def parse(args):
    return _build_parser().parse_args(args)


# ── upload ────────────────────────────────────────────────────────────────────

def test_upload_positional_args():
    args = parse(["upload", "owner/repo", "/some/folder"])
    assert args.command == "upload"
    assert args.repo_id == "owner/repo"
    assert args.folder == "/some/folder"
    assert args.files is None
    assert args.tags is None
    assert args.token is None


def test_upload_with_tags():
    args = parse(["upload", "owner/repo", "/folder", "--tags", "v2.0", "latest"])
    assert args.tags == ["v2.0", "latest"]


def test_upload_with_files():
    args = parse(["upload", "owner/repo", "/folder", "--files", "a.parquet", "b.parquet"])
    assert args.files == ["a.parquet", "b.parquet"]


def test_upload_with_token():
    args = parse(["upload", "owner/repo", "/folder", "--token", "mytoken"])
    assert args.token == "mytoken"


def test_upload_missing_repo_id_errors():
    with pytest.raises(SystemExit):
        parse(["upload"])


# ── download ──────────────────────────────────────────────────────────────────

def test_download_positional_args():
    args = parse(["download", "owner/repo", "/dest"])
    assert args.command == "download"
    assert args.repo_id == "owner/repo"
    assert args.folder == "/dest"
    assert args.tag is None


def test_download_with_tag():
    args = parse(["download", "owner/repo", "/dest", "--tag", "v1.0"])
    assert args.tag == "v1.0"


# ── tag ───────────────────────────────────────────────────────────────────────

def test_tag_list_tags():
    args = parse(["tag", "owner/repo", "--list-tags"])
    assert args.command == "tag"
    assert args.list_tags is True
    assert args.add_tags is None


def test_tag_add_tags():
    args = parse(["tag", "owner/repo", "--add-tags", "v2.0", "latest"])
    assert args.add_tags == ["v2.0", "latest"]
    assert args.list_tags is False


def test_tag_add_tags_and_list_tags_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse(["tag", "owner/repo", "--add-tags", "v2.0", "--list-tags"])


def test_tag_requires_one_of_add_or_list():
    with pytest.raises(SystemExit):
        parse(["tag", "owner/repo"])


# ── general ───────────────────────────────────────────────────────────────────

def test_unknown_subcommand_exits():
    with pytest.raises(SystemExit):
        parse(["unknowncmd"])


def test_upload_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        parse(["upload", "--help"])
    assert exc.value.code == 0


def test_download_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        parse(["download", "--help"])
    assert exc.value.code == 0


def test_tag_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        parse(["tag", "--help"])
    assert exc.value.code == 0
