"""Tests for list_tags, add_tags, and version comparison in huggingface.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))
from huggingface import (
    add_tags,
    check_version_tags,
    is_version_tag,
    list_tags,
    parse_version_tag,
)


# ── Version utilities ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("tag,expected", [
    ("v1.0", True), ("V2.0", True), ("v1.2.3", True),
    ("latest", False), ("main", False), ("1.0", False),
])
def test_is_version_tag(tag, expected):
    assert is_version_tag(tag) == expected


@pytest.mark.parametrize("tag,major,minor", [
    ("v1.0", 1, 0), ("V2.3", 2, 3), ("v1.10", 1, 10),
])
def test_parse_version_tag_valid(tag, major, minor):
    v = parse_version_tag(tag)
    assert v is not None
    assert v.major == major
    assert v.minor == minor


def test_parse_version_tag_invalid_returns_none():
    assert parse_version_tag("vnot-a-version") is None


def test_version_ordering():
    assert parse_version_tag("v1.9") < parse_version_tag("v1.10")
    assert parse_version_tag("v1.0") < parse_version_tag("v2.0")
    assert parse_version_tag("v2.0") == parse_version_tag("V2.0")


# ── check_version_tags ────────────────────────────────────────────────────────

def test_no_existing_tags_always_passes():
    assert check_version_tags(["v1.0"], [], input_fn=lambda _: "n") is True


def test_new_tag_older_warns_and_user_confirms():
    assert check_version_tags(["v1.0"], ["v2.0"], input_fn=lambda _: "y") is True


def test_new_tag_older_warns_and_user_declines():
    assert check_version_tags(["v1.0"], ["v2.0"], input_fn=lambda _: "n") is False


def test_new_tag_same_warns_and_user_declines():
    assert check_version_tags(["v2.0"], ["v2.0"], input_fn=lambda _: "n") is False


def test_new_tag_newer_no_warning():
    call_count = [0]
    def counter(_):
        call_count[0] += 1
        return "y"
    result = check_version_tags(["v3.0"], ["v2.0"], input_fn=counter)
    assert result is True
    assert call_count[0] == 0


def test_non_version_tags_not_compared():
    # "latest" has no leading v, should not trigger warning even if existing versions exist
    assert check_version_tags(["latest"], ["v2.0"], input_fn=lambda _: "n") is True


def test_unparseable_new_tag_skipped_with_warning(capsys):
    result = check_version_tags(["vnot-valid"], ["v1.0"], input_fn=lambda _: "n")
    assert result is True
    err = capsys.readouterr().err
    assert "could not parse" in err


# ── list_tags ─────────────────────────────────────────────────────────────────

def _mock_api_with_tags(*names):
    api = MagicMock()
    tags = []
    for name in names:
        t = MagicMock()
        t.name = name
        tags.append(t)
    api.list_repo_refs.return_value.tags = tags
    return api


def test_list_tags_sorted_alphabetically(capsys):
    api = _mock_api_with_tags("v2.0", "latest", "v1.0")
    with patch("huggingface.HfApi", return_value=api):
        result = list_tags("owner/repo", token="tok")
    assert result == ["latest", "v1.0", "v2.0"]
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines == ["latest", "v1.0", "v2.0"]


def test_list_tags_empty_prints_message(capsys):
    api = _mock_api_with_tags()
    with patch("huggingface.HfApi", return_value=api):
        result = list_tags("owner/repo", token="tok")
    assert result == []
    out = capsys.readouterr().out
    assert "No tags" in out


# ── add_tags ──────────────────────────────────────────────────────────────────

def test_add_tags_calls_create_tag_for_each():
    api = _mock_api_with_tags()
    with patch("huggingface.HfApi", return_value=api):
        result = add_tags("owner/repo", ["v2.0", "latest"], token="tok")
    assert result is True
    assert api.create_tag.call_count == 2


def test_add_tags_version_warning_respected():
    api = _mock_api_with_tags("v3.0")
    with patch("huggingface.HfApi", return_value=api):
        result = add_tags("owner/repo", ["v1.0"], token="tok", input_fn=lambda _: "n")
    assert result is False
    api.create_tag.assert_not_called()


def test_add_tags_applies_all_when_confirmed():
    api = _mock_api_with_tags("v1.0")
    with patch("huggingface.HfApi", return_value=api):
        result = add_tags("owner/repo", ["v2.0", "latest"], token="tok", input_fn=lambda _: "y")
    assert result is True
    assert api.create_tag.call_count == 2
