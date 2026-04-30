"""Tests for clean_range_markers in ebible.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))
from ebible import clean_range_markers


def write_corpus(tmp_path, lines):
    """Write lines to a temp corpus file and return its path."""
    p = tmp_path / "test.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def read_corpus(path):
    return [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]


def test_valid_range_preserved(tmp_path):
    lines = ["In the beginning", "<range>", "<range>"]
    p = write_corpus(tmp_path, lines)
    count = clean_range_markers(p)
    assert count == 0
    assert read_corpus(p) == lines


def test_orphaned_range_replaced(tmp_path):
    lines = ["", "<range>"]
    p = write_corpus(tmp_path, lines)
    count = clean_range_markers(p)
    assert count == 1
    assert read_corpus(p) == ["", ""]


def test_cascade(tmp_path):
    """Two consecutive <range> lines after an empty line are both replaced."""
    lines = ["", "<range>", "<range>"]
    p = write_corpus(tmp_path, lines)
    count = clean_range_markers(p)
    assert count == 2
    assert read_corpus(p) == ["", "", ""]


def test_first_line_range(tmp_path):
    """A <range> on line 0 has no predecessor and is treated as orphaned."""
    lines = ["<range>", "Some text"]
    p = write_corpus(tmp_path, lines)
    count = clean_range_markers(p)
    assert count == 1
    assert read_corpus(p) == ["", "Some text"]


def test_empty_file(tmp_path):
    lines = ["", "", ""]
    p = write_corpus(tmp_path, lines)
    count = clean_range_markers(p)
    assert count == 0
    assert read_corpus(p) == lines


def test_no_ranges(tmp_path):
    lines = ["Genesis 1:1 text", "Genesis 1:2 text", ""]
    p = write_corpus(tmp_path, lines)
    count = clean_range_markers(p)
    assert count == 0
    assert read_corpus(p) == lines


def test_mixed(tmp_path):
    """Valid and orphaned ranges in the same file — only orphaned are replaced."""
    lines = [
        "Text at verse 1",
        "<range>",   # valid — preceded by text
        "",
        "<range>",   # orphaned — preceded by ""
        "Text at verse 5",
    ]
    p = write_corpus(tmp_path, lines)
    count = clean_range_markers(p)
    assert count == 1
    assert read_corpus(p) == [
        "Text at verse 1",
        "<range>",
        "",
        "",
        "Text at verse 5",
    ]


def test_idempotent(tmp_path):
    lines = ["", "<range>", "<range>", "Text", "<range>"]
    p = write_corpus(tmp_path, lines)
    clean_range_markers(p)
    result_after_first = read_corpus(p)
    clean_range_markers(p)
    result_after_second = read_corpus(p)
    assert result_after_first == result_after_second


def test_return_count(tmp_path):
    lines = ["", "<range>", "<range>", "Text", "<range>", ""]
    p = write_corpus(tmp_path, lines)
    count = clean_range_markers(p)
    # lines[1] orphaned (prev=""), lines[2] orphaned (prev becomes ""), lines[4] valid (prev="Text")
    assert count == 2
