"""Tests for authentication token resolution in huggingface.py."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ebible_code"))
from huggingface import _AUTH_HELP, require_token, resolve_token


def test_cli_token_takes_priority(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=env_token\n")
    with patch("huggingface.HfFolder.get_token", return_value="cached_token"):
        assert resolve_token("cli_token") == "cli_token"


def test_env_token_used_when_no_cli_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=env_token\n")
    with patch("huggingface.HfFolder.get_token", return_value=None):
        # Must reload dotenv from the tmp_path .env
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}, clear=False):
            result = resolve_token(None)
    assert result == "env_token"


def test_cached_token_used_when_no_cli_or_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch("huggingface.HfFolder.get_token", return_value="cached_token"):
        with patch("huggingface.load_dotenv"):
            result = resolve_token(None)
    assert result == "cached_token"


def test_resolve_token_returns_none_when_all_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch("huggingface.HfFolder.get_token", return_value=None):
        with patch("huggingface.load_dotenv"):
            result = resolve_token(None)
    assert result is None


def test_require_token_exits_when_all_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch("huggingface.HfFolder.get_token", return_value=None):
        with patch("huggingface.load_dotenv"):
            with pytest.raises(SystemExit) as exc_info:
                require_token(None)
    assert exc_info.value.code == 1


def test_require_token_error_message_mentions_all_three_options(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch("huggingface.HfFolder.get_token", return_value=None):
        with patch("huggingface.load_dotenv"):
            with pytest.raises(SystemExit):
                require_token(None)
    err = capsys.readouterr().err
    assert "--token" in err
    assert "HF_TOKEN" in err
    assert "huggingface-cli login" in err


def test_require_token_returns_token_when_available():
    with patch("huggingface.resolve_token", return_value="good_token"):
        result = require_token("good_token")
    assert result == "good_token"
