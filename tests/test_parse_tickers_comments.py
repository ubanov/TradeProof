"""``cli.headless.parse_tickers``: ``#``-prefixed comment lines and blank
lines in a ``--tickers-file`` are ignored (needed by the EvidenceArchive
universe files, e.g. ``config/evidence-universe.txt``, which document their
own tickers inline). Comma-separated ``--tickers`` values are unaffected --
only file lines get comment-stripping."""

from __future__ import annotations

import pytest

from cli.headless import parse_tickers


@pytest.mark.unit
def test_comment_lines_are_ignored(tmp_path):
    f = tmp_path / "universe.txt"
    f.write_text("# this is a comment\nNVDA\n# another comment\nMSFT\n", encoding="utf-8")
    assert parse_tickers(None, f) == ["NVDA", "MSFT"]


@pytest.mark.unit
def test_blank_lines_are_ignored(tmp_path):
    f = tmp_path / "universe.txt"
    f.write_text("NVDA\n\n\nMSFT\n", encoding="utf-8")
    assert parse_tickers(None, f) == ["NVDA", "MSFT"]


@pytest.mark.unit
def test_whitespace_is_trimmed(tmp_path):
    f = tmp_path / "universe.txt"
    f.write_text("  NVDA  \n\tMSFT\t\n", encoding="utf-8")
    assert parse_tickers(None, f) == ["NVDA", "MSFT"]


@pytest.mark.unit
def test_duplicates_across_file_lines_are_deduped(tmp_path):
    f = tmp_path / "universe.txt"
    f.write_text("NVDA\nMSFT\nNVDA\n", encoding="utf-8")
    assert parse_tickers(None, f) == ["NVDA", "MSFT"]


@pytest.mark.unit
def test_comment_indented_with_leading_whitespace_is_also_ignored(tmp_path):
    f = tmp_path / "universe.txt"
    f.write_text("NVDA\n   # indented comment\nMSFT\n", encoding="utf-8")
    assert parse_tickers(None, f) == ["NVDA", "MSFT"]


@pytest.mark.unit
def test_comma_separated_value_path_is_unaffected_by_comment_stripping():
    # Comment-stripping applies only to tickers_file lines, never to the
    # --tickers comma-separated string.
    assert parse_tickers("NVDA,MSFT") == ["NVDA", "MSFT"]


@pytest.mark.unit
def test_file_and_value_combine_with_comments_skipped(tmp_path):
    f = tmp_path / "universe.txt"
    f.write_text("# header\nLLY\n", encoding="utf-8")
    assert parse_tickers("NVDA", f) == ["NVDA", "LLY"]


@pytest.mark.unit
def test_a_file_with_only_comments_and_blanks_raises_value_error(tmp_path):
    f = tmp_path / "universe.txt"
    f.write_text("# nothing here\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_tickers(None, f)
