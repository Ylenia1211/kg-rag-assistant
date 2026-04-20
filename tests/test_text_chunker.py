from __future__ import annotations

import pytest

from app.services.text_chunker import TextChunkingError, chunk_text, normalize_text


def test_normalize_text_collapses_whitespace():
    assert normalize_text(" hello\n\nworld\t test ") == "hello world test"


def test_chunk_text_returns_empty_list_for_blank_input():
    assert chunk_text("   \n\t  ", chunk_size=10, chunk_overlap=2) == []


def test_chunk_text_returns_single_chunk_for_short_text():
    chunks = chunk_text("hello world", chunk_size=50, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len("hello world")


def test_chunk_text_splits_long_text_with_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 3
    assert chunks[0].text == "abcdefghij"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 10

    assert chunks[1].text == "ijklmnopqr"
    assert chunks[1].start_char == 8
    assert chunks[1].end_char == 18

    assert chunks[2].text == "qrstuvwxyz"
    assert chunks[2].start_char == 16
    assert chunks[2].end_char == 26


def test_chunk_text_raises_for_invalid_chunk_size():
    with pytest.raises(TextChunkingError, match="chunk_size must be greater than 0"):
        chunk_text("hello", chunk_size=0, chunk_overlap=0)


def test_chunk_text_raises_for_negative_overlap():
    with pytest.raises(TextChunkingError, match="chunk_overlap must be greater than or equal to 0"):
        chunk_text("hello", chunk_size=10, chunk_overlap=-1)


def test_chunk_text_raises_when_overlap_is_not_smaller_than_chunk_size():
    with pytest.raises(TextChunkingError, match="chunk_overlap must be smaller than chunk_size"):
        chunk_text("hello", chunk_size=10, chunk_overlap=10)