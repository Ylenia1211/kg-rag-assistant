from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.services.document_parser import (
    DocumentParsingError,
    detect_extension,
    parse_document,
    parse_pdf_file,
    parse_text_file,
)


def build_empty_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_detect_extension_lowercases_suffix():
    assert detect_extension("Report.PDF") == ".pdf"
    assert detect_extension("notes.md") == ".md"


def test_parse_text_file_success():
    assert parse_text_file(b"hello\nworld\n") == "hello\nworld"


def test_parse_text_file_raises_on_invalid_utf8():
    with pytest.raises(DocumentParsingError, match="Unable to decode text file as UTF-8"):
        parse_text_file(b"\xff\xfe\x00\x00")


def test_parse_document_supports_txt_and_md():
    assert parse_document("note.txt", b"abc") == "abc"
    assert parse_document("readme.md", b"# title") == "# title"


def test_parse_document_rejects_unsupported_extension():
    with pytest.raises(DocumentParsingError, match="Unsupported file extension: .docx"):
        parse_document("file.docx", b"data")


def test_parse_pdf_file_empty_page_returns_empty_string():
    pdf_bytes = build_empty_pdf_bytes()
    assert parse_pdf_file(pdf_bytes) == ""


def test_parse_document_supports_pdf():
    pdf_bytes = build_empty_pdf_bytes()
    assert parse_document("sample.pdf", pdf_bytes) == ""


def test_parse_pdf_file_raises_on_invalid_content():
    with pytest.raises(DocumentParsingError, match="Unable to parse PDF file"):
        parse_pdf_file(b"not-a-real-pdf")