from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class DocumentParsingError(Exception):
    pass


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def detect_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def parse_text_file(content: bytes, encoding: str = "utf-8") -> str:
    try:
        return content.decode(encoding).strip()
    except UnicodeDecodeError as exc:
        raise DocumentParsingError("Unable to decode text file as UTF-8") from exc


def parse_pdf_file(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
        pages: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                pages.append(extracted.strip())
        return "\n\n".join(pages).strip()
    except Exception as exc:
        raise DocumentParsingError("Unable to parse PDF file") from exc


def parse_document(filename: str, content: bytes) -> str:
    extension = detect_extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParsingError(f"Unsupported file extension: {extension or 'unknown'}")

    if extension in {".txt", ".md"}:
        return parse_text_file(content)

    if extension == ".pdf":
        return parse_pdf_file(content)

    raise DocumentParsingError(f"No parser available for extension: {extension}")