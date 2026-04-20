from __future__ import annotations

from dataclasses import dataclass


class TextChunkingError(Exception):
    pass


@dataclass(slots=True)
class TextChunk:
    chunk_id: int
    text: str
    start_char: int
    end_char: int


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    if chunk_size <= 0:
        raise TextChunkingError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise TextChunkingError("chunk_overlap must be greater than or equal to 0")

    if chunk_overlap >= chunk_size:
        raise TextChunkingError("chunk_overlap must be smaller than chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    chunk_id = 1
    step = chunk_size - chunk_overlap

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk_text_value = normalized[start:end].strip()

        if chunk_text_value:
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text_value,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_id += 1

        if end >= len(normalized):
            break

        start += step

    return chunks