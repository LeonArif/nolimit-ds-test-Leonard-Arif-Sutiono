from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import fitz


@dataclass(frozen=True)
class PDFChunk:
    file_name: str
    page_number: int
    chunk_index: int
    text: str


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_words(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_pdf_bytes(pdf_bytes: bytes, file_name: str, *, chunk_size: int = 220, overlap: int = 40) -> list[PDFChunk]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks: list[PDFChunk] = []

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        extracted_text = _normalize_text(page.get_text("text"))
        if not extracted_text:
            continue

        page_chunks = _split_words(extracted_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunks.append(
                PDFChunk(
                    file_name=file_name,
                    page_number=page_index + 1,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

    return chunks


def load_pdf_paths(paths: Sequence[Path], *, chunk_size: int = 220, overlap: int = 40) -> list[PDFChunk]:
    chunks: list[PDFChunk] = []
    for path in paths:
        chunks.extend(
            load_pdf_bytes(
                path.read_bytes(),
                path.name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return chunks