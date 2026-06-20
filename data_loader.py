"""
PDF loading and chunking — fully in-memory, no disk writes.

Uses pypdf instead of llama-index's PDFReader because PDFReader requires
a file path and parses the entire document into memory before chunking,
which is what made 50MB+ PDFs risky on a 512MB Render free-tier instance.
pypdf lets us read page-by-page directly from bytes.
"""
from __future__ import annotations

import hashlib
import logging
import os
from io import BytesIO
from typing import Generator

import pypdf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("uvicorn")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
CHUNK_SIZE = 1000       # approx characters per chunk
CHUNK_OVERLAP = 200     # approx character overlap between consecutive chunks
EMBED_BATCH_SIZE = 32   # texts per embedding API call


class InvalidPDFError(Exception):
    """Raised when an uploaded file is not a valid, readable PDF."""


def compute_file_hash(file_bytes: bytes) -> str:
    """MD5 hash of the raw file bytes — used for duplicate detection."""
    return hashlib.md5(file_bytes).hexdigest()


def validate_pdf(file_bytes: bytes) -> int:
    """
    Validate that the bytes are a readable PDF.
    Returns the page count if valid, raises InvalidPDFError otherwise.
    """
    if not file_bytes:
        raise InvalidPDFError("The uploaded file is empty.")

    if file_bytes[:4] != b"%PDF":
        raise InvalidPDFError("This file doesn't look like a valid PDF.")

    try:
        reader = pypdf.PdfReader(BytesIO(file_bytes))
        page_count = len(reader.pages)
    except Exception as exc:
        logger.error(f"PDF validation failed: {exc}")
        raise InvalidPDFError(
            "This PDF appears to be corrupted or password-protected and couldn't be read."
        ) from exc

    if page_count == 0:
        raise InvalidPDFError("This PDF has no readable pages.")

    return page_count


def _split_into_chunks(text: str) -> list[str]:
    """Simple overlapping character-window chunker. No external splitter dependency."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def iter_pdf_chunks(file_bytes: bytes) -> Generator[tuple[int, int, str], None, None]:
    """
    Stream text chunks from an in-memory PDF, page by page.
    Yields (current_page, total_pages, chunk_text) so callers can report progress
    without having to know the final chunk count up front.
    """
    reader = pypdf.PdfReader(BytesIO(file_bytes))
    total_pages = len(reader.pages)

    for page_num in range(total_pages):
        try:
            text = reader.pages[page_num].extract_text() or ""
        except Exception as exc:
            logger.warning(f"Could not extract text from page {page_num + 1}: {exc}")
            text = ""

        for chunk in _split_into_chunks(text):
            yield (page_num + 1, total_pages, chunk)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in batches (avoids oversized single requests that
    were causing OpenRouter 400s). Filters out empty/whitespace entries.
    """
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=batch)
            all_embeddings.extend(item.embedding for item in response.data)
        except Exception as exc:
            logger.error(f"Embedding batch failed at offset {i}: {exc}")
            raise

    return all_embeddings
