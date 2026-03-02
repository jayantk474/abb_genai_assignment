from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re


@dataclass
class PageText:
    text: str
    page: int  # 1-indexed


def _clean(text: str) -> str:
    # Normalize whitespace, keep paragraph breaks.
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str) -> List[PageText]:
    """Extract per-page text.

    For SEC filings, tables are the main pain point. pdfplumber with layout extraction
    and tight tolerances usually preserves row order better than default settings.
    """

    import pdfplumber

    pages: List[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = (
                page.extract_text(
                    x_tolerance=1,
                    y_tolerance=1,
                    layout=True,
                )
                or ""
            )
            pages.append(PageText(text=_clean(raw), page=i))
    return pages


SECTION_RE = re.compile(
    r"\b(Item\s+\d+[A-Z]?)\b|\b(Signatures?)\b|\b(Notes?\s+to\s+Consolidated\s+Financial\s+Statements)\b",
    re.IGNORECASE,
)

NOTE_RE = re.compile(r"\bNote\s+(\d+)\b", re.IGNORECASE)


def guess_section(text: str) -> str:
    # Best-effort section extraction from chunk text.
    m = SECTION_RE.search(text)
    if m:
        for g in m.groups():
            if g:
                g = g.strip()
                if re.match(r"signatures?", g, re.IGNORECASE):
                    return "Signature page"
                return g
    m2 = NOTE_RE.search(text)
    if m2:
        return f"Note {m2.group(1)}"
    return "Unknown section"


def chunk_pages(pages: List[PageText], doc_name: str, chunk_chars: int, overlap_chars: int):
    """Chunk pages into overlapping chunks.

    Important: Increase chunk size for financial sections (Item 7 / Item 8 / Notes)
    so table rows (label + number) stay together.
    """

    FIN_KWS = (
        "item 7",
        "item 8",
        "notes to consolidated financial statements",
        "consolidated statements",
        "note ",
        "net sales",
        "total net sales",
        "total revenue",
        "term debt",
        "consolidated statements of operations",
        "consolidated statements of income",
    )

    def is_financial(text: str) -> bool:
        t = (text or "").lower()
        return any(k in t for k in FIN_KWS)

    DEFAULT_CHUNK = int(chunk_chars)
    DEFAULT_OVERLAP = int(overlap_chars)
    FIN_CHUNK = max(DEFAULT_CHUNK, 6000)
    FIN_OVERLAP = max(DEFAULT_OVERLAP, 600)

    buffer = ""
    start_page = pages[0].page if pages else 1
    current_page = start_page

    current_chunk = DEFAULT_CHUNK
    current_overlap = DEFAULT_OVERLAP

    for p in pages:
        if not p.text:
            continue

        # Choose chunking regime per page
        want_fin = is_financial(p.text)
        next_chunk = FIN_CHUNK if want_fin else DEFAULT_CHUNK
        next_overlap = FIN_OVERLAP if want_fin else DEFAULT_OVERLAP

        # If regime changes, flush to avoid mixing narrative + table text into one chunk
        if buffer and next_chunk != current_chunk:
            section = guess_section(buffer)
            yield {
                "text": buffer.strip(),
                "metadata": {
                    "document": doc_name,
                    "page_start": start_page,
                    "page_end": current_page,
                    "section": section,
                },
            }
            buffer = ""

        current_chunk = next_chunk
        current_overlap = next_overlap

        if not buffer:
            start_page = p.page

        buffer += "\n\n" + p.text
        current_page = p.page

        while len(buffer) >= current_chunk:
            chunk_text = buffer[:current_chunk]
            section = guess_section(chunk_text)

            yield {
                "text": chunk_text.strip(),
                "metadata": {
                    "document": doc_name,
                    "page_start": start_page,
                    "page_end": current_page,
                    "section": section,
                },
            }

            buffer = buffer[current_chunk - current_overlap :]
            start_page = current_page

    if buffer.strip():
        section = guess_section(buffer)
        yield {
            "text": buffer.strip(),
            "metadata": {
                "document": doc_name,
                "page_start": start_page,
                "page_end": current_page,
                "section": section,
            },
        }
