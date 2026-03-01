from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple, Optional
from pypdf import PdfReader
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
    reader = PdfReader(pdf_path)
    pages: List[PageText] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        pages.append(PageText(text=_clean(raw), page=i+1))
    return pages

SECTION_RE = re.compile(
    r"\b(Item\s+\d+[A-Z]?)\b|\b(Signatures?)\b|\b(Notes?\s+to\s+Consolidated\s+Financial\s+Statements)\b",
    re.IGNORECASE
)

NOTE_RE = re.compile(r"\bNote\s+(\d+)\b", re.IGNORECASE)

def guess_section(text: str) -> str:
    # Best-effort section extraction from chunk text.
    m = SECTION_RE.search(text)
    if m:
        for g in m.groups():
            if g:
                # Canonicalize capitalization a bit
                return g.strip().replace("SIGNATURES", "Signatures").replace("Signatures", "Signature page")
    m2 = NOTE_RE.search(text)
    if m2:
        return f"Note {m2.group(1)}"
    return "Unknown section"

def chunk_pages(pages: List[PageText], doc_name: str, chunk_chars: int, overlap_chars: int):
    """
    Page-level chunking.
    Each page becomes its own chunk.
    This avoids mixing tables with unrelated numeric paragraphs.
    """
    for p in pages:
        if not p.text.strip():
            continue

        section = guess_section(p.text)

        yield {
            "text": p.text.strip(),
            "metadata": {
                "document": doc_name,
                "page_start": p.page,
                "page_end": p.page,
                "section": section,
            },
        }
