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
    # Use pdfplumber for better table/text extraction on 10-K style PDFs
    import pdfplumber

    pages: List[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            pages.append(PageText(text=_clean(raw), page=i + 1))
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
    buffer = ""
    start_page = pages[0].page if pages else 1
    current_page = start_page

    for p in pages:
        if not p.text:
            continue

        if not buffer:
            start_page = p.page

        buffer += "\n\n" + p.text
        current_page = p.page

        while len(buffer) >= chunk_chars:
            chunk_text = buffer[:chunk_chars]
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

            buffer = buffer[chunk_chars - overlap_chars:]
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
