"""Report documents: text extraction, NTSB-aware sectioning, chunking and hybrid search.

A report becomes a ``ReportDoc``:
    pages    -> text per PDF page (1-based page numbers kept for citations)
    sections -> recognised NTSB headings (Probable Cause, History of Flight, ...)
    chunks   -> ~120-word overlapping passages, each tagged with page + section
    index    -> BM25 keyword index, optionally fused with dense embeddings (fastembed)
"""
from __future__ import annotations

import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

# Headings used in NTSB aviation final / factual / preliminary reports (and docket factual reports).
NTSB_SECTIONS = [
    "Probable Cause and Findings", "Probable Cause", "Findings", "Analysis", "Executive Summary",
    "Factual Information", "History of Flight", "Pilot Information", "Flight Crew Information",
    "Aircraft and Owner/Operator Information", "Aircraft Information",
    "Meteorological Information and Flight Plan", "Meteorological Information",
    "Airport Information", "Wreckage and Impact Information", "Flight Recorders",
    "Medical and Pathological Information", "Survival Aspects", "Tests and Research",
    "Organizational and Management Information", "Additional Information",
    "Safety Recommendations", "Recommendations", "Conclusions", "Administrative Information",
    "Preliminary Information", "Injuries to Persons", "Damage to Aircraft", "Personnel Information",
]
_HEAD_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(" + "|".join(re.escape(h) for h in
                      sorted(NTSB_SECTIONS, key=len, reverse=True)) + r")\s*:?\s*$", re.I)

STOP = set("""a an and are as at be been but by for from had has have he her his i in into is it its
of on or our she so than that the their them then there these they this to was were which while who
will with would not no do did does about after before during over under up down out off can could
should may might also any all each other such only own same very just""".split())

TOKEN = re.compile(r"[a-z0-9][a-z0-9\-/\.]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass
class Chunk:
    id: int
    text: str
    page: int
    section: str


@dataclass
class ReportDoc:
    doc_id: str
    title: str
    pages: list[str]
    source: str = "report"
    meta: dict[str, Any] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    _bm25: Optional["BM25"] = None
    _emb: Any = None

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    def section_text(self, *names: str) -> str:
        want = {n.lower() for n in names}
        return "\n".join(s["text"] for s in self.sections if s["name"].lower() in want).strip()

    def outline(self) -> list[dict[str, Any]]:
        return [{"name": s["name"], "page": s["page"], "words": len(s["text"].split())} for s in self.sections]


# ---------------------------------------------------------------------------- building
def pdf_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:  # damaged page: keep position for citations
            t = ""
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"(?im)^\s*page\s+\d+\s+of\s+\d+\s*$", "", t)  # running footers
        t = re.sub(r"(?i)\bpage\s+\d+\s+of\s+\d+\s*", "", t)
        pages.append(t.strip())
    return pages


def build_doc(doc_id: str, title: str, pages: list[str], *, source: str = "report",
              meta: Optional[dict[str, Any]] = None, embedder: Any = None) -> ReportDoc:
    doc = ReportDoc(doc_id=doc_id, title=title, pages=pages, source=source, meta=meta or {})
    doc.sections = _split_sections(pages)
    doc.chunks = _chunk(doc.sections)
    doc._bm25 = BM25([c.text for c in doc.chunks])
    if embedder is not None and doc.chunks:
        try:
            doc._emb = embedder.embed([c.text for c in doc.chunks])
        except Exception:
            doc._emb = None
    return doc


def doc_from_case(case: dict[str, Any]) -> ReportDoc:
    """Build a document from structured case data when no PDF is available."""
    parts = [("Probable Cause", case.get("probable_cause")),
             ("Findings", "\n".join(f"{f['text']} - {f.get('cause_factor') or ''}" for f in case.get("findings") or [])),
             ("Analysis", case.get("narrative_analysis")),
             ("History of Flight", case.get("narrative_factual")),
             ("Preliminary Information", case.get("narrative_prelim"))]
    page = "\n".join(f"{h}\n{t}" for h, t in parts if t)
    return build_doc(case.get("ntsb_number") or "case", f"Case record {case.get('ntsb_number')}",
                     [page], source="case-record", meta={"ntsb_number": case.get("ntsb_number")})


def _split_sections(pages: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    cur = {"name": "Front matter", "page": 1, "lines": []}
    for pno, page in enumerate(pages, 1):
        for line in page.splitlines():
            m = _HEAD_RE.match(line)
            if m and len(line) < 70:
                if cur["lines"] or cur["name"] != "Front matter":
                    sections.append(cur)
                canon = next(h for h in NTSB_SECTIONS if h.lower() == m.group(1).lower())
                cur = {"name": canon, "page": pno, "lines": []}
            else:
                cur["lines"].append((pno, line))
    sections.append(cur)
    out = []
    for s in sections:
        text = "\n".join(l for _, l in s["lines"]).strip()
        if text or s["name"] != "Front matter":
            out.append({"name": s["name"], "page": s["page"], "text": text, "lines": s["lines"]})
    return out


def _chunk(sections: list[dict[str, Any]], size: int = 120, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s in sections:
        words: list[tuple[str, int]] = [(w, p) for p, line in s["lines"] for w in line.split()]
        if not words:
            continue
        step = max(1, size - overlap)
        for start in range(0, len(words), step):
            seg = words[start:start + size]
            chunks.append(Chunk(id=len(chunks), text=" ".join(w for w, _ in seg), page=seg[0][1], section=s["name"]))
            if start + size >= len(words):
                break
    return chunks


# ---------------------------------------------------------------------------- search
class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.len = [len(t) for t in self.toks]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out


def search(doc: ReportDoc, query: str, k: int = 6, embedder: Any = None) -> list[dict[str, Any]]:
    """Hybrid search: BM25 + (optional) cosine similarity, fused with reciprocal rank fusion."""
    if not doc.chunks or not query.strip():
        return []
    bm = doc._bm25.scores(query) if doc._bm25 else [0.0] * len(doc.chunks)
    rankings = [sorted(range(len(bm)), key=lambda i: -bm[i])]
    dense = None
    if embedder is not None and doc._emb is not None:
        try:
            import numpy as np
            qv = np.asarray(embedder.embed([query])[0])
            m = np.asarray(doc._emb)
            dense = (m @ qv) / (np.linalg.norm(m, axis=1) * np.linalg.norm(qv) + 1e-9)
            rankings.append(list(np.argsort(-dense)))
        except Exception:
            dense = None
    fused: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r):
            fused[int(i)] = fused.get(int(i), 0) + 1 / (60 + rank)
    phrase = query.lower().strip()
    hits = []
    for i in sorted(fused, key=lambda i: -fused[i]):
        if bm[i] <= 0 and (dense is None or dense[i] < 0.35) and phrase not in doc.chunks[i].text.lower():
            continue
        c = doc.chunks[i]
        hits.append({"chunk_id": c.id, "page": c.page, "section": c.section, "score": round(fused[i] * 100, 3),
                     "bm25": round(bm[i], 3), "semantic": None if dense is None else round(float(dense[i]), 3),
                     "text": c.text, "highlights": highlight_spans(c.text, query)})
        if len(hits) >= k:
            break
    return hits


def highlight_spans(text: str, query: str) -> list[list[int]]:
    """Character [start, end] spans of query terms (and the exact phrase) inside text."""
    spans = []
    low = text.lower()
    phrase = query.lower().strip()
    if len(phrase) > 3:
        for m in re.finditer(re.escape(phrase), low):
            spans.append([m.start(), m.end()])
    for t in set(tokenize(query)):
        if len(t) < 3:
            continue
        for m in re.finditer(r"\b" + re.escape(t) + r"\w*", low):
            spans.append([m.start(), m.end()])
    spans.sort()
    merged: list[list[int]] = []
    for s in spans:
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(s)
    return merged


def find_in_text(doc: ReportDoc, phrase: str, limit: int = 50) -> list[dict[str, Any]]:
    """Exact (case-insensitive) find, like Ctrl+F, with page and context."""
    out = []
    p = phrase.lower()
    for pno, page in enumerate(doc.pages, 1):
        low = page.lower()
        for m in re.finditer(re.escape(p), low):
            a, b = max(0, m.start() - 90), min(len(page), m.end() + 90)
            out.append({"page": pno, "context": page[a:b].replace("\n", " "), "start": m.start() - a,
                        "end": m.end() - a})
            if len(out) >= limit:
                return out
    return out
