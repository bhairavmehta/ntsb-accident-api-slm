"""Report intelligence endpoints: load a report, search it, summarize it, ask it, chart it."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .client import NTSBClient, NTSBError
from .documents import ReportDoc, build_doc, doc_from_case, find_in_text, pdf_pages, search
from .insights import build_insights
from .slm import ReportAnalyst

router = APIRouter(prefix="/api/v1", tags=["Report intelligence"])

MAX_UPLOAD = 60 * 1024 * 1024


class DocStore:
    """Small in-memory LRU of parsed reports (re-parsed on demand after eviction)."""

    def __init__(self, size: int = 64):
        self.size, self._d = size, OrderedDict()
        self.cases: dict[str, dict[str, Any]] = {}

    def get(self, doc_id: str) -> Optional[ReportDoc]:
        d = self._d.get(doc_id)
        if d:
            self._d.move_to_end(doc_id)
        return d

    def put(self, doc: ReportDoc) -> ReportDoc:
        self._d[doc.doc_id] = doc
        self._d.move_to_end(doc.doc_id)
        while len(self._d) > self.size:
            self._d.popitem(last=False)
        return doc


def _store(req: Request) -> DocStore:
    return req.app.state.docs


def _analyst(req: Request) -> ReportAnalyst:
    return req.app.state.analyst


def _client(req: Request) -> NTSBClient:
    return req.app.state.client


def _doc(req: Request, doc_id: str) -> ReportDoc:
    d = _store(req).get(doc_id)
    if not d:
        raise HTTPException(404, "Document not loaded. Load it with POST /api/v1/documents/from-case/{ntsb_number} "
                                 "or /documents/upload, then retry.")
    return d


def _meta(doc: ReportDoc) -> dict[str, Any]:
    return {"doc_id": doc.doc_id, "title": doc.title, "source": doc.source, "pages": len(doc.pages),
            "words": len(doc.text.split()), "chunks": len(doc.chunks), "sections": doc.outline(), "meta": doc.meta}


# ---------------------------------------------------------------------------- loading
@router.post("/documents/from-case/{ntsb_number}", summary="Load a case's report")
async def load_from_case(req: Request, ntsb_number: str,
                         source: Literal["auto", "report", "case"] = Query(
                             "auto", description="report = NTSB PDF, case = structured record, auto = PDF then record")):
    """Fetches the NTSB report PDF for a case, extracts text by page, splits it into NTSB sections
    and builds a search index. Falls back to the structured case record if no PDF is published."""
    num = ntsb_number.upper()
    doc_id = f"case-{num}"
    if (d := _store(req).get(doc_id)) and source == "auto":
        return _meta(d)
    c = _client(req)
    try:
        case = await c.get_case(num)
    except NTSBError as e:
        raise HTTPException(502, f"Upstream NTSB error: {e}")
    if not case:
        raise HTTPException(404, f"No case found for {num}")
    _store(req).cases[doc_id] = case
    doc, note = None, None
    if source in ("auto", "report") and case.get("mkey"):
        try:
            pages = pdf_pages(await c.report_pdf_bytes(case["mkey"]))
            if sum(len(p) for p in pages) > 200:
                doc = build_doc(doc_id, f"NTSB report {num}", pages, source="report-pdf",
                                meta={"ntsb_number": num, "mkey": case["mkey"]}, embedder=_analyst(req).embedder)
            else:
                note = "Report PDF has no extractable text (it may be scanned); using the case record."
        except Exception as e:
            if source == "report":
                raise HTTPException(502, f"Could not load the report PDF: {e}")
            note = f"Report PDF unavailable ({e}); using the case record."
    if doc is None:
        doc = doc_from_case(case)
        doc.doc_id = doc_id
    if note:
        doc.meta["note"] = note
    _store(req).put(doc)
    return _meta(doc)


@router.post("/documents/upload", summary="Upload a PDF")
async def upload(req: Request, file: UploadFile = File(..., description="A PDF report or docket document")):
    """Analyse any PDF you already have (final report, docket factual report, CVR transcript)."""
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "PDF is larger than 60 MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Only PDF files are supported")
    doc_id = "upload-" + hashlib.sha1(data).hexdigest()[:12]
    try:
        pages = pdf_pages(data)
    except Exception as e:
        raise HTTPException(422, f"Could not read the PDF: {e}")
    doc = build_doc(doc_id, file.filename or "Uploaded PDF", pages, source="upload",
                    meta={"filename": file.filename}, embedder=_analyst(req).embedder)
    return _meta(_store(req).put(doc))


class UrlIn(BaseModel):
    url: str = Field(..., description="An ntsb.gov document URL, e.g. a docket docBLOB link")
    title: Optional[str] = None


@router.post("/documents/from-url", summary="Load a docket document")
async def from_url(req: Request, body: UrlIn):
    """Load a document from an ntsb.gov URL (for example a link from `/cases/{n}/docket`)."""
    try:
        data = await _client(req).fetch_document(body.url)
        pages = pdf_pages(data)
    except NTSBError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(422, f"Could not read that document as a PDF: {e}")
    doc_id = "url-" + hashlib.sha1(body.url.encode()).hexdigest()[:12]
    doc = build_doc(doc_id, body.title or body.url.rsplit("/", 1)[-1], pages, source="docket",
                    meta={"url": body.url}, embedder=_analyst(req).embedder)
    return _meta(_store(req).put(doc))


@router.get("/documents/{doc_id}", summary="Document outline")
async def get_doc(req: Request, doc_id: str):
    return _meta(_doc(req, doc_id))


@router.get("/documents/{doc_id}/pages/{page}", summary="Page text")
async def page_text(req: Request, doc_id: str, page: int):
    d = _doc(req, doc_id)
    if not 1 <= page <= len(d.pages):
        raise HTTPException(404, f"Page must be between 1 and {len(d.pages)}")
    return {"page": page, "text": d.pages[page - 1]}


# ---------------------------------------------------------------------------- search
@router.get("/documents/{doc_id}/search", summary="Search inside a report")
async def doc_search(req: Request, doc_id: str, q: str = Query(..., min_length=2, examples=["reverse thrust"]),
                     k: int = Query(8, ge=1, le=30),
                     mode: Literal["relevant", "exact"] = Query("relevant",
                         description="relevant = ranked passages (BM25 + embeddings); exact = every literal match")):
    d = _doc(req, doc_id)
    if mode == "exact":
        hits = find_in_text(d, q)
        return {"query": q, "mode": "exact", "count": len(hits), "hits": hits}
    hits = search(d, q, k=k, embedder=_analyst(req).embedder)
    return {"query": q, "mode": "relevant", "count": len(hits), "hits": hits}


# ---------------------------------------------------------------------------- SLM
@router.get("/documents/{doc_id}/summary", summary="Summarize a report")
async def summary(req: Request, doc_id: str, refresh: bool = False):
    """Structured summary (headline, what happened, probable cause, factors, key facts, lessons)
    with page citations. Written by the small language model when one is configured, otherwise extractive."""
    return await _analyst(req).summarize(_doc(req, doc_id), refresh=refresh)


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, examples=["Why did the airplane leave the runway?"])
    k: int = Field(6, ge=1, le=12, description="Passages retrieved as context")
    stream: bool = Field(False, description="Stream newline-delimited JSON events")


@router.post("/documents/{doc_id}/ask", summary="Ask a question about a report")
async def ask(req: Request, doc_id: str, body: AskIn):
    """Retrieval-augmented answer grounded in the report, with the source passages returned."""
    d, a = _doc(req, doc_id), _analyst(req)
    if body.stream:
        async def gen():
            async for ev in a.ask_stream(d, body.question, k=body.k):
                yield json.dumps(ev) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    return await a.ask(d, body.question, k=body.k)


@router.get("/documents/{doc_id}/insights", summary="Visualization data")
async def insights(req: Request, doc_id: str,
                   model_timeline: bool = Query(False, description="Ask the model to build a timeline if rules find none")):
    """Chart-ready data: sequence-of-events timeline, findings tree, injuries, altitudes, speeds,
    crew experience, wind, section lengths and key terms."""
    d = _doc(req, doc_id)
    out = build_insights(d, _store(req).cases.get(doc_id))
    if model_timeline and not out["timeline"]:
        out["timeline"] = await _analyst(req).llm_timeline(d)
        out["timeline_source"] = "model"
    return out


@router.get("/slm/status", summary="Language model status")
async def slm_status(req: Request):
    return _analyst(req).status()
