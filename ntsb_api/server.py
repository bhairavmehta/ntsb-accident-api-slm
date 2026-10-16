"""FastAPI service: a clean, documented REST API over NTSB aviation accident data.

Run:  uvicorn ntsb_api.server:app --reload      (UI at /, Swagger at /docs, ReDoc at /redoc)
"""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from contextlib import asynccontextmanager
from importlib import resources
from datetime import date
from typing import Annotated, Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .client import NTSBClient, NTSBError
from .config import settings
from .normalize import normalize_case, summarize
from .query import CaseFilters, Rule, field_catalog, search_payload

TAGS = [
    {"name": "Cases", "description": "Search and retrieve NTSB aviation accident & incident investigations."},
    {"name": "Documents", "description": "Final/preliminary report PDFs and public docket documents."},
    {"name": "Analytics", "description": "Aggregations and bulk exports for analysis."},
    {"name": "Report intelligence", "description": "Load a report, search inside it, summarize it and ask "
                                                    "questions with an embedded small language model, and get chart data."},
    {"name": "Reference", "description": "Searchable fields, operators and allowed values."},
]

DESCRIPTION = """
A typed REST API over the **NTSB CAROL** investigation database (data.ntsb.gov), with
normalized JSON, report PDFs and docket documents.

* **Friendly search** – `GET /api/v1/cases?make=Boeing&airline_only=true&date_from=2020-01-01`
* **Advanced rules** – `POST /api/v1/cases/query` with any CAROL column/operator
* **Case detail** – narratives, probable cause, findings, aircraft, links
* **Documents** – `/report` streams the NTSB report PDF; `/docket` lists docket files
* **Report intelligence** – search inside a report, SLM summary and Q&A with page citations, chart data
* **Analytics** – `/stats` aggregates, `/export` downloads CSV/JSON

Data comes live from NTSB's public endpoints; results are cached for a short time.
Set `NTSB_MOCK=1` to run against built-in synthetic data (no network).
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .doc_routes import DocStore
    from .slm import ReportAnalyst, make_backend, make_embedder
    app.state.client = NTSBClient()
    app.state.docs = DocStore()
    app.state.analyst = ReportAnalyst(make_backend(), make_embedder())
    yield
    await app.state.client.aclose()


app = FastAPI(title="NTSB Aviation Accident API", version=__version__, description=DESCRIPTION,
              openapi_tags=TAGS, lifespan=lifespan,
              contact={"name": "NTSB data source", "url": "https://www.ntsb.gov/Pages/CAROL.aspx"})
from .doc_routes import router as doc_router  # noqa: E402

app.include_router(doc_router)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


def client() -> NTSBClient:
    return app.state.client


Client = Annotated[NTSBClient, Depends(client)]
NtsbNo = Annotated[str, Path(description="NTSB number, e.g. DCA24MA063", pattern=r"^[A-Za-z0-9]{5,20}$")]


def case_filters(
    text: Optional[str] = Query(None, description="Free text in the factual narrative"),
    probable_cause: Optional[str] = Query(None, description="Text in the probable cause"),
    finding: Optional[str] = Query(None, description="Text in an NTSB finding, e.g. 'thrust reverser'"),
    ntsb_number: Optional[str] = Query(None, description="NTSB number contains"),
    make: Optional[str] = Query(None, description="Aircraft make contains, e.g. Boeing"),
    model: Optional[str] = Query(None, description="Aircraft model contains, e.g. 737"),
    registration: Optional[str] = Query(None, description="Tail number contains"),
    operator_name: Optional[str] = Query(None, description="Operator / airline name contains"),
    airport: Optional[str] = Query(None, description="Airport ID contains, e.g. SEA"),
    city: Optional[str] = Query(None),
    state: Optional[str] = Query(None, description="US state code, e.g. WA"),
    country: Optional[str] = Query(None, description="CAROL country code"),
    date_from: Optional[date] = Query(None, description="Event date on/after (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Event date on/before (YYYY-MM-DD)"),
    highest_injury: Optional[Literal["Fatal", "Serious", "Minor", "None"]] = Query(None),
    event_type: Optional[Literal["ACC", "INC", "OCC"]] = Query(None, description="ACC accident, INC incident"),
    damage: Optional[Literal["Destroyed", "Substantial", "Minor", "None", "Unknown"]] = Query(None),
    weather: Optional[Literal["IMC", "VMC"]] = Query(None),
    airline_only: bool = Query(False, description="Only 14 CFR Part 121 air-carrier operations"),
    has_safety_rec: Optional[bool] = Query(None, description="Only cases with safety recommendations"),
) -> CaseFilters:
    return CaseFilters(**{k: v for k, v in locals().items()})


Filters = Annotated[CaseFilters, Depends(case_filters)]


# ---------------------------------------------------------------------------- models
class Vehicle(BaseModel):
    make: Optional[str] = None
    model: Optional[str] = None
    registration: Optional[str] = None
    category: Optional[str] = None
    damage: Optional[str] = None
    operator: Optional[str] = None
    far_part: Optional[str] = None
    flight_operation: Optional[str] = None
    engines: Optional[int] = None
    amateur_built: Any = None


class Finding(BaseModel):
    text: str
    cause_factor: Optional[str] = Field(None, description="C = cause, F = contributing factor (when provided)")


class Links(BaseModel):
    report_pdf: Optional[str] = Field(None, description="NTSB-generated report PDF")
    docket: Optional[str] = Field(None, description="Public docket page")
    carol_search: Optional[str] = None


class CaseSummary(BaseModel):
    ntsb_number: Optional[str] = None
    mkey: Optional[int] = Field(None, description="Internal NTSB case key used for report PDFs")
    event_date: Optional[str] = None
    event_type: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    registration: Optional[str] = None
    operator: Optional[str] = None
    far_part: Optional[str] = None
    highest_injury: Optional[str] = None
    fatal: Optional[int] = None
    serious: Optional[int] = None
    damage: Optional[str] = None
    report_status: Optional[str] = None
    has_safety_rec: Any = None
    links: Links


class CaseDetail(CaseSummary):
    event_id: Optional[str] = None
    mode: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    airport_id: Optional[str] = None
    airport_name: Optional[str] = None
    minor: Optional[int] = None
    weather: Optional[str] = None
    report_type: Optional[str] = None
    report_number: Optional[str] = None
    report_date: Optional[str] = None
    probable_cause: Optional[str] = None
    narrative_prelim: Optional[str] = None
    narrative_factual: Optional[str] = None
    narrative_analysis: Optional[str] = None
    findings: list[Finding] = []
    vehicles: list[Vehicle] = []
    raw: Optional[dict[str, Any]] = Field(None, description="Untouched upstream record")


class SearchPage(BaseModel):
    total: int
    limit: int
    offset: int
    source: str
    max_result_count_reached: Optional[bool] = None
    items: list[CaseSummary]


class RuleQuery(BaseModel):
    rules: list[Rule] = Field(..., min_length=1, examples=[[
        {"column": "Aircraft.AircraftMake", "operator": "contains", "value": "Boeing"},
        {"column": "Finding.FindingText", "operator": "contains", "value": "thrust reverser"},
    ]])
    match: Literal["all", "any"] = "all"
    limit: int = Field(25, ge=1, le=500)
    offset: int = Field(0, ge=0)
    aviation_only: bool = True


class DocketItem(BaseModel):
    title: str
    url: str
    file_type: Optional[str] = None
    pages: Optional[int] = None
    row: list[str] = []


class Docket(BaseModel):
    ntsb_number: str
    docket_url: str
    count: int
    items: list[DocketItem]
    note: Optional[str] = None


def _guard(e: NTSBError) -> HTTPException:
    return HTTPException(status_code=502, detail=f"Upstream NTSB error: {e}")


# ---------------------------------------------------------------------------- cases
@app.get("/api/v1/cases", response_model=SearchPage, tags=["Cases"], summary="Search cases")
async def search_cases(c: Client, f: Filters,
                       limit: int = Query(25, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Search aviation investigations with friendly filters (all AND-ed)."""
    try:
        page = await c.search(f.to_rules(), limit=limit, offset=offset)
    except NTSBError as e:
        raise _guard(e)
    return {**page, "items": [summarize(x) for x in page["items"]]}


@app.post("/api/v1/cases/query", response_model=SearchPage, tags=["Cases"], summary="Advanced rule query")
async def query_cases(c: Client, q: RuleQuery):
    """Run raw CAROL rules (any column from `/api/v1/fields`), matched `all` or `any`."""
    rules = list(q.rules)
    if q.aviation_only and not any(r.column == "Event.Mode" for r in rules):
        rules.insert(0, Rule(column="Event.Mode", operator="is", value="Aviation"))
    try:
        page = await c.search(rules, limit=q.limit, offset=q.offset, match=q.match)
    except NTSBError as e:
        raise _guard(e)
    return {**page, "items": [summarize(x) for x in page["items"]]}


@app.get("/api/v1/cases/{ntsb_number}", response_model=CaseDetail, tags=["Cases"], summary="Case detail")
async def get_case(c: Client, ntsb_number: NtsbNo,
                   include_raw: bool = Query(False, description="Include the untouched upstream record")):
    """Full case: narratives, probable cause, findings, aircraft and document links."""
    try:
        case = await c.get_case(ntsb_number)
    except NTSBError as e:
        raise _guard(e)
    if not case:
        raise HTTPException(404, f"No case found for {ntsb_number}")
    if not include_raw:
        case = {**case, "raw": None}
    return case


@app.get("/api/v1/cases/{ntsb_number}/report", tags=["Documents"], summary="Report PDF",
         response_class=StreamingResponse,
         responses={200: {"content": {"application/pdf": {}}, "description": "NTSB report PDF"}})
async def case_report(c: Client, ntsb_number: NtsbNo,
                      download: bool = Query(False, description="Send as attachment instead of inline")):
    """Stream the newest NTSB-generated report PDF for the case."""
    try:
        case = await c.get_case(ntsb_number)
        if not case or not case.get("mkey"):
            raise HTTPException(404, "Case or case key not found")
        stream = await c.stream_report_pdf(case["mkey"])
    except NTSBError as e:
        raise _guard(e)
    disp = "attachment" if download else "inline"
    return StreamingResponse(stream, media_type="application/pdf",
                             headers={"Content-Disposition": f'{disp}; filename="{ntsb_number.upper()}.pdf"'})


@app.get("/api/v1/cases/{ntsb_number}/docket", response_model=Docket, tags=["Documents"], summary="Docket documents")
async def case_docket(c: Client, ntsb_number: NtsbNo):
    """List documents in the case's public docket (factual reports, transcripts, photos)."""
    try:
        return await c.docket(ntsb_number)
    except NTSBError as e:
        raise _guard(e)


# ---------------------------------------------------------------------------- analytics
@app.get("/api/v1/stats", tags=["Analytics"], summary="Aggregate statistics")
async def stats(c: Client, f: Filters,
                sample: int = Query(1000, ge=10, le=5000, description="Max records to aggregate")):
    """Counts by year, make, injury, FAR part, state and top findings for the filtered set."""
    try:
        raw = await c.export(f.to_rules(), limit=sample)
    except NTSBError as e:
        raise _guard(e)
    cases = [normalize_case(r, include_raw=False) for r in raw]

    def top(key: str, n: int = 12) -> list[dict[str, Any]]:
        cnt = Counter((x.get(key) or "Unknown") for x in cases)
        return [{"key": k, "count": v} for k, v in cnt.most_common(n)]

    years = Counter((x.get("event_date") or "????")[:4] for x in cases)
    findings = Counter(fd["text"] for x in cases for fd in x.get("findings") or [])
    return {
        "records": len(cases),
        "fatalities": sum(x.get("fatal") or 0 for x in cases),
        "serious_injuries": sum(x.get("serious") or 0 for x in cases),
        "by_year": [{"key": k, "count": v} for k, v in sorted(years.items())],
        "by_make": top("make"), "by_injury": top("highest_injury"), "by_far_part": top("far_part"),
        "by_state": top("state"), "by_damage": top("damage"),
        "top_findings": [{"key": k, "count": v} for k, v in findings.most_common(15)],
    }


@app.get("/api/v1/export", tags=["Analytics"], summary="Bulk export (CSV or JSON)")
async def export(c: Client, f: Filters,
                 fmt: Literal["csv", "json", "ntsb-csv"] = Query("csv"),
                 limit: int = Query(500, ge=1, le=5000)):
    """`csv`/`json` = normalized records; `ntsb-csv` = CAROL's own summary CSV."""
    rules = f.to_rules()
    try:
        if fmt == "ntsb-csv":
            data = await c.export_csv(rules, limit=limit)
            return Response(data, media_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="ntsb_summary.csv"'})
        cases = [normalize_case(r, include_raw=False) for r in await c.export(rules, limit=limit)]
    except NTSBError as e:
        raise _guard(e)
    if fmt == "json":
        return Response(json.dumps(cases, indent=1), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="ntsb_cases.json"'})
    buf = io.StringIO()
    cols = ["ntsb_number", "event_date", "event_type", "city", "state", "country", "airport_id", "make", "model",
            "registration", "operator", "far_part", "highest_injury", "fatal", "serious", "minor", "damage",
            "weather", "report_status", "probable_cause"]
    w = csv.DictWriter(buf, fieldnames=cols + ["findings", "report_pdf", "docket"], extrasaction="ignore")
    w.writeheader()
    for x in cases:
        w.writerow({**x, "findings": " | ".join(fd["text"] for fd in x["findings"]),
                    "report_pdf": x["links"]["report_pdf"], "docket": x["links"]["docket"]})
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="ntsb_cases.csv"'})


# ---------------------------------------------------------------------------- reference
@app.get("/api/v1/fields", tags=["Reference"], summary="Searchable fields")
async def fields(group: Optional[str] = Query(None, description="e.g. Aircraft, Event, Narrative")):
    """Every CAROL column usable in `POST /api/v1/cases/query`, with operators and allowed values."""
    out = list(field_catalog().values())
    return [x for x in out if not group or x["group"].lower() == group.lower()]


@app.post("/api/v1/debug/payload", tags=["Reference"], summary="Preview the NTSB payload")
async def debug_payload(q: RuleQuery):
    """Shows the exact JSON this service would POST to NTSB for a rule query (no call is made)."""
    return search_payload(q.rules, limit=q.limit, offset=q.offset, match=q.match)


@app.get("/health", tags=["Reference"], summary="Health check")
async def health():
    a = getattr(app.state, "analyst", None)
    return {"status": "ok", "mock": settings.mock, "version": __version__, "slm": a.status() if a else None}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return resources.files("ntsb_api").joinpath("static/index.html").read_text("utf-8")
