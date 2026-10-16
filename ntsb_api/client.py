"""Async client for NTSB's public CAROL API, report generator and docket site."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
import zipfile
from typing import Any, Optional

import httpx

from . import mock
from .config import Settings, settings as default_settings
from .docket import parse_docket_html
from .normalize import normalize_case, normalize_many
from .query import Rule, export_payload, search_payload


class NTSBError(RuntimeError):
    """Raised when NTSB returns an error or an unexpected payload."""


class _TTLCache:
    def __init__(self, ttl: int, max_items: int = 512) -> None:
        self.ttl, self.max_items = ttl, max_items
        self._d: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any:
        hit = self._d.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        self._d.pop(key, None)
        return None

    def set(self, key: str, val: Any) -> None:
        if len(self._d) >= self.max_items:
            self._d.pop(next(iter(self._d)))
        self._d[key] = (time.monotonic() + self.ttl, val)


def _key(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


# Column -> normalized-case key, used to evaluate rules locally (mock mode).
_LOCAL_COLUMNS = {
    "Event.Mode": "mode", "Event.NTSBNumber": "ntsb_number", "Event.City": "city",
    "Event.State": "state", "Event.EventDate": "event_date", "Event.HighestInjury": "highest_injury",
    "Event.EventType": "event_type", "Aircraft.AircraftMake": "make", "Aircraft.AircraftModel": "model",
    "Aircraft.RegistrationNumber": "registration", "Aircraft.Damage": "damage",
    "AviationOperation.OperatorName": "operator", "Airport.AirportId": "airport_id",
    "AviationOperation.RegulationFlightConductedUnder": "far_part",
    "Narrative.Factual": "narrative_factual", "Case.ProbableCause": "probable_cause",
    "Finding.FindingText": "findings", "HasSafetyRec": "has_safety_rec",
    "Weather.AccidentSiteCondition": "weather",
}


def local_match(case: dict[str, Any], rule: Rule) -> bool:
    key = _LOCAL_COLUMNS.get(rule.column)
    if key is None:
        return True  # unsupported locally: don't filter
    val = case.get(key)
    if key == "findings":
        val = " ".join(f["text"] for f in val or [])
    if isinstance(val, bool):
        val = "true" if val else "false"
    a, b, op = str(val or "").lower(), rule.value.lower(), rule.operator
    return {
        "is": a == b, "is not": a != b, "contains": b in a, "does not contain": b not in a,
        "starts with": a.startswith(b), "ends with": a.endswith(b),
        "is on or after": a >= b, "is on or before": a <= b, "is after": a > b, "is before": a < b,
        "is greater than": a > b, "is less than": a < b,
    }.get(op, True)


class NTSBClient:
    """High-level client.

    >>> async with NTSBClient() as c:
    ...     page = await c.search([Rule(column="Aircraft.AircraftMake", value="Boeing")])
    """

    def __init__(self, cfg: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg or default_settings
        self._http = httpx.AsyncClient(
            timeout=self.cfg.timeout_s, transport=transport, follow_redirects=True,
            headers={"User-Agent": self.cfg.user_agent, "Accept": "application/json, */*",
                     "Origin": "https://data.ntsb.gov", "Referer": "https://data.ntsb.gov/carol-main-public/query-builder"},
        )
        self._cache = _TTLCache(self.cfg.cache_ttl_s)
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def __aenter__(self) -> "NTSBClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ transport
    async def _request(self, method: str, url: str, **kw: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            async with self._lock:  # polite pacing against a public service
                wait = self.cfg.min_interval_s - (time.monotonic() - self._last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_call = time.monotonic()
            try:
                r = await self._http.request(method, url, **kw)
                if r.status_code >= 500 or r.status_code == 429:
                    raise NTSBError(f"NTSB returned HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise NTSBError(f"NTSB rejected the request (HTTP {r.status_code}): {r.text[:300]}")
                return r
            except (httpx.TransportError, NTSBError) as e:
                last_exc = e
                if isinstance(e, NTSBError) and "rejected" in str(e):
                    raise
                await asyncio.sleep(0.8 * attempt)
        raise NTSBError(f"NTSB request failed after {self.cfg.max_retries} attempts: {last_exc}")

    # ------------------------------------------------------------------ search
    async def search(self, rules: list[Rule], *, limit: int = 25, offset: int = 0,
                     match: str = "all", include_raw: bool = False) -> dict[str, Any]:
        """Paged search. Returns {"total", "limit", "offset", "items": [normalized cases]}."""
        ck = _key("search", [r.model_dump() for r in rules], limit, offset, match, include_raw)
        if (hit := self._cache.get(ck)) is not None:
            return hit

        if self.cfg.mock:
            cases = normalize_many(mock.MOCK_CASES, include_raw=include_raw)
            test = all if match == "all" else any
            cases = [c for c in cases if test(local_match(c, r) for r in rules)]
            result = {"total": len(cases), "limit": limit, "offset": offset,
                      "items": cases[offset:offset + limit], "source": "mock"}
        else:
            payload = search_payload(rules, limit=limit, offset=offset, match=match)
            r = await self._request("POST", f"{self.cfg.carol_base}/Main", json=payload)
            body = r.json()
            raw_items = body.get("Results") or body.get("results") or []
            result = {
                "total": body.get("ResultListCount", len(raw_items)),
                "max_result_count_reached": body.get("MaxResultCountReached"),
                "limit": limit, "offset": offset,
                "items": normalize_many(raw_items, include_raw=include_raw),
                "source": "ntsb-carol",
            }
        self._cache.set(ck, result)
        return result

    async def export(self, rules: list[Rule], *, limit: int = 500, match: str = "all") -> list[dict[str, Any]]:
        """Full-detail records (narratives, findings, vehicles) via CAROL FileExport (ZIP of JSON)."""
        ck = _key("export", [r.model_dump() for r in rules], limit, match)
        if (hit := self._cache.get(ck)) is not None:
            return hit
        if self.cfg.mock:
            test = all if match == "all" else any
            raw = [c for c in mock.MOCK_CASES
                   if test(local_match(normalize_case(c, include_raw=False), r) for r in rules)][:limit]
        else:
            r = await self._request("POST", f"{self.cfg.carol_base}/FileExport",
                                    json=export_payload(rules, fmt="data", limit=limit, match=match))
            raw = self._unzip_json(r.content)
        self._cache.set(ck, raw)
        return raw

    async def export_csv(self, rules: list[Rule], *, limit: int = 500) -> bytes:
        """CAROL's own summary CSV for a query."""
        if self.cfg.mock:
            import csv
            buf = io.StringIO()
            cases = [normalize_case(c, include_raw=False) for c in await self.export(rules, limit=limit)]
            cols = ["ntsb_number", "event_date", "city", "state", "make", "model", "operator",
                    "far_part", "highest_injury", "fatal", "probable_cause"]
            w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(cases)
            return buf.getvalue().encode()
        r = await self._request("POST", f"{self.cfg.carol_base}/FileExport",
                                json=export_payload(rules, fmt="summary", limit=limit))
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = next((n for n in z.namelist() if n.lower().endswith(".csv")), None)
            if not name:
                raise NTSBError("Export ZIP contained no CSV")
            return z.read(name)

    @staticmethod
    def _unzip_json(content: bytes) -> list[dict[str, Any]]:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                name = next((n for n in z.namelist() if n.lower().endswith(".json")), None)
                if not name:
                    return []
                data = json.loads(z.read(name))
        except zipfile.BadZipFile:
            data = json.loads(content)  # some deployments return plain JSON
        if isinstance(data, dict):
            data = data.get("data") or data.get("cases") or data.get("Results") or [data]
        return data

    # ------------------------------------------------------------------ single case
    async def get_case(self, ntsb_number: str) -> Optional[dict[str, Any]]:
        rules = [Rule(column="Event.NTSBNumber", operator="is", value=ntsb_number.upper())]
        raw = await self.export(rules, limit=5)
        if not raw:
            # fall back to the search API (less detail, but more tolerant)
            page = await self.search(rules, limit=1, include_raw=True)
            return page["items"][0] if page["items"] else None
        exact = [r for r in raw if (normalize_case(r, include_raw=False)["ntsb_number"] or "").upper()
                 == ntsb_number.upper()]
        return normalize_case((exact or raw)[0], include_raw=True)

    # ------------------------------------------------------------------ documents
    def report_pdf_url(self, mkey: int) -> str:
        return f"{self.cfg.report_base}/GenerateNewestReport/{mkey}/pdf"

    async def stream_report_pdf(self, mkey: int):
        """Async iterator over the PDF bytes of the newest generated report for a case."""
        if self.cfg.mock:
            raw = next((c for c in mock.MOCK_CASES if c["cm_mkey"] == mkey), None)
            data = mock.mock_report_pdf(raw) if raw else _sample_pdf(f"SYNTHETIC SAMPLE - case key {mkey}")

            async def _fake():
                yield data
            return _fake()
        req = self._http.build_request("GET", self.report_pdf_url(mkey), headers={"Accept": "application/pdf"})
        resp = await self._http.send(req, stream=True)
        if resp.status_code >= 400:
            await resp.aclose()
            raise NTSBError(f"Report not available (HTTP {resp.status_code})")

        async def _iter():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
        return _iter()

    async def report_pdf_bytes(self, mkey: int, max_bytes: int = 60_000_000) -> bytes:
        buf = bytearray()
        async for chunk in await self.stream_report_pdf(mkey):
            buf += chunk
            if len(buf) > max_bytes:
                raise NTSBError("Report PDF is larger than the configured limit")
        return bytes(buf)

    async def fetch_document(self, url: str, max_bytes: int = 60_000_000) -> bytes:
        """Download a docket/NTSB document. Only data.ntsb.gov / ntsb.gov hosts are allowed."""
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        if not (host == "ntsb.gov" or host.endswith(".ntsb.gov")):
            raise NTSBError("Only ntsb.gov document URLs can be fetched")
        if self.cfg.mock:
            raw = mock.MOCK_CASES[0]
            return mock.mock_report_pdf(raw)
        r = await self._request("GET", url, headers={"Accept": "application/pdf, */*"})
        if len(r.content) > max_bytes:
            raise NTSBError("Document is larger than the configured limit")
        return r.content

    async def docket(self, ntsb_number: str) -> dict[str, Any]:
        url = f"{self.cfg.docket_base}/?NTSBNumber={ntsb_number.upper()}"
        ck = _key("docket", url)
        if (hit := self._cache.get(ck)) is not None:
            return hit
        if self.cfg.mock:
            html = mock.mock_docket_html(ntsb_number)
        else:
            html = (await self._request("GET", url, headers={"Accept": "text/html"})).text
        items = parse_docket_html(html, base_url=f"{self.cfg.docket_base}/")
        out = {"ntsb_number": ntsb_number.upper(), "docket_url": url, "count": len(items), "items": items,
               "note": None if items else
               "No documents parsed. The docket may not be released yet, or the page is rendered client-side - open docket_url in a browser."}
        self._cache.set(ck, out)
        return out


def _sample_pdf(text: str) -> bytes:
    """Tiny valid one-page PDF (used only in mock mode)."""
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)
