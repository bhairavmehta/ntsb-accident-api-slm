"""Normalize NTSB CAROL records into one stable schema.

CAROL returns records in two shapes:

* the search API (``Query/Main``) returns ``{"Fields": [{"FieldName": .., "Values": [..]}]}``
* the export API (``Query/FileExport``) returns flat JSON with ``cm_``-prefixed keys
  and nested ``cm_vehicles`` / ``cm_findings`` arrays.

Field names have shifted over time, so every output field is resolved from a list
of candidate keys, matched case-insensitively. The untouched upstream record is
always kept under ``raw`` so nothing is lost if NTSB adds or renames fields.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .config import settings


def flatten_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Turn the search API's ``Fields`` list into a plain dict."""
    if isinstance(record.get("Fields"), list):
        out: dict[str, Any] = {}
        for f in record["Fields"]:
            name = f.get("FieldName") or f.get("Name")
            vals = f.get("Values")
            if vals is None:
                vals = [f.get("Value")]
            out[name] = vals[0] if isinstance(vals, list) and len(vals) == 1 else vals
        return out
    return record


def _index(d: dict[str, Any]) -> dict[str, Any]:
    idx: dict[str, Any] = {}
    for k, v in d.items():
        key = k.lower().replace("_", "")
        if key.startswith("cm"):
            idx.setdefault(key[2:], v)
        idx.setdefault(key, v)
    return idx


def _pick(idx: dict[str, Any], *names: str) -> Any:
    for n in names:
        v = idx.get(n.lower().replace("_", ""))
        if v not in (None, "", []):
            return v
    return None


def _int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, list):
        v = ", ".join(str(x) for x in v if x not in (None, ""))
    s = str(v).strip()
    return s or None


def _date(v: Any) -> Optional[str]:
    s = _str(v)
    return s[:10] if s else None


def _vehicle(v: dict[str, Any]) -> dict[str, Any]:
    i = _index(v)
    return {
        "make": _str(_pick(i, "make", "aircraftMake")),
        "model": _str(_pick(i, "model", "aircraftModel")),
        "registration": _str(_pick(i, "registrationNumber", "registration", "nNumber")),
        "category": _str(_pick(i, "aircraftCategory", "category")),
        "damage": _str(_pick(i, "damageLevel", "damage", "aircraftDamage")),
        "operator": _str(_pick(i, "operatorName", "operator")),
        "far_part": _str(_pick(i, "regulationFlightConductedUnder", "farPart", "far")),
        "flight_operation": _str(_pick(i, "flightOperationType", "operationType")),
        "engines": _int(_pick(i, "numberOfEngines", "engineCount")),
        "amateur_built": _pick(i, "amateurBuilt"),
    }


def _findings(raw: Any) -> list[dict[str, Any]]:
    out = []
    for f in raw or []:
        if isinstance(f, str):
            out.append({"text": f, "cause_factor": None})
            continue
        i = _index(f)
        out.append({
            "text": _str(_pick(i, "findingText", "text", "finding", "findingDescription")),
            "cause_factor": _str(_pick(i, "causeFactor", "cause", "findingType")),
        })
    return [f for f in out if f["text"]]


def normalize_case(record: dict[str, Any], *, include_raw: bool = True) -> dict[str, Any]:
    flat = flatten_fields(record)
    i = _index(flat)
    vehicles_raw = _pick(i, "vehicles", "aircraft") or []
    vehicles = [_vehicle(v) for v in vehicles_raw if isinstance(v, dict)]
    first = vehicles[0] if vehicles else {}

    ntsb = _str(_pick(i, "ntsbNum", "ntsbNo", "ntsbNumber", "ntsbnum"))
    mkey = _int(_pick(i, "mkey", "caseKey", "mKey"))

    case = {
        "ntsb_number": ntsb,
        "mkey": mkey,
        "event_id": _str(_pick(i, "eventId", "ev_id", "eventID")),
        "event_date": _date(_pick(i, "eventDate", "date")),
        "event_type": _str(_pick(i, "eventType")),
        "mode": _str(_pick(i, "mode")),
        "city": _str(_pick(i, "city")),
        "state": _str(_pick(i, "state")),
        "country": _str(_pick(i, "country")),
        "latitude": _float(_pick(i, "latitude", "lat")),
        "longitude": _float(_pick(i, "longitude", "lon", "lng")),
        "airport_id": _str(_pick(i, "airportId", "airportCode")),
        "airport_name": _str(_pick(i, "airportName")),
        "highest_injury": _str(_pick(i, "highestInjury", "highestInjuryLevel")),
        "fatal": _int(_pick(i, "fatalInjuryCount", "totalFatalInjuries", "fatal")),
        "serious": _int(_pick(i, "seriousInjuryCount", "totalSeriousInjuries", "serious")),
        "minor": _int(_pick(i, "minorInjuryCount", "totalMinorInjuries", "minor")),
        "make": first.get("make") or _str(_pick(i, "make", "aircraftMake")),
        "model": first.get("model") or _str(_pick(i, "model", "aircraftModel")),
        "registration": first.get("registration") or _str(_pick(i, "registrationNumber", "nNumber")),
        "operator": first.get("operator") or _str(_pick(i, "operatorName", "operator")),
        "far_part": first.get("far_part") or _str(_pick(i, "regulationFlightConductedUnder", "farPart")),
        "damage": first.get("damage") or _str(_pick(i, "damage", "damageLevel")),
        "weather": _str(_pick(i, "weatherCondition", "accidentSiteCondition", "basicWeather")),
        "report_status": _str(_pick(i, "reportStatus", "completionStatus", "status")),
        "report_type": _str(_pick(i, "reportType")),
        "report_number": _str(_pick(i, "reportNo", "reportNumber")),
        "report_date": _date(_pick(i, "originalPublishedDate", "originalPublishDate", "publishedDate")),
        "has_safety_rec": _pick(i, "hasSafetyRec", "hasSafetyRecommendations"),
        "probable_cause": _str(_pick(i, "probableCause", "probCause")),
        "narrative_prelim": _str(_pick(i, "prelimNarrative", "narrativePrelim")),
        "narrative_factual": _str(_pick(i, "factualNarrative", "narrativeFactual")),
        "narrative_analysis": _str(_pick(i, "analysisNarrative", "narrativeAnalysis")),
        "findings": _findings(_pick(i, "findings")),
        "vehicles": vehicles,
        "links": links_for(ntsb, mkey),
    }
    if include_raw:
        case["raw"] = record
    return case


def links_for(ntsb_number: Optional[str], mkey: Optional[int]) -> dict[str, Optional[str]]:
    return {
        "report_pdf": f"{settings.report_base}/GenerateNewestReport/{mkey}/pdf" if mkey else None,
        "docket": f"{settings.docket_base}/?NTSBNumber={ntsb_number}" if ntsb_number else None,
        "carol_search": (
            "https://data.ntsb.gov/carol-main-public/basic-search"
        ),
    }


def summarize(case: dict[str, Any]) -> dict[str, Any]:
    """Lightweight list-view projection of a normalized case."""
    keys = ("ntsb_number", "mkey", "event_date", "event_type", "city", "state", "country",
            "make", "model", "registration", "operator", "far_part", "highest_injury",
            "fatal", "serious", "damage", "report_status", "has_safety_rec", "links")
    return {k: case.get(k) for k in keys}


def normalize_many(records: Iterable[dict[str, Any]], include_raw: bool = False) -> list[dict[str, Any]]:
    return [normalize_case(r, include_raw=include_raw) for r in records]
