"""Deterministic extraction of chartable facts from a report.

Everything here is rule-based and runs without a model, so visuals are always
available. When a language model is configured, ``slm.py`` can add a model-built
timeline for reports whose wording the rules miss.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .documents import STOP, ReportDoc, tokenize

_TIME = re.compile(r"\b(?:at|about|around|by)\s+(?:about\s+)?(\d{4})(?:\s*(?:local time|local|lt|cdt|cst|edt|est|pdt|pst|mdt|mst|akdt|hst|utc|z))?\b[,:]?\s*([^.]{8,220})\.", re.I)
_CLOCK = re.compile(r"\b(\d{1,2}:\d{2})(?:\s*(?:a\.?m\.?|p\.?m\.?))?\b[,:]?\s*([^.]{8,220})\.", re.I)
_ALT = re.compile(r"\b(\d{1,2},\d{3}|\d{3,5})\s*(?:ft|feet)\b(?:\s*(msl|agl))?", re.I)
_SPEED = re.compile(r"\b(\d{2,3})\s*(?:knots|kts|kt)\b", re.I)
_HOURS = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)\s+(total\s+)?(?:flight\s+)?hours?\b(?:\s+of\s+(?:total\s+)?(?:flight\s+)?experience)?(\s+in\s+(?:type|make and model|the\s+\S+))?", re.I)
_ROLE = re.compile(r"(captain|first officer|pilot|flight instructor|student pilot|check airman)", re.I)
_WIND = re.compile(r"wind\w*\s+(?:from\s+)?(\d{3})\s*(?:degrees|°)?\s+at\s+(\d{1,2})\s*knots(?:,?\s*gusting\s+(?:to\s+)?(\d{1,2}))?", re.I)
_INJ = re.compile(r"(\d+)\s+(fatal|serious|minor)", re.I)

# Words that carry no signal in an aviation report's term cloud.
_DOMAIN_STOP = STOP | set("""airplane aircraft flight report ntsb investigation accident incident sample synthetic
pilot crew time local about approximately reported information national transportation safety board
determined data page per also however following within without based found held event events certificate
airline transport probable cause causes findings finding involved operated occurrence near total hours
including type fatal serious minor part general investigator final date aviation injuries owner private
normally damage administrative location number defining aircraft make model registration engines""".split())


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def timeline(doc: ReportDoc, limit: int = 30) -> list[dict[str, Any]]:
    events, seen = [], set()
    for pno, page in enumerate(doc.pages, 1):
        flat = re.sub(r"\s*\n\s*", " ", page)
        for m in list(_TIME.finditer(flat)) + list(_CLOCK.finditer(flat)):
            raw_t, what = m.group(1), m.group(2).strip()
            if ":" in raw_t:
                h, mi = raw_t.split(":")
            else:
                h, mi = raw_t[:2], raw_t[2:]
            if not (0 <= int(h) <= 23 and 0 <= int(mi) <= 59):
                continue
            key = (raw_t, what[:40])
            if key in seen:
                continue
            seen.add(key)
            events.append({"time": f"{int(h):02d}:{mi}", "minutes": int(h) * 60 + int(mi),
                           "event": what[0].upper() + what[1:], "page": pno})
    events.sort(key=lambda e: (e["minutes"], e["page"]))
    return events[:limit]


def _context(text: str, pos: int, width: int = 60) -> str:
    seg = text[max(0, pos - width):pos]
    seg = re.split(r"[.;]\s", seg)[-1]
    return seg.strip(" ,")[-width:]


def metrics(doc: ReportDoc) -> dict[str, Any]:
    text = re.sub(r"\s*\n\s*", " ", doc.text)
    wind = []
    wind_spans = []
    for m in _WIND.finditer(text):
        d, sp, g = m.groups()
        wind.append({"direction": int(d), "speed": int(sp), "gust": int(g) if g else None})
        wind_spans.append(m.span())
    in_wind = lambda pos: any(a <= pos <= b for a, b in wind_spans)
    alts = []
    for m in _ALT.finditer(text):
        v = _num(m.group(1))
        if 50 <= v <= 60000:
            alts.append({"value": v, "ref": (m.group(2) or "").upper() or None, "context": _context(text, m.start())})
    speeds = [{"value": int(m.group(1)), "context": _context(text, m.start())}
              for m in _SPEED.finditer(text) if 5 <= int(m.group(1)) <= 600 and not in_wind(m.start())]
    hours = []
    for m in _HOURS.finditer(text):
        n = _num(m.group(1))
        if not 1 <= n <= 60000:
            continue
        back = text[max(0, m.start() - 220):m.start()]
        crew = re.findall(r"(captain|first officer|flight instructor|student pilot|check airman|copilot)", back, re.I)
        role = crew[-1].title() if crew else "Pilot"
        kind = "in type" if m.group(3) else ("total" if m.group(2) or "experience" in m.group(0).lower() else "other")
        hours.append({"role": role, "kind": kind, "hours": n})
    inj = Counter()
    for n, k in _INJ.findall(text[:4000]):
        inj[k.title()] = max(inj[k.title()], int(n))
    return {"altitudes_ft": alts[:30], "speeds_kt": speeds[:30], "experience_hours": hours[:12],
            "wind": wind[:5], "injuries": dict(inj)}


def finding_categories(findings: list[str]) -> list[dict[str, Any]]:
    """NTSB findings are hierarchical: 'Aircraft-Aircraft power plant-...' -> top two levels."""
    rows = []
    for f in findings:
        text = re.sub(r"\s+-\s*[CF]\s*$", "", f).strip()
        cf = re.search(r"-\s*([CF])\s*$", f)
        parts = [p.strip() for p in text.split("-") if p.strip()]
        if not parts:
            continue
        rows.append({"category": parts[0], "subcategory": parts[1] if len(parts) > 1 else parts[0],
                     "detail": " › ".join(parts[2:]) or None, "cause_factor": cf.group(1) if cf else None,
                     "text": text})
    return rows


def key_terms(doc: ReportDoc, k: int = 24) -> list[dict[str, Any]]:
    toks = [t for t in tokenize(doc.text) if t not in _DOMAIN_STOP and len(t) > 2
            and not re.search(r"\d", t)]
    if not toks:
        return []
    tf = Counter(toks)
    # weight terms concentrated in few chunks higher (rough IDF inside the document)
    df = Counter(w for c in doc.chunks for w in set(tokenize(c.text)))
    n = max(1, len(doc.chunks))
    scored = {w: c * (1 + math.log(n / (1 + df.get(w, 0)) + 1)) for w, c in tf.items() if c >= 2}
    bigrams = Counter(zip(toks, toks[1:]))
    for (a, b), c in bigrams.items():
        if c >= 2:
            scored[f"{a} {b}"] = c * 2.2
    ranked = sorted(scored.items(), key=lambda x: -x[1])
    chosen_bi = {w for w, _ in ranked[:k] if " " in w}
    covered = {p for b in chosen_bi for p in b.split()}
    top = [(w, sc) for w, sc in ranked if not (" " not in w and w in covered and tf[w] <= 1.5 * max(
        (bigrams[tuple(b.split())] for b in chosen_bi if w in b.split()), default=0))][:k]
    return [{"term": w, "weight": round(s, 2), "count": tf.get(w) or bigrams.get(tuple(w.split()), 0)} for w, s in top]


def findings_from_doc(doc: ReportDoc) -> list[str]:
    sec = doc.section_text("Findings")
    lines = [l.strip() for l in sec.splitlines() if l.strip()]
    return [l for l in lines if "-" in l and len(l) > 8][:30]


def build_insights(doc: ReportDoc, case: dict[str, Any] | None = None) -> dict[str, Any]:
    findings = [f"{f['text']} - {f.get('cause_factor') or ''}" for f in (case or {}).get("findings") or []] \
        or findings_from_doc(doc)
    cats = finding_categories(findings)
    m = metrics(doc)
    if case:
        for k, key in (("Fatal", "fatal"), ("Serious", "serious"), ("Minor", "minor")):
            if case.get(key) is not None:
                m["injuries"][k] = case[key]
    return {
        "doc_id": doc.doc_id,
        "pages": len(doc.pages),
        "words": len(doc.text.split()),
        "sections": doc.outline(),
        "timeline": timeline(doc),
        "metrics": m,
        "findings": cats,
        "finding_categories": [{"key": k, "count": v} for k, v in Counter(c["category"] for c in cats).most_common()],
        "key_terms": key_terms(doc),
    }
