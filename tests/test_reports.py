import json

import httpx
import pytest

from ntsb_api import mock
from ntsb_api.client import NTSBClient
from ntsb_api.doc_routes import DocStore
from ntsb_api.documents import build_doc, find_in_text, pdf_pages, search
from ntsb_api.insights import build_insights
from ntsb_api.pdfwrite import text_pdf
from ntsb_api.server import app
from ntsb_api.slm import OllamaBackend, OpenAICompatBackend, ReportAnalyst


def _doc(i=3):
    return build_doc("d", "t", pdf_pages(mock.mock_report_pdf(mock.MOCK_CASES[i])))


def test_sections_chunks_and_search():
    d = _doc()
    names = [s["name"] for s in d.sections]
    assert {"Analysis", "Probable Cause and Findings", "History of Flight", "Pilot Information"} <= set(names)
    hits = search(d, "pilot flight experience hours")
    assert hits and hits[0]["section"] == "Pilot Information" and hits[0]["page"] >= 1
    assert hits[0]["highlights"]
    assert find_in_text(d, "probable cause")


def test_insights_extract_chartable_facts():
    i = build_insights(_doc(10))
    assert len(i["timeline"]) >= 3 and i["timeline"] == sorted(i["timeline"], key=lambda e: e["minutes"])
    roles = {h["role"] for h in i["metrics"]["experience_hours"]}
    assert {"Captain", "First Officer"} <= roles
    assert i["metrics"]["wind"][0]["direction"] == 240
    assert i["finding_categories"] and i["key_terms"]


def test_multi_page_pdf_writer():
    blocks = [("h2", "History of Flight"), ("p", "word " * 4000)]
    assert len(pdf_pages(text_pdf(blocks, "T"))) > 3


class FakeLLM:
    name, model = "fake", "fake-1b"

    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def info(self):
        return {"backend": self.name, "model": self.model, "generative": True}

    async def complete(self, messages, **kw):
        self.seen.append(messages)
        return self.reply

    async def stream(self, messages, **kw):
        self.seen.append(messages)
        for w in self.reply.split(" "):
            yield w + " "


async def test_generative_summary_is_grounded_and_parsed():
    reply = '```json\n{"headline":"Reverser deployed late","what_happened":"The left reverser lagged [p.1].",' \
            '"probable_cause":"Degraded actuator [p.1]","contributing_factors":"Training","key_facts":[],"safety_lessons":[]}\n```'
    llm = FakeLLM(reply)
    s = await ReportAnalyst(llm).summarize(_doc())
    assert s["mode"] == "generative" and s["headline"] == "Reverser deployed late"
    assert s["contributing_factors"] == ["Training"]
    prompt = llm.seen[0][1]["content"]
    assert "[p.1]" in prompt and "Probable Cause" in prompt  # model only sees labelled report text


async def test_bad_model_output_falls_back_to_extractive():
    s = await ReportAnalyst(FakeLLM("not json at all")).summarize(_doc())
    assert s["mode"] == "extractive" and "warning" in s


async def test_ask_returns_sources_and_streams():
    llm = FakeLLM("About 4 seconds later [S1] [p.1].")
    an = ReportAnalyst(llm)
    r = await an.ask(_doc(10), "How late did the left thrust reverser deploy?")
    assert r["mode"] == "generative" and r["sources"] and "[S1]" in llm.seen[0][1]["content"]
    events = [e async for e in an.ask_stream(_doc(10), "reverser deploy")]
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"


async def test_ollama_and_openai_wire_formats():
    def ollama(req):
        body = json.loads(req.content)
        assert req.url.path == "/api/chat" and body["model"] == "qwen2.5:1.5b" and body["stream"]
        lines = [json.dumps({"message": {"content": t}}) for t in ("Hel", "lo")]
        return httpx.Response(200, text="\n".join(lines))

    def openai(req):
        assert req.url.path == "/v1/chat/completions"
        chunks = [f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}' for t in ("Hi", "!")]
        return httpx.Response(200, text="\n\n".join(chunks + ["data: [DONE]"]))

    o = OllamaBackend("qwen2.5:1.5b", "http://x", transport=httpx.MockTransport(ollama))
    assert await o.complete([{"role": "user", "content": "q"}]) == "Hello"
    oa = OpenAICompatBackend("m", "http://x/v1", transport=httpx.MockTransport(openai))
    assert await oa.complete([{"role": "user", "content": "q"}]) == "Hi!"


@pytest.fixture
async def api():
    app.state.client, app.state.docs, app.state.analyst = NTSBClient(), DocStore(), ReportAnalyst()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_report_endpoints(api):
    num = (await api.get("/api/v1/cases", params={"finding": "thrust reverser", "limit": 1})).json()["items"][0]["ntsb_number"]
    meta = (await api.post(f"/api/v1/documents/from-case/{num}")).json()
    assert meta["source"] == "report-pdf" and meta["pages"] >= 1
    did = meta["doc_id"]
    assert (await api.get(f"/api/v1/documents/{did}/summary")).json()["probable_cause"]
    s = (await api.get(f"/api/v1/documents/{did}/search", params={"q": "reverse thrust"})).json()
    assert s["count"] > 0
    e = (await api.get(f"/api/v1/documents/{did}/search", params={"q": "thrust", "mode": "exact"})).json()
    assert e["count"] > 0 and "context" in e["hits"][0]
    a = (await api.post(f"/api/v1/documents/{did}/ask", json={"question": "Why did the airplane yaw?"})).json()
    assert a["answer"] and a["sources"]
    st = await api.post(f"/api/v1/documents/{did}/ask", json={"question": "reverser", "stream": True})
    evs = [json.loads(l) for l in st.text.splitlines()]
    assert evs[0]["type"] == "sources" and evs[-1]["type"] == "done"
    ins = (await api.get(f"/api/v1/documents/{did}/insights")).json()
    assert ins["timeline"] and ins["metrics"]["injuries"]
    assert (await api.get("/api/v1/documents/nope/summary")).status_code == 404


async def test_upload_pdf(api):
    pdf = text_pdf([("h2", "Probable Cause"), ("p", "The pilot's failure to maintain airspeed.")], "Upload test")
    r = await api.post("/api/v1/documents/upload", files={"file": ("r.pdf", pdf, "application/pdf")})
    assert r.status_code == 200 and r.json()["source"] == "upload"
    bad = await api.post("/api/v1/documents/upload", files={"file": ("x.txt", b"hello", "text/plain")})
    assert bad.status_code == 415
