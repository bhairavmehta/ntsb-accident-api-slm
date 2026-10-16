import httpx
import pytest

from ntsb_api.client import NTSBClient
from ntsb_api.config import Settings
from ntsb_api.query import Rule
from ntsb_api.server import app


@pytest.fixture
async def api():
    app.state.client = NTSBClient()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_search_and_detail(api):
    r = await api.get("/api/v1/cases", params={"make": "boeing", "airline_only": True, "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0 and len(body["items"]) <= 5
    assert all(x["make"] == "BOEING" and x["far_part"].startswith("Part 121") for x in body["items"])
    num = body["items"][0]["ntsb_number"]
    d = (await api.get(f"/api/v1/cases/{num}")).json()
    assert d["ntsb_number"] == num and d["findings"] and d["narrative_factual"]


async def test_documents_stats_export(api):
    num = (await api.get("/api/v1/cases", params={"limit": 1})).json()["items"][0]["ntsb_number"]
    pdf = await api.get(f"/api/v1/cases/{num}/report")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    dk = (await api.get(f"/api/v1/cases/{num}/docket")).json()
    assert dk["count"] == 5
    st = (await api.get("/api/v1/stats")).json()
    assert st["records"] == 60 and st["by_year"]
    csv = await api.get("/api/v1/export", params={"fmt": "csv", "make": "airbus"})
    assert csv.text.startswith("ntsb_number,") and "AIRBUS" in csv.text


async def test_rule_query_and_fields(api):
    r = await api.post("/api/v1/cases/query", json={"rules": [
        {"column": "Finding.FindingText", "operator": "contains", "value": "thrust reverser"}]})
    assert r.status_code == 200 and r.json()["total"] > 0
    bad = await api.post("/api/v1/cases/query", json={"rules": [{"column": "X.Y", "value": "1"}]})
    assert bad.status_code == 422
    assert len((await api.get("/api/v1/fields")).json()) == 50


async def test_live_path_uses_carol_contract():
    """Live (non-mock) mode posts the CAROL payload and parses the response shape."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(req.url), req.content
        return httpx.Response(200, json={"ResultListCount": 1, "MaxResultCountReached": False, "Results": [
            {"Fields": [{"FieldName": "NtsbNo", "Values": ["DCA19MA086"]}, {"FieldName": "Mkey", "Values": ["99"]}]}]})

    c = NTSBClient(Settings(mock=False, min_interval_s=0), transport=httpx.MockTransport(handler))
    page = await c.search([Rule(column="Aircraft.AircraftMake", value="Boeing")])
    await c.aclose()
    assert seen["url"].endswith("/carol-main-public/api/Query/Main")
    assert b'"Aircraft.AircraftMake"' in seen["body"]
    assert page["total"] == 1 and page["items"][0]["ntsb_number"] == "DCA19MA086"
