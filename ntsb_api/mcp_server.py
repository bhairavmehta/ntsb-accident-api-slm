"""Optional MCP server so AI agents (Claude, LangGraph, CrewAI...) can query NTSB data as tools.

    pip install "ntsb-accident-api[mcp]"
    python -m ntsb_api.mcp_server          # stdio transport
"""
from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

from .client import NTSBClient
from .normalize import summarize
from .query import CaseFilters

mcp = FastMCP("ntsb-aviation")
_client: Optional[NTSBClient] = None


def _c() -> NTSBClient:
    global _client
    _client = _client or NTSBClient()
    return _client


@mcp.tool()
async def search_accidents(make: str | None = None, model: str | None = None, operator_name: str | None = None,
                           text: str | None = None, finding: str | None = None, date_from: str | None = None,
                           date_to: str | None = None, highest_injury: str | None = None,
                           airline_only: bool = False, limit: int = 20) -> dict:
    """Search NTSB aviation accident/incident investigations. Dates are YYYY-MM-DD."""
    f = CaseFilters(make=make, model=model, operator_name=operator_name, text=text, finding=finding,
                    date_from=date_from, date_to=date_to, highest_injury=highest_injury,
                    airline_only=airline_only)
    page = await _c().search(f.to_rules(), limit=min(limit, 100))
    return {**page, "items": [summarize(x) for x in page["items"]]}


@mcp.tool()
async def get_accident(ntsb_number: str) -> dict:
    """Full details for one investigation: narratives, probable cause, findings, aircraft, links."""
    case = await _c().get_case(ntsb_number)
    if case:
        case.pop("raw", None)
    return case or {"error": f"no case {ntsb_number}"}


@mcp.tool()
async def list_docket_documents(ntsb_number: str) -> dict:
    """List documents in the NTSB public docket for an investigation."""
    return await _c().docket(ntsb_number)


_docs: dict = {}


async def _doc(ntsb_number: str):
    from .cli import _load_doc
    key = ntsb_number.upper()
    if key not in _docs:
        _docs[key] = await _load_doc(_c(), key)
    return _docs[key]


@mcp.tool()
async def summarize_report(ntsb_number: str) -> dict:
    """Summarize an investigation report (headline, what happened, cause, factors, lessons) with page cites."""
    from .slm import ReportAnalyst, make_backend
    return await ReportAnalyst(make_backend()).summarize(await _doc(ntsb_number))


@mcp.tool()
async def search_report(ntsb_number: str, query: str, k: int = 6) -> list:
    """Find the passages of a report most relevant to a query, with page and section."""
    from .documents import search
    return search(await _doc(ntsb_number), query, k=k)


if __name__ == "__main__":
    mcp.run()
