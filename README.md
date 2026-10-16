# NTSB Aviation Accident API

A typed REST API, Python client, CLI, MCP tool server and interactive web UI over the
NTSB's public aviation investigation data (CAROL, report PDFs, public dockets), with an
embedded small language model that summarizes a selected report, answers questions about it
with page citations, searches inside it and turns it into charts.

```
             ┌──────────────── ntsb-accident-api ────────────────┐
 Browser UI ─┤  FastAPI  /api/v1/*   (OpenAPI → /docs, /redoc)    │
 CLI (ntsb) ─┤  NTSBClient: payload builder · retries · pacing ·  ├──► data.ntsb.gov
 MCP agents ─┤  TTL cache · normalizer · docket parser            │    • carol-main-public/api/Query/{Main,FileExport}
             └────────────────────────────────────────────────────┘    • carol-repgen/api/Aviation/ReportMain/.../pdf
                                                                         • Docket/?NTSBNumber=...
```

## Quick start

```bash
pip install -e ".[dev]"          # add ,mcp for the MCP server
ntsb serve                        # → http://127.0.0.1:8000  (UI), /docs (Swagger), /redoc
NTSB_MOCK=1 ntsb serve            # offline: built-in synthetic data, no NTSB calls
pytest                            # runs against mocked NTSB responses
```

Docker: `docker build -t ntsb-api . && docker run -p 8000:8000 ntsb-api` (add `--build-arg WITH_SLM=1` to bake in the model)

## Report intelligence (embedded small language model)

Open any case, or upload a PDF, or pick a docket document, and the report becomes a searchable,
summarizable document:

```
report PDF ──► text per page ──► NTSB sections ──► ~120-word passages (page + section tagged)
                                                     │
                     ┌───────────────────────────────┼──────────────────────────────┐
                 search index                  small language model           rule-based extractors
           BM25 + bge-small embeddings      (Qwen2.5-1.5B, in-process)      timeline · findings tree ·
           fused with reciprocal rank       summary + grounded Q&A           injuries · crew hours ·
                                            citing [p.N] / [S1]              wind · altitudes · terms
```

| Tab in the UI | Endpoint | What you get |
|---|---|---|
| Summary | `GET /api/v1/documents/{id}/summary` | Headline, what happened, probable cause, contributing factors, key facts, safety lessons — every claim tagged with its page |
| Search report | `GET /api/v1/documents/{id}/search?q=...&mode=relevant\|exact` | Ranked passages with highlighted terms, or every literal match with context (find-in-page) |
| Ask | `POST /api/v1/documents/{id}/ask` (`"stream": true` for token streaming) | Answer written from the retrieved passages, plus the passages themselves as evidence |
| Visuals | `GET /api/v1/documents/{id}/insights` | Chart data: sequence of events, findings by category (cause vs factor), injuries, crew experience, report anatomy, numbers mentioned, key terms |

Load a document first: `POST /api/v1/documents/from-case/{ntsb_number}` (report PDF, falling back to
the case record), `POST /api/v1/documents/upload` (any PDF), or `POST /api/v1/documents/from-url`
(an ntsb.gov docket link). `GET /api/v1/slm/status` shows which model is active.

### Turning the model on

```bash
pip install -e ".[slm]"          # llama-cpp-python + huggingface_hub + fastembed
ntsb model pull                   # ~1 GB: Qwen2.5-1.5B-Instruct Q4_K_M GGUF + bge-small embeddings
ntsb serve                        # the model now runs inside the API process, on CPU
```

| `NTSB_SLM_BACKEND` | Runs where | Setup |
|---|---|---|
| `auto` (default) | first available of the three below, else extractive | – |
| `llamacpp` | inside this process (embedded) | `ntsb model pull`, or `NTSB_SLM_MODEL_PATH=/path/model.gguf` |
| `ollama` | local Ollama daemon | `ollama pull qwen2.5:1.5b`; `NTSB_OLLAMA_MODEL`, `OLLAMA_HOST` |
| `openai` | any OpenAI-compatible server (llama.cpp server, vLLM, LM Studio) | `NTSB_OPENAI_BASE`, `NTSB_OPENAI_MODEL`, `NTSB_OPENAI_KEY` |
| `extractive` | no model | Summaries and answers are sentences selected from the report |

Model choices that fit: `qwen2.5-1.5b-instruct` (default; fast on a laptop CPU), `Llama-3.2-3B-Instruct`
or `Phi-3.5-mini` (better writing, ~2x slower), `qwen2.5-0.5b` (fastest, weaker). Point
`NTSB_SLM_REPO` / `NTSB_SLM_FILE` at another GGUF to switch. Set `NTSB_EMBEDDINGS=off` for
keyword-only search.

**How answers stay grounded.** The model never sees the open internet or other reports. Long reports
are cut to a priority-ordered context (probable cause, findings, analysis, history of flight, …) that
fits a small model's window, each excerpt labelled with its page. Q&A retrieves the most relevant
passages first and returns them with the answer, so every claim can be checked. If the model returns
malformed output or is unreachable, the API falls back to the extractive result and says so in a
`warning` field. Treat model output as a reading aid, not as NTSB findings.

**Limits.** Scanned PDFs without a text layer fall back to the case record (add OCR such as
`ocrmypdf` upstream if you need them). Very long reports (100+ pages) are summarized from their
highest-priority sections, not cover to cover.

CLI: `ntsb summarize DCA24MA063`, `ntsb ask DCA24MA063 "What did the CVR show?"`, `ntsb find DCA24MA063 "fuel"`.
MCP: `summarize_report` and `search_report` tools are added for agents.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET  | `/api/v1/cases` | Search with friendly filters (`make`, `model`, `operator_name`, `date_from`, `date_to`, `highest_injury`, `event_type`, `finding`, `probable_cause`, `text`, `state`, `airport`, `airline_only`, …) + `limit`/`offset` |
| POST | `/api/v1/cases/query` | Advanced: any CAROL column/operator, matched `all` or `any` |
| GET  | `/api/v1/cases/{ntsb_number}` | Full case: narratives, probable cause, findings (cause/factor), aircraft, links; `include_raw=true` for the untouched record |
| GET  | `/api/v1/cases/{ntsb_number}/report` | Streams the NTSB-generated report PDF |
| GET  | `/api/v1/cases/{ntsb_number}/docket` | Lists public docket documents with download URLs |
| GET  | `/api/v1/stats` | Aggregates for a filter: by year, make, injury, FAR part, state, top findings |
| GET  | `/api/v1/export` | Bulk download: `fmt=csv` / `json` (normalized) or `ntsb-csv` (CAROL's own summary) |
| POST | `/api/v1/documents/from-case/{n}` · `/upload` · `/from-url` | Load a report for analysis (see above) |
| GET/POST | `/api/v1/documents/{id}/summary` · `/search` · `/ask` · `/insights` · `/pages/{n}` | Report intelligence |
| GET  | `/api/v1/fields` | Every searchable CAROL column with operators and allowed values |
| POST | `/api/v1/debug/payload` | Shows the exact JSON that would be sent to NTSB |

Examples:

```bash
# Part 121 airline accidents on Boeing aircraft since 2020
curl "localhost:8000/api/v1/cases?make=Boeing&airline_only=true&event_type=ACC&date_from=2020-01-01"

# Any case whose findings mention thrust reversers on a 737
curl -X POST localhost:8000/api/v1/cases/query -H 'content-type: application/json' -d '{
  "rules": [{"column":"Finding.FindingText","operator":"contains","value":"thrust reverser"},
            {"column":"Aircraft.AircraftModel","operator":"contains","value":"737"}]}'

curl localhost:8000/api/v1/cases/DCA24MA063
curl -o report.pdf localhost:8000/api/v1/cases/DCA24MA063/report
```

## Generating client SDKs

`python scripts/export_openapi.py` writes `openapi.json`. Generate a typed client in any language:

```bash
npx @openapitools/openapi-generator-cli generate -i openapi.json -g python     -o sdk/python
npx @openapitools/openapi-generator-cli generate -i openapi.json -g typescript-fetch -o sdk/ts
```

## CLI

```bash
ntsb search --make Boeing --airline-only --from 2020-01-01 --limit 10
ntsb case DCA24MA063
ntsb report DCA24MA063 -o report.pdf
ntsb docket DCA24MA063 --download ./docket
ntsb export --finding "thrust reverser" -o reversers.json
```

## Python

```python
from ntsb_api import NTSBClient, CaseFilters

async with NTSBClient() as c:
    page = await c.search(CaseFilters(make="Airbus", airline_only=True).to_rules(), limit=50)
    case = await c.get_case(page["items"][0]["ntsb_number"])
    print(case["probable_cause"], case["findings"])
```

## MCP server (for agents)

`python -m ntsb_api.mcp_server` exposes `search_accidents`, `get_accident` and
`list_docket_documents` as tools for Claude Desktop, LangGraph, CrewAI or any MCP client.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `NTSB_MOCK` | `0` | Serve synthetic data, never call NTSB |
| `NTSB_CACHE_TTL` | `900` | Seconds to cache upstream responses |
| `NTSB_MIN_INTERVAL` | `0.5` | Minimum seconds between upstream calls (be polite) |
| `NTSB_TIMEOUT` / `NTSB_MAX_RETRIES` | `60` / `3` | Upstream timeout and retries |
| `NTSB_AIRLINE_REG` | `Part 121: Air carrier` | CAROL value used by `airline_only` |
| `NTSB_CORS` | `*` | Allowed browser origins |
| `NTSB_CAROL_BASE`, `NTSB_REPORT_BASE`, `NTSB_DOCKET_BASE` | data.ntsb.gov URLs | Override endpoints |

## How it works and what to know

* **Source.** CAROL's public JSON query API is the same one the data.ntsb.gov search UI calls. It is
  unauthenticated but undocumented, so NTSB can change it. The client keeps all endpoint and
  payload knowledge in `query.py` / `config.py`, and the normalizer resolves each field from several
  candidate names and keeps the untouched record under `raw`, so a rename upstream degrades
  gracefully instead of breaking.
* **Two read paths.** `Query/Main` gives paged results plus a total count (search lists).
  `Query/FileExport` returns a ZIP of full JSON (narratives, findings, vehicles) and is used for case
  detail, stats and exports.
* **Dockets.** NTSB publishes no docket JSON API, so `docket.py` parses the docket page's document
  links. If NTSB renders a docket client-side, the endpoint returns the docket URL with a note.
* **Official API.** NTSB also runs a keyed developer portal (developer.ntsb.gov) and is beta-testing a
  new CAROL with its own API. To move over, add a backend in `client.py`; the REST surface stays the same.
* **Airline filter.** `airline_only` uses the CAROL value `Part 121: Air carrier`. If NTSB changes
  the label, set `NTSB_AIRLINE_REG`. Use `GET /api/v1/fields` to see current columns.
* **Etiquette.** Requests are paced, retried with backoff and cached. Keep bulk pulls reasonable.
* **Mock data** (`NTSB_MOCK=1`, the UI's demo mode) is synthetic. Every case number starts with `DEMO`.
