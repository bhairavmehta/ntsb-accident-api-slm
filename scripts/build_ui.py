"""Build the web UI: inlines synthetic demo data into the template.

Writes ntsb_api/static/index.html (served at / by the API) and dist/explorer-preview.html
(a standalone preview fragment that runs entirely on the synthetic sample data).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ntsb_api.mock import MOCK_CASES  # noqa: E402
from ntsb_api.normalize import normalize_case  # noqa: E402

tpl = (ROOT / "scripts/ui_template.html").read_text()
demo = [normalize_case(c, include_raw=False) for c in MOCK_CASES]
for d in demo:
    d.pop("links", None)
from ntsb_api.documents import build_doc, pdf_pages  # noqa: E402
from ntsb_api.insights import build_insights  # noqa: E402
from ntsb_api import mock  # noqa: E402
from ntsb_api.slm import ReportAnalyst  # noqa: E402

docs = {}
an = ReportAnalyst()
for raw, case in zip(MOCK_CASES, demo):
    num = case["ntsb_number"]
    d = build_doc(num, f"NTSB report {num} (synthetic sample)", pdf_pages(mock.mock_report_pdf(raw)))
    ins = build_insights(d, case)
    ins.pop("doc_id", None)
    docs[num] = {"title": d.title, "pages": d.pages, "words": len(d.text.split()), "sections": d.outline(),
                 "chunks": [{"p": c.page, "s": c.section, "t": c.text} for c in d.chunks],
                 "summary": an._extractive_summary(d), "insights": ins}


def _js(o):
    return json.dumps(o, separators=(",", ":")).replace("</", "<\\/")


body = tpl.replace("__DEMO_DATA__", _js(demo)).replace("__DEMO_DOCS__", _js(docs))
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
(ROOT / "ntsb_api/static/index.html").write_text(page)
(ROOT / "dist").mkdir(exist_ok=True)
(ROOT / "dist/explorer-preview.html").write_text(body)
print("built", len(page), "bytes")
