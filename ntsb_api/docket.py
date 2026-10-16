"""Parse an NTSB public docket page into a list of downloadable documents.

The docket web app (``data.ntsb.gov/Docket/?NTSBNumber=...``) lists every document
filed for an investigation (factual reports, interview transcripts, photos, CVR
transcripts...). Each row links to a ``Document/docBLOB?ID=...`` download. NTSB
does not publish a documented JSON API for dockets, so we parse the HTML table
defensively: any anchor that looks like a document download is captured with the
text of its table row as the title.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

DOC_HREF = re.compile(r"(docBLOB|/Document/|\.pdf$|FileExtension=)", re.I)


class _DocketParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []
        self._row_text: list[str] = []
        self._row_links: list[tuple[str, str]] = []
        self._in_row = False
        self._href: str | None = None
        self._a_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._in_row, self._row_text, self._row_links = True, [], []
        elif tag == "a" and a.get("href") and DOC_HREF.search(a["href"]):
            self._href, self._a_text = a["href"], []

    def handle_endtag(self, tag):
        if tag == "a" and self._href:
            text = " ".join("".join(self._a_text).split())
            if self._in_row:
                self._row_links.append((self._href, text))
            else:
                self.rows.append({"cells": [text], "links": [(self._href, text)]})
            self._href = None
        elif tag == "tr" and self._in_row:
            if self._row_links:
                cells = [c for c in (" ".join(t.split()) for t in self._row_text) if c]
                self.rows.append({"cells": cells, "links": self._row_links})
            self._in_row = False

    def handle_data(self, data):
        if self._href is not None:
            self._a_text.append(data)
        if self._in_row:
            self._row_text.append(data)


def parse_docket_html(html: str, base_url: str) -> list[dict[str, Any]]:
    p = _DocketParser()
    p.feed(html)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in p.rows:
        for href, text in row["links"]:
            url = urljoin(base_url, href)
            if url in seen:
                continue
            seen.add(url)
            ext = re.search(r"FileExtension=\.?(\w+)", href, re.I) or re.search(r"\.(\w{2,4})$", href)
            pages = next((c for c in reversed(row["cells"]) if re.fullmatch(r"\d{1,4}", c)), None)
            title = text or next((c for c in row["cells"] if len(c) > 6), "Document")
            items.append({
                "title": title,
                "url": url,
                "file_type": ext.group(1).upper() if ext else None,
                "pages": int(pages) if pages else None,
                "row": row["cells"],
            })
    return items
