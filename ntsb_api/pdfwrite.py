"""Minimal dependency-free PDF writer (multi-page, wrapped Helvetica text).

Used only to produce synthetic sample reports in mock mode and tests.
"""
from __future__ import annotations

import textwrap


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1", "replace").decode("latin-1")


def text_pdf(blocks: list[tuple[str, str]], title: str = "") -> bytes:
    """blocks = [(kind, text)] where kind is 'h1', 'h2' or 'p'. Returns PDF bytes."""
    lines: list[tuple[str, str]] = []
    if title:
        lines += [("h1", title), ("p", "")]
    for kind, text in blocks:
        if kind == "p":
            for para in text.split("\n"):
                lines += [("p", w) for w in (textwrap.wrap(para, 92) or [""])]
            lines.append(("p", ""))
        else:
            lines += [(kind, text), ("p", "")]

    pages: list[list[tuple[str, str]]] = [[]]
    used = 0
    for kind, text in lines:
        h = {"h1": 26, "h2": 20, "p": 14}[kind]
        if used + h > 680:
            pages.append([]); used = 0
        pages[-1].append((kind, text)); used += h

    objs: list[bytes] = []
    def add(b: bytes) -> int:
        objs.append(b); return len(objs)

    cat = add(b"")  # placeholder
    pages_id = add(b"")
    f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    kids = []
    for n, pg in enumerate(pages, 1):
        y, ops = 740, []
        for kind, text in pg:
            size, font = {"h1": (16, "F2"), "h2": (12, "F2"), "p": (10, "F1")}[kind]
            if text:
                ops.append(f"BT /{font} {size} Tf 56 {y} Td ({_esc(text)}) Tj ET")
            y -= {"h1": 26, "h2": 20, "p": 14}[kind]
        ops.append(f"BT /F1 8 Tf 520 30 Td (Page {n} of {len(pages)}) Tj ET")
        stream = "\n".join(ops).encode("latin-1", "replace")
        c = add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        kids.append(add(f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] /Contents {c} 0 R "
                        f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> >> >>".encode()))
    objs[cat - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    objs[pages_id - 1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()

    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    out += f"trailer << /Size {len(objs) + 1} /Root {cat} 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)
