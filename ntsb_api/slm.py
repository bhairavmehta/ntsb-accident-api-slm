"""Small language model layer: summarize a report, answer questions with citations.

Backends (chosen with NTSB_SLM_BACKEND, default ``auto``):

* ``llamacpp``   – the model runs *inside* this process via llama-cpp-python and a GGUF file
                   (default: Qwen2.5-1.5B-Instruct Q4_K_M, ~1 GB, CPU-friendly). Truly embedded.
* ``ollama``     – a local Ollama daemon (``ollama pull qwen2.5:1.5b``).
* ``openai``     – any OpenAI-compatible server: llama.cpp server, vLLM, LM Studio, TGI.
* ``extractive`` – no model. Sentence ranking + retrieval. Always available as the fallback.

``auto`` picks llamacpp if a model file is present, else Ollama if reachable, else extractive.
All prompts are grounded: the model only sees passages from the report, tagged with page numbers,
and is told to cite them. Answers come back with the source passages so a reader can verify.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx

from .documents import ReportDoc, search, tokenize

DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_DIR = Path(os.getenv("NTSB_MODEL_DIR", Path.home() / ".cache" / "ntsb-api" / "models"))

SYSTEM = ("You are an aviation safety analyst summarizing NTSB investigation reports. "
          "Use only the report excerpts provided. Every excerpt is labelled with its page like [p.3]. "
          "Cite pages in square brackets after each claim. If the excerpts do not contain the answer, say so. "
          "Never invent times, numbers, names or causes.")


# ---------------------------------------------------------------------------- backends
class Backend:
    name = "base"
    model = None

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 600,
                       temperature: float = 0.2, json_mode: bool = False) -> str:
        out = []
        async for t in self.stream(messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode):
            out.append(t)
        return "".join(out)

    async def stream(self, messages, **kw) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError
        yield ""

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "generative": self.name != "extractive"}


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(self, model_path: str, n_ctx: int = 8192, threads: Optional[int] = None):
        from llama_cpp import Llama  # optional dependency
        self.model = Path(model_path).name
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=threads or os.cpu_count(),
                          verbose=False)
        self._lock = asyncio.Lock()  # llama.cpp contexts are not re-entrant

    async def stream(self, messages, max_tokens=600, temperature=0.2, json_mode=False):
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def run():
            try:
                kw = {"response_format": {"type": "json_object"}} if json_mode else {}
                for part in self._llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                             temperature=temperature, stream=True, **kw):
                    tok = part["choices"][0]["delta"].get("content")
                    if tok:
                        loop.call_soon_threadsafe(q.put_nowait, tok)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        async with self._lock:
            fut = loop.run_in_executor(None, run)
            while (tok := await q.get()) is not None:
                yield tok
            await fut


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str, host: str, transport: Any = None):
        self.model, self.host, self._transport = model, host.rstrip("/"), transport

    async def stream(self, messages, max_tokens=600, temperature=0.2, json_mode=False):
        body = {"model": self.model, "messages": messages, "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}}
        if json_mode:
            body["format"] = "json"
        async with httpx.AsyncClient(timeout=300, transport=self._transport) as c:
            async with c.stream("POST", f"{self.host}/api/chat", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.strip():
                        tok = json.loads(line).get("message", {}).get("content")
                        if tok:
                            yield tok


class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str = "none", transport: Any = None):
        self.model, self.base, self.key, self._transport = model, base_url.rstrip("/"), api_key, transport

    async def stream(self, messages, max_tokens=600, temperature=0.2, json_mode=False):
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": max_tokens, "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=300, transport=self._transport,
                                     headers={"Authorization": f"Bearer {self.key}"}) as c:
            async with c.stream("POST", f"{self.base}/chat/completions", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data:") and "[DONE]" not in line:
                        delta = json.loads(line[5:])["choices"][0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]


class ExtractiveBackend(Backend):
    """No model: the engine falls back to extractive methods (never called to generate)."""
    name = "extractive"
    model = "none (extractive)"

    async def stream(self, messages, **kw):
        yield ""


# ---------------------------------------------------------------------------- selection
def _model_file() -> Optional[Path]:
    p = os.getenv("NTSB_SLM_MODEL_PATH")
    if p and Path(p).exists():
        return Path(p)
    cand = MODEL_DIR / DEFAULT_FILE
    return cand if cand.exists() else None


def _ollama_up(host: str) -> bool:
    try:
        return httpx.get(f"{host}/api/tags", timeout=1.5).status_code == 200
    except Exception:
        return False


def make_backend() -> Backend:
    kind = os.getenv("NTSB_SLM_BACKEND", "auto").lower()
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model = os.getenv("NTSB_OLLAMA_MODEL", "qwen2.5:1.5b")
    if kind in ("llamacpp", "auto"):
        f = _model_file()
        if f:
            try:
                return LlamaCppBackend(str(f), n_ctx=int(os.getenv("NTSB_SLM_CTX", "8192")))
            except ImportError:
                if kind == "llamacpp":
                    raise RuntimeError("Install llama-cpp-python: pip install 'ntsb-accident-api[slm]'")
    if kind == "ollama" or (kind == "auto" and _ollama_up(ollama_host)):
        return OllamaBackend(ollama_model, ollama_host)
    if kind == "openai":
        return OpenAICompatBackend(os.getenv("NTSB_OPENAI_MODEL", "local-model"),
                                   os.getenv("NTSB_OPENAI_BASE", "http://127.0.0.1:8080/v1"),
                                   os.getenv("NTSB_OPENAI_KEY", "none"))
    return ExtractiveBackend()


def download_default_model() -> Path:
    """Fetch the default GGUF from Hugging Face into NTSB_MODEL_DIR."""
    from huggingface_hub import hf_hub_download
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=os.getenv("NTSB_SLM_REPO", DEFAULT_REPO),
                                filename=os.getenv("NTSB_SLM_FILE", DEFAULT_FILE), local_dir=MODEL_DIR))


# ---------------------------------------------------------------------------- embeddings
class Embedder:
    """Optional small embedding model (fastembed, ONNX, bge-small ~ 33M params)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.model_name = model
        self._m = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]):
        return [list(v) for v in self._m.embed(texts)]


def make_embedder() -> Optional[Embedder]:
    if os.getenv("NTSB_EMBEDDINGS", "auto").lower() in ("0", "off", "false", "none"):
        return None
    try:
        return Embedder(os.getenv("NTSB_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    except Exception:
        return None


# ---------------------------------------------------------------------------- extractive helpers
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text)
    return [s.strip() for s in _SENT.split(flat) if 25 <= len(s.strip()) <= 600]


def rank_sentences(text: str, k: int = 5, query: str = "") -> list[str]:
    """Frequency-centrality ranking (TextRank-lite) biased toward an optional query."""
    sents = sentences(text)
    if not sents:
        return []
    freq = Counter(t for s in sents for t in tokenize(s))
    q = set(tokenize(query))
    scored = []
    for i, s in enumerate(sents):
        toks = tokenize(s)
        if not toks:
            continue
        sc = sum(freq[t] for t in toks) / (len(toks) ** 0.6)
        sc *= 1 + 2.5 * len(q & set(toks)) if q else 1
        sc *= 1.25 if i < 3 else 1
        if "synthetic sample" in s.lower() or "illustrates the structure" in s.lower():
            sc *= 0.05
        scored.append((sc, i, s))
    top = sorted(scored, reverse=True)[:k]
    return [s for _, _, s in sorted(top, key=lambda x: x[1])]


def _page_of(doc: ReportDoc, snippet: str) -> Optional[int]:
    key = re.sub(r"\s+", " ", snippet[:60]).lower()
    for pno, page in enumerate(doc.pages, 1):
        if key in re.sub(r"\s+", " ", page).lower():
            return pno
    return None


def _cite(doc: ReportDoc, s: str) -> str:
    p = _page_of(doc, s)
    return f"{s} [p.{p}]" if p else s


# ---------------------------------------------------------------------------- engine
class ReportAnalyst:
    def __init__(self, backend: Optional[Backend] = None, embedder: Any = None):
        self.backend = backend or ExtractiveBackend()
        self.embedder = embedder
        self._summaries: dict[str, dict[str, Any]] = {}

    @property
    def generative(self) -> bool:
        return self.backend.name != "extractive"

    def status(self) -> dict[str, Any]:
        return {**self.backend.info(),
                "embeddings": getattr(self.embedder, "model_name", None),
                "search": "hybrid (BM25 + embeddings)" if self.embedder else "BM25 keyword"}

    # ---- context building ------------------------------------------------
    def _context(self, doc: ReportDoc, budget_chars: int = 9000) -> str:
        priority = ["Probable Cause and Findings", "Probable Cause", "Findings", "Executive Summary", "Analysis",
                    "History of Flight", "Conclusions", "Safety Recommendations", "Front matter",
                    "Preliminary Information", "Wreckage and Impact Information", "Meteorological Information and Flight Plan",
                    "Pilot Information", "Flight Crew Information", "Survival Aspects", "Tests and Research"]
        order = sorted(doc.sections, key=lambda s: priority.index(s["name"]) if s["name"] in priority else 99)
        parts, used = [], 0
        for s in order:
            if not s["text"]:
                continue
            body = s["text"]
            room = budget_chars - used
            if room <= 200:
                break
            body = body[:room]
            parts.append(f"## {s['name']} [p.{s['page']}]\n{body}")
            used += len(body) + len(s["name"]) + 12
        return "\n\n".join(parts)

    # ---- summary -----------------------------------------------------------
    async def summarize(self, doc: ReportDoc, refresh: bool = False) -> dict[str, Any]:
        if not refresh and doc.doc_id in self._summaries:
            return self._summaries[doc.doc_id]
        out = None
        if self.generative:
            try:
                out = await self._llm_summary(doc)
            except Exception as e:  # model failure -> extractive, but say so
                out = self._extractive_summary(doc)
                out["warning"] = f"Model summary failed ({type(e).__name__}); showing extractive summary."
        else:
            out = self._extractive_summary(doc)
        self._summaries[doc.doc_id] = out
        return out

    async def _llm_summary(self, doc: ReportDoc) -> dict[str, Any]:
        ctx = self._context(doc)
        schema = ('{"headline": str (<=20 words), "what_happened": str (2-4 sentences), '
                  '"probable_cause": str, "contributing_factors": [str], "key_facts": [str], '
                  '"safety_lessons": [str]}')
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Report excerpts:\n\n{ctx}\n\nReturn ONLY JSON matching {schema}. "
                                            "Keep page citations like [p.2] inside the strings."}]
        raw = await self.backend.complete(msgs, max_tokens=700, json_mode=True)
        data = _parse_json(raw)
        if not isinstance(data, dict) or not data.get("what_happened"):
            raise ValueError("model did not return the expected JSON")
        for k in ("contributing_factors", "key_facts", "safety_lessons"):
            v = data.get(k) or []
            data[k] = [str(x) for x in (v if isinstance(v, list) else [v])][:8]
        data.update(mode="generative", model=self.backend.model, backend=self.backend.name)
        return data

    def _extractive_summary(self, doc: ReportDoc) -> dict[str, Any]:
        cause = doc.section_text("Probable Cause and Findings", "Probable Cause")
        m = re.search(r"probable cause\(?s?\)? of this (?:accident|incident)[^:]*:\s*(.+?)(?:\n\n|$)", cause, re.I | re.S)
        cause_text = re.sub(r"\s+", " ", (m.group(1) if m else cause)).strip()[:700]
        analysis = doc.section_text("Analysis", "Executive Summary", "Conclusions")
        history = doc.section_text("History of Flight", "Preliminary Information") or doc.text
        what = rank_sentences(history, 3)
        facts = rank_sentences(doc.section_text("Pilot Information", "Flight Crew Information",
                                                "Aircraft and Owner/Operator Information",
                                                "Meteorological Information and Flight Plan",
                                                "Wreckage and Impact Information") or doc.text, 4)
        lessons = rank_sentences(analysis, 3)
        findings = [l.strip() for l in doc.section_text("Findings").splitlines() if l.strip()][:6]
        head = what[0] if what else (doc.title or "")
        head = " ".join(head.split()[:20])
        return {
            "headline": head,
            "what_happened": " ".join(_cite(doc, s) for s in what),
            "probable_cause": cause_text + (f" [p.{_page_of(doc, cause_text)}]" if cause_text and _page_of(doc, cause_text) else ""),
            "contributing_factors": findings,
            "key_facts": [_cite(doc, s) for s in facts],
            "safety_lessons": [_cite(doc, s) for s in lessons],
            "mode": "extractive", "model": None, "backend": "extractive",
            "note": "Extractive summary: sentences selected from the report. Configure a small language model for written summaries.",
        }

    # ---- question answering -------------------------------------------------
    def retrieve(self, doc: ReportDoc, question: str, k: int = 6) -> list[dict[str, Any]]:
        return search(doc, question, k=k, embedder=self.embedder)

    def _qa_messages(self, doc: ReportDoc, question: str, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
        src = "\n\n".join(f"[S{i}] (p.{h['page']}, {h['section']})\n{h['text']}" for i, h in enumerate(hits, 1))
        return [{"role": "system", "content": SYSTEM + " Cite sources as [S1], [S2] and pages as [p.N]."},
                {"role": "user", "content": f"Sources from report '{doc.title}':\n\n{src}\n\nQuestion: {question}\n"
                                            "Answer in 2-6 sentences using only the sources."}]

    async def ask(self, doc: ReportDoc, question: str, k: int = 6) -> dict[str, Any]:
        hits = self.retrieve(doc, question, k)
        if not hits:
            return {"answer": "The report does not appear to contain information about that.", "sources": [],
                    "mode": "generative" if self.generative else "extractive"}
        if self.generative:
            try:
                ans = await self.backend.complete(self._qa_messages(doc, question, hits), max_tokens=450)
                return {"answer": ans.strip(), "sources": hits, "mode": "generative", "model": self.backend.model}
            except Exception as e:
                fallback = self._extractive_answer(doc, question, hits)
                fallback["warning"] = f"Model unavailable ({type(e).__name__}); showing extractive answer."
                return fallback
        return self._extractive_answer(doc, question, hits)

    async def ask_stream(self, doc: ReportDoc, question: str, k: int = 6) -> AsyncIterator[dict[str, Any]]:
        hits = self.retrieve(doc, question, k)
        yield {"type": "sources", "sources": hits}
        if not hits:
            yield {"type": "token", "text": "The report does not appear to contain information about that."}
        elif self.generative:
            async for tok in self.backend.stream(self._qa_messages(doc, question, hits), max_tokens=450):
                yield {"type": "token", "text": tok}
        else:
            yield {"type": "token", "text": self._extractive_answer(doc, question, hits)["answer"]}
        yield {"type": "done", "mode": "generative" if self.generative else "extractive"}

    def _extractive_answer(self, doc: ReportDoc, question: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
        pool = " ".join(h["text"] for h in hits[:4])
        best = rank_sentences(pool, 3, query=question)
        answer = " ".join(f"{s} [p.{_page_of(doc, s) or hits[0]['page']}]" for s in best) or hits[0]["text"][:400]
        return {"answer": answer, "sources": hits, "mode": "extractive",
                "note": "Most relevant sentences from the report. Configure a small language model for written answers."}

    # ---- model-assisted timeline (when the rules find nothing) -------------
    async def llm_timeline(self, doc: ReportDoc) -> list[dict[str, Any]]:
        if not self.generative:
            return []
        hist = doc.section_text("History of Flight", "Preliminary Information", "Front matter")[:6000]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Excerpt:\n{hist}\n\nList the sequence of events as JSON "
                                            '{"events":[{"time":"HH:MM or null","event":str,"page":int}]}. Max 12.'}]
        try:
            data = _parse_json(await self.backend.complete(msgs, max_tokens=600, json_mode=True))
            return [e for e in (data or {}).get("events", []) if isinstance(e, dict) and e.get("event")][:12]
        except Exception:
            return []


def _parse_json(raw: str) -> Any:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else None
