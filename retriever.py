"""Retrieval with two modes (config.RETRIEVAL_MODE):
  dense  : question -> embedding -> Chroma top-k (the original retriever)
  hybrid : paper 2 style. Multi-HyDE (several different queries, each with a hypothetical passage,
           searched in the dense index) + BM25 keyword search -> merge -> rerank with the
           cross-encoder -> top-k
"""
import hashlib
import json
import os
import re

import chromadb

import config
from evaluator import score_pairs
from llm import chat

_embedder = None
_bm25 = None
_bm25_state = {"count": -1, "ids": [], "docs": [], "metas": []}
LAST_TRACE = {} 
INDEX_REF = re.compile(r"\b(?:Chart|Table|Box|Figure|Fig\.)\s*:?\s*[IVX]*\.?\d+(?:\.\d+)?", re.I)


def _is_index_chunk(text: str) -> bool:
    """Contents pages like 'List of Charts ... Chart I.21 ... Chart I.22 ...' carry no facts."""
    if text.lstrip().startswith(("[Table from", "[Chart/visual")):
        return False
    return len(INDEX_REF.findall(text)) >= 5


def _drop_noise(docs: list[dict]) -> list[dict]:
    """Remove contents-page chunks and clearly irrelevant ones, but never return fewer than 3."""
    min_score = getattr(config, "MIN_RERANK", 0.02)
    candidates = [d for d in docs if not _is_index_chunk(d["text"])]
    strong = [d for d in candidates if d["rerank_score"] >= min_score or "(table)" in d["source"]]
    return strong if len(strong) >= 3 else (candidates or docs)[:3]         # details of the last retrieval (queries, candidate counts), for the UI trace tab

HYDE_PROMPT = """You help a finance search system. For the question below, write {n} different search queries.
The queries must NOT be paraphrases of each other. Each one should ask for a different piece of information
that a single finance document could answer (for example: the definition, the exact figure or table row,
the reason or context, or one part of the question when it has several parts).
For each query also write a short hypothetical passage (2-3 sentences, in the style of an annual report,
a regulator circular or an investor guide) that could answer it. Exact numbers in the passage do not matter.
Return ONLY a JSON list like [{{"query": "...", "passage": "..."}}]

Question: {question}"""


# ---------------------------------------------------------------- basics
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def get_collection():
    client = chromadb.PersistentClient(path=config.DB_DIR)
    return client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})


def _label(meta: dict) -> str:
    label = f'{meta["source"]} p.{meta["page"]}'
    if meta.get("kind") == "visual":
        label += " (chart/visual)"
    elif meta.get("kind") == "table":
        label += " (table)"
    return label


def _doc(chunk_id: str, text: str, meta: dict, via: str) -> dict:
    return {"id": chunk_id, "text": text, "source": _label(meta),
            "image": meta.get("image", ""), "via": via}


# ---------------------------------------------------------------- dense search
def _dense_search(embedding, k: int) -> list[dict]:
    res = get_collection().query(query_embeddings=[embedding], n_results=k)
    return [_doc(i, t, m, "dense")
            for i, t, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])]


# ---------------------------------------------------------------- BM25 keyword search
TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())          # keeps numbers such as 5.50 or 1,25,000 as one token


def _get_bm25():
    """Builds the BM25 index from the Chroma collection (rebuilt when the chunk count changes)."""
    global _bm25
    from rank_bm25 import BM25Okapi
    col = get_collection()
    count = col.count()
    if _bm25 is None or _bm25_state["count"] != count:
        data = col.get(include=["documents", "metadatas"])
        _bm25_state.update(count=count, ids=data["ids"], docs=data["documents"], metas=data["metadatas"])
        _bm25 = BM25Okapi([_tokens(d) or ["_"] for d in data["documents"]])
    return _bm25


def _bm25_search(query_text: str, k: int) -> list[dict]:
    tokens = _tokens(query_text)
    if not tokens:
        return []
    bm25 = _get_bm25()
    scores = bm25.get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    s = _bm25_state
    return [_doc(s["ids"][i], s["docs"][i], s["metas"][i], "bm25") for i in order if scores[i] > 0]


# ---------------------------------------------------------------- Multi-HyDE
def _parse_hyde(text: str, question: str):
    """Returns (items, ok). items = [{'query', 'passage'}]. Falls back to the plain question."""
    try:
        m = re.search(r"\[.*\]", text, re.S)
        raw = json.loads(m.group(0)) if m else []
        items = []
        for it in raw:
            if isinstance(it, dict) and str(it.get("query", "")).strip():
                query = str(it["query"]).strip()
                items.append({"query": query, "passage": str(it.get("passage") or query).strip()})
        if items:
            return items[:config.HYDE_QUERIES], True
    except Exception:
        pass
    return [{"query": question, "passage": question}], False


def _load_cache() -> dict:
    try:
        with open(config.HYDE_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def multi_hyde(question: str) -> list[dict]:
    """Several different queries + a hypothetical passage for each (cached on disk).
    The passages are only used to build search embeddings. They are never shown as evidence."""
    key = hashlib.sha1(f"{config.HYDE_QUERIES}|{question}".encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]
    reply = chat(HYDE_PROMPT.format(n=config.HYDE_QUERIES, question=question),
                 model=getattr(config, "HELPER_MODEL", None))
    items, ok = _parse_hyde(reply, question)
    if ok:                                       # never cache a failed attempt
        cache[key] = items
        with open(config.HYDE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
    return items


# ---------------------------------------------------------------- main entry
def retrieve(question: str, k: int = config.TOP_K, mode: str | None = None) -> list[dict]:
    mode = mode or config.RETRIEVAL_MODE
    embedder = get_embedder()

    if mode == "dense":
        LAST_TRACE.clear()
        LAST_TRACE.update(mode="dense")
        return _dense_search(embedder.encode([question])[0].tolist(), k)

    try:
        items = multi_hyde(question)
    except Exception as e:                       # e.g. rate limit: fall back to the plain question
        print(f"   Multi-HyDE failed, using the plain question: {str(e)[:120]}")
        items = [{"query": question, "passage": question}]

    candidates: dict[str, dict] = {}

    def add(docs: list[dict]):
        for d in docs:
            if d["id"] in candidates:
                ways = set(candidates[d["id"]]["via"].split("+")) | {d["via"]}
                candidates[d["id"]]["via"] = "+".join(sorted(ways))
            else:
                candidates[d["id"]] = d

    vectors = embedder.encode([question] + [it["passage"] for it in items])
    for vec in vectors:                          # the plain question is kept as a safety net
        add(_dense_search(vec.tolist(), config.DENSE_PER_QUERY))
    add(_bm25_search(question + " " + " ".join(it["query"] for it in items), config.BM25_TOP_K))

    docs = list(candidates.values())
    scores = score_pairs(question, [d["text"] for d in docs])      # rerank with the cross-encoder
    for d, s in zip(docs, scores):
        d["rerank_score"] = round(s, 4)
    docs.sort(key=lambda d: d["rerank_score"], reverse=True)

    LAST_TRACE.clear()
    LAST_TRACE.update(mode="hybrid", queries=[it["query"] for it in items], candidates=len(docs),
                      from_dense=sum("dense" in d["via"] for d in docs),
                      from_bm25=sum("bm25" in d["via"] for d in docs))
    return _drop_noise(docs)[:k]