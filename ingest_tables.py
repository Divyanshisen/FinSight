"""S3: PDF tables -> searchable chunks (PyMuPDF find_tables). No LLM or vision calls."""
import glob
import os

import pymupdf as fitz  # PyMuPDF

from retriever import get_collection, get_embedder

ROWS_PER_CHUNK = 8     # chhote chunks, taaki embedding model table ko kaate nahi
MIN_ROWS = 3
MIN_COLS = 2


def clean(cell) -> str:
    return " ".join(str(cell or "").split())


def table_chunks(rows) -> list[str]:
    rows = [[clean(c) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < MIN_ROWS or max(len(r) for r in rows) < MIN_COLS:
        return []
    header, body = rows[0], rows[1:]
    chunks = []
    for i in range(0, len(body), ROWS_PER_CHUNK):
        lines = [" | ".join(header)] + [" | ".join(r) for r in body[i:i + ROWS_PER_CHUNK]]
        chunks.append("\n".join(lines))          # har chunk me header repeat hota hai
    return chunks


def main():
    col = get_collection()
    embedder = get_embedder()
    pdfs = glob.glob(os.path.join("data", "**", "*.pdf"), recursive=True)
    print(f"PDFs found: {len(pdfs)} | chunks before: {col.count()}")

    added = 0
    for path in pdfs:
        source = os.path.basename(path)
        if source == "ITR-2-2026-Eng.pdf":
            continue
        ids, docs, metas = [], [], []
        try:
            pdf = fitz.open(path)
        except Exception as e:
            print(f"  skip {source}: {e}")
            continue
        for pno, page in enumerate(pdf, 1):
            try:
                tables = page.find_tables().tables
            except Exception:
                continue
            for ti, t in enumerate(tables):
                for ci, text in enumerate(table_chunks(t.extract())):
                    ids.append(f"tbl-{source}-p{pno}-t{ti}-c{ci}")
                    docs.append(f"[Table from {source}, page {pno}]\n{text}")
                    metas.append({"source": source, "page": pno, "kind": "table"})
        if not docs:
            continue
        vectors = embedder.encode(docs, batch_size=32).tolist()
        for i in range(0, len(docs), 200):
            col.upsert(ids=ids[i:i + 200], documents=docs[i:i + 200],
                       embeddings=vectors[i:i + 200], metadatas=metas[i:i + 200])
        added += len(docs)
        print(f"  {source}: {len(docs)} table chunks")

    print(f"Done. Table chunks added: {added} | chunks now: {col.count()}")


if __name__ == "__main__":
    main()