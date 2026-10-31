"""Reads every PDF in data/ (including subfolders): text chunks + chart/graph descriptions -> Chroma DB."""
import glob
import os

import chromadb

import config
import vision
from retriever import get_embedder

try:
    import pymupdf as fitz          # newer PyMuPDF
except ImportError:
    import fitz                     # older PyMuPDF


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if len(piece) > 50:
            chunks.append(piece)
        start += size - overlap
    return chunks


def main():
    pdfs = glob.glob(os.path.join(config.DATA_DIR, "**", "*.pdf"), recursive=True)
    if not pdfs:
        print(f"No PDF found in '{config.DATA_DIR}/'. Please add PDFs first.")
        return

    client = chromadb.PersistentClient(path=config.DB_DIR)
    try:
        client.delete_collection(config.COLLECTION)      # fresh start
    except Exception:
        pass
    col = client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    embedder = get_embedder()

    ids, texts, metas = [], [], []
    total_text = total_visual = 0
    vision_on = config.USE_VISION

    for path in sorted(pdfs):
        name = os.path.basename(path)
        folder = os.path.relpath(os.path.dirname(path), config.DATA_DIR)
        category = "general" if folder == "." else folder    # folder name = category
        try:
            doc = fitz.open(path)
        except Exception as e:
            print(f"Skipped (unreadable): {name} -> {e}")
            continue

        # ---- 1) normal text chunks ----
        page_texts = [page.get_text() for page in doc]
        n_text = 0
        for pno, page_text in enumerate(page_texts, start=1):
            for i, chunk in enumerate(chunk_text(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
                ids.append(f"{category}-{name}-{pno}-t{i}")
                texts.append(chunk)
                metas.append({"source": name, "page": pno, "category": category, "kind": "text", "image": ""})
                n_text += 1

        # ---- 2) chart / graph / image-only pages -> vision model ----
        n_visual = 0
        if vision_on:
            candidates = vision.select_visual_pages(doc, [len(t) for t in page_texts])
            for idx in candidates:
                pno = idx + 1
                try:
                    result = vision.describe_page(doc[idx])
                except vision.VisionQuotaExceeded:
                    print("   Daily vision limit reached. Skipping charts for the rest of this run.")
                    print("   Run ingest again later: pages already read are cached and cost nothing.")
                    vision_on = False
                    break
                except Exception as e:
                    print(f"   vision failed on {name} p.{pno}: {e}")
                    continue
                if result is None:
                    continue
                desc_chunks = chunk_text(result["description"], int(config.CHUNK_SIZE * 1.5), config.CHUNK_OVERLAP)
                for i, chunk in enumerate(desc_chunks):
                    ids.append(f"{category}-{name}-{pno}-v{i}")
                    texts.append(f"[Chart/visual on page {pno}] {chunk}")
                    metas.append({"source": name, "page": pno, "category": category,
                                  "kind": "visual", "image": result["image"]})
                    n_visual += 1
                print(f"   vision OK: {name} p.{pno}")

        total_text += n_text
        total_visual += n_visual
        print(f"Read: [{category}] {name} ({len(doc)} pages, {n_text} text chunks, {n_visual} visual chunks)")
        if n_text == 0 and n_visual == 0:
            print("   ^ Nothing readable found (scanned PDF with no detectable visuals?)")

    if not texts:
        print("Nothing to store.")
        return
    for s in range(0, len(texts), 64):
        batch = slice(s, s + 64)
        col.add(ids=ids[batch], documents=texts[batch], metadatas=metas[batch],
                embeddings=embedder.encode(texts[batch]).tolist())
    print(f"Done. Stored {len(texts)} chunks ({total_text} text + {total_visual} visual).")


if __name__ == "__main__":
    main()