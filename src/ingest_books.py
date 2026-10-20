"""
ingest_books.py  —  Ingest a folder of PDF textbooks into the RAG.

Reads every .pdf in  books/  and adds it to the "medical_books" collection in
the SAME ChromaDB that ask.py reads. This is the collection ask.py needs for
its TEXTBOOK FACTS; if it's missing you get:
    chromadb.errors.NotFoundError: Collection [medical_books] does not exist

Companion to ingest_interview.py (which handles the SEPARATE
"patient_interviews" collection). Kept separate so textbook facts and patient
case precedent never blur together — exactly the split ask.py relies on.

Citations: ask.py cites every fact as  [Source: {book}, page {page}].  So each
chunk is tagged with:
    book  = the PDF's filename (e.g. "Kaplan_Sadock.pdf")
    page  = the 1-based PDF page the chunk came from
Chunks never span across pages, so a citation always points to one real page.

PATH SAFETY: DB_DIR and BOOKS_DIR are anchored to THIS FILE's location
(the PyFiles folder), not the directory you happen to run from. That avoids the
classic "two vectordb folders" bug. The canonical DB lives at PyFiles\\vectordb
and the PDFs at PyFiles\\books — ask.py, analyze_case.py and ingest_interview.py
all anchor to the SAME PyFiles\\vectordb, so every script reads one database.

SETUP (Windows, inside your venv):
    pip install -U pypdf chromadb ollama
    (ollama must be running, with the nomic-embed-text model pulled:
        ollama pull nomic-embed-text )

USAGE:
    python ingest_books.py                 ingest every PDF in books/
    python ingest_books.py --rebuild       wipe medical_books first, then ingest
    python ingest_books.py --delete Kaplan_Sadock.pdf   remove one book's chunks
"""

import os
import re
import argparse
import chromadb
import ollama
from pypdf import PdfReader

# ---------------- Settings (mirror ask.py) ----------------
HERE          = os.path.dirname(os.path.abspath(__file__))
# Anchored paths: same DB no matter where you launch from.
DB_DIR        = os.path.join(HERE, "vectordb")
BOOKS_DIR     = os.path.join(HERE, "books")
COLLECTION    = "medical_books"        # the collection ask.py reads for FACTS
EMBED_MODEL   = "nomic-embed-text"
CHUNK_CHARS   = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH   = 64                     # add to Chroma in batches, not all at once
# ----------------------------------------------------------


def clean_text(text):
    """Light cleanup of extracted PDF text: collapse runaway whitespace and
    drop lone hyphen line-breaks (word-\\nwrap -> wordwrap). Conservative on
    purpose — we don't want to mangle drug names or doses."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    # join hyphenated line breaks: "treat-\nment" -> "treatment"
    text = re.sub(r"-\n(?=\w)", "", text)
    # collapse 3+ newlines to a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse runs of spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Greedy paragraph-aware chunking with character overlap. Operates on the
    text of a SINGLE page so every chunk maps to one citable page."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = (cur[-overlap:] if overlap < len(cur) else cur)
        cur += p + "\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def get_collection(client):
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def delete_book(client, book_name):
    get_collection(client).delete(where={"book": book_name})
    print(f"[*] Deleted all chunks for book={book_name!r}.")


def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def ingest_one(col, path):
    """Ingest a single PDF. Returns (status, n_chunks). status in
    {"ok","empty_no_text"}."""
    book_name = os.path.basename(path)
    reader = PdfReader(path)

    # idempotent: drop any prior version of THIS book before re-adding
    col.delete(where={"book": book_name})

    ids, embs, docs, metas = [], [], [], []
    n_chunks = 0
    pages_with_text = 0

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if len(text) < 20:          # near-empty page (or scanned/no text layer)
            continue
        pages_with_text += 1

        for ci, ch in enumerate(chunk_text(text)):
            ids.append(f"{book_name}__p{page_num}__c{ci}")
            embs.append(embed(ch))
            docs.append(ch)
            metas.append({
                "book": book_name,
                "page": page_num,
                "source": "textbook",
                "chunk": ci,
            })
            n_chunks += 1

            # flush in batches so a huge book doesn't build one giant payload
            if len(ids) >= EMBED_BATCH:
                col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
                ids, embs, docs, metas = [], [], [], []

    if ids:
        col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)

    if pages_with_text == 0:
        # No extractable text on any page => almost certainly a scanned PDF
        # (images only, no text layer). pypdf can't read those; you'd need OCR.
        return ("empty_no_text", 0)
    return ("ok", n_chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the whole medical_books collection first, then ingest")
    ap.add_argument("--delete", metavar="BOOK.pdf",
                    help="delete all chunks for one book (by filename), then exit")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=DB_DIR)
    print(f"[*] Database: {DB_DIR}")

    if args.delete:
        delete_book(client, args.delete)
        return

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION)
            print(f"[*] Rebuilt: dropped existing '{COLLECTION}' collection.")
        except Exception:
            pass  # didn't exist yet — fine

    col = get_collection(client)

    if not os.path.isdir(BOOKS_DIR):
        print(f"[!] No '{BOOKS_DIR}' folder. Create it and put your book PDFs in it.")
        return
    pdfs = sorted(f for f in os.listdir(BOOKS_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        print(f"[!] No .pdf files found in {BOOKS_DIR}")
        return

    print(f"[*] Ingesting {len(pdfs)} PDF(s) into '{COLLECTION}'...\n")
    total_chunks = 0
    scanned = []
    for i, fname in enumerate(pdfs, 1):
        path = os.path.join(BOOKS_DIR, fname)
        print(f"[{i}/{len(pdfs)}] {fname} ... ", end="", flush=True)
        try:
            status, n = ingest_one(col, path)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        if status == "ok":
            print(f"{n} chunk(s)")
            total_chunks += n
        else:
            print("NO TEXT LAYER — looks scanned, skipped (needs OCR)")
            scanned.append(fname)

    final = col.count()
    print(f"\n[*] Done. Added {total_chunks} new chunk(s). "
          f"Collection '{COLLECTION}' now holds {final} chunk(s) total.")
    if scanned:
        print("\n[!] These PDFs had no extractable text (scanned image-only):")
        for s in scanned:
            print(f"      - {s}")
        print("    They were skipped. To use them you'd need to OCR them first")
        print("    (e.g. with the same Sarvam OCR flow, or ocrmypdf).")
    print('\n    Test with:  python ask.py "your clinical question"')


if __name__ == "__main__":
    main()
