"""
ingest_interview.py (v2, BATCH) — Ingest a WHOLE folder of reviewed summaries into the RAG.

Reads every corrected summary in  summaries/  and adds it to the SEPARATE
"patient_interviews" collection in the same ChromaDB that ask.py uses.

Kept separate from "medical_books" so:
  - textbook answers aren't polluted by case chunks
  - a patient's data can be deleted cleanly (DPDPA right to erasure)
  - case chunks carry metadata (patient_id, date, source=interview)

SAFETY: by default this SKIPS any summary that still contains the tag
"[AI-INFERRED - CONFIRM OR REWRITE]", because that means the doctor has not yet
confirmed the reasoning. Use --force to ingest anyway (not recommended).

Workflow:
    python summarize_interview.py     -> summaries/*_summary.txt
    [doctor reviews & corrects every file, removing the [AI-INFERRED] tags]
    python ingest_interview.py        -> ingests all reviewed summaries  (THIS)

Usage:
    python ingest_interview.py                 ingest all reviewed summaries
    python ingest_interview.py --force         ingest even unreviewed ones
    python ingest_interview.py --delete Patient_042   delete one patient's data
"""

import os
import re
import argparse
import datetime
import chromadb
import ollama

# ---------------- Settings (mirror ask.py) ----------------
HERE          = os.path.dirname(os.path.abspath(__file__))
ROOT          = os.path.dirname(HERE)
DB_DIR        = os.path.join(HERE, "vectordb")   # same DB as ask.py / ingest_books.py
COLLECTION    = "patient_interviews"   # SEPARATE from "medical_books"
EMBED_MODEL   = "nomic-embed-text"
SUMMARY_DIR   = os.path.join(ROOT, "summaries")
CHUNK_CHARS   = 900
CHUNK_OVERLAP = 150
UNREVIEWED_TAG = "[AI-INFERRED - CONFIRM OR REWRITE]"
# ----------------------------------------------------------


def read_summary_body(path):
    """Return (body_text, patient_id, date) from a summary file, dropping the header."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    patient_id, date_str = None, None
    for line in text.splitlines():
        if line.startswith("Patient ID:"):
            patient_id = line.split(":", 1)[1].strip()
        elif line.startswith("Date:"):
            date_str = line.split(":", 1)[1].strip()
    # body = everything after the header separator (a run of 40+ '=').
    # NOTE: analyze_case.py writes that line as '='*55; do NOT hard-code a count
    # here (an exact "="*58 split silently fails and embeds the header into the
    # RAG). Match any 40+ run so the two stay in sync no matter what.
    parts = re.split(r"={40,}", text, maxsplit=1)
    body = parts[-1].strip() if len(parts) > 1 else text
    return body, patient_id, date_str


def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap < len(cur) else cur
        cur += p + "\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def get_collection(client):
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def delete_patient(client, patient_id):
    get_collection(client).delete(where={"patient_id": patient_id})
    print(f"[*] Deleted all chunks for patient_id={patient_id}.")


def ingest_one(client, path, force):
    body, patient_id, date_str = read_summary_body(path)
    patient_id = patient_id or os.path.splitext(os.path.basename(path))[0].replace("_summary", "")
    date_str = date_str or datetime.date.today().isoformat()

    if UNREVIEWED_TAG in body and not force:
        return ("skipped_unreviewed", patient_id, 0)

    chunks = chunk_text(body)
    if not chunks:
        return ("empty", patient_id, 0)

    col = get_collection(client)
    # idempotent: drop any prior version of this patient+file first
    col.delete(where={"$and": [{"patient_id": patient_id},
                               {"source_file": os.path.basename(path)}]})

    ids, embs, docs, metas = [], [], [], []
    for i, ch in enumerate(chunks):
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=ch)["embedding"]
        ids.append(f"{patient_id}__{os.path.basename(path)}__{i}")
        embs.append(emb)
        docs.append(ch)
        metas.append({
            "patient_id": patient_id,
            "date": date_str,
            "source": "interview",
            "source_file": os.path.basename(path),
            "chunk": i,
            "reviewed": UNREVIEWED_TAG not in body,
        })
    col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    return ("ok", patient_id, len(chunks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="ingest even summaries still containing [AI-INFERRED] tags")
    ap.add_argument("--delete", metavar="PATIENT_ID",
                    help="delete all chunks for a patient id, then exit")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=DB_DIR)

    if args.delete:
        delete_patient(client, args.delete)
        return

    if not os.path.isdir(SUMMARY_DIR):
        print(f"No '{SUMMARY_DIR}' folder. Run summarize_interview.py first.")
        return
    files = sorted(f for f in os.listdir(SUMMARY_DIR) if f.lower().endswith(".txt"))
    if not files:
        print(f"No summary .txt files in '{SUMMARY_DIR}'.")
        return

    print(f"[*] Ingesting {len(files)} summary file(s) into '{COLLECTION}'...\n")
    ok = skipped = empty = 0
    for i, fname in enumerate(files, 1):
        path = os.path.join(SUMMARY_DIR, fname)
        status, pid, n = ingest_one(client, path, args.force)
        if status == "ok":
            print(f"[{i}/{len(files)}] {fname} -> {n} chunk(s) [{pid}]")
            ok += 1
        elif status == "skipped_unreviewed":
            print(f"[{i}/{len(files)}] {fname} -> SKIPPED (still has [AI-INFERRED] tags; "
                  f"doctor must review). Use --force to override.")
            skipped += 1
        else:
            print(f"[{i}/{len(files)}] {fname} -> empty, skipped")
            empty += 1

    print(f"\n[*] Done. {ok} ingested, {skipped} skipped (unreviewed), {empty} empty.")
    if skipped:
        print("    Skipped files still need doctor review before ingestion.")
    print("\n    Test with:  python ask.py \"your clinical question\"")


if __name__ == "__main__":
    main()
