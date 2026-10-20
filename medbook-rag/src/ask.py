"""
ask.py  (v6)  —  RAG over medical books + the expert's past-case corpus.

Purpose: help a future, less-experienced doctor by blending
  (a) TEXTBOOK FACTS (medical_books), and
  (b) how THIS expert handled SIMILAR PAST CASES (patient_interviews),
into one answer - facts cited to books, precedent cited to anonymised cases.

New in v6:
  * Always retrieves similar past cases from the corpus (not just one patient).
  * Answer has a "HOW THIS CLINIC HAS HANDLED SIMILAR CASES" section, drawn from
    the expert's confirmed reasoning - clearly separated from textbook facts.
  * --patient X still pulls one specific patient's own history if you want it.

Usage:
    python ask.py "young man, new psychosis after marriage, possession beliefs - approach?"
    python ask.py "follow-up plan for this patient" --patient Patient_042
"""

import os
import json
import argparse
import chromadb
import ollama
from sentence_transformers import CrossEncoder

# ---------------- Settings ----------------
# Anchor the DB to THIS script's folder (PyFiles), never the directory you
# happen to launch from. Every DB script does the same, so they all read/write
# the ONE database at PyFiles\vectordb — no more "two vectordb folders" bug.
HERE               = os.path.dirname(os.path.abspath(__file__))
DB_DIR             = os.path.join(HERE, "vectordb")
BOOK_COLLECTION    = "medical_books"
CASE_COLLECTION    = "patient_interviews"
EMBED_MODEL        = "nomic-embed-text"
LLM_MODEL          = "qwen3:4b"
RERANK_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES_PER_SUBQUERY = 8
FINAL_K            = 10          # textbook passages
CASE_K             = 5          # similar past cases
KEEP_ALIVE         = "30m"
MAX_ANSWER_TOKENS  = 2600       # raised: facts + treatment + precedent all fit
# ------------------------------------------

_reranker = CrossEncoder(RERANK_MODEL)


def split_question(question):
    prompt = (
        "Break this clinical question into 3-5 short, focused search queries covering "
        "the relevant angles. ALWAYS include one query specifically about DRUG TREATMENT / "
        "MEDICATIONS / DOSES whenever the question involves a condition that may be treated. "
        "Other useful angles: differential/causes, investigations, pathophysiology. "
        "Reply with ONLY a JSON array of strings, nothing else.\n\n"
        f"Question: {question}"
    )
    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False, keep_alive=KEEP_ALIVE, options={"temperature": 0},
        )
        text = resp["message"]["content"].strip()
        arr = text[text.find("["): text.rfind("]") + 1]
        subs = [q for q in json.loads(arr) if isinstance(q, str) and q.strip()]
        return subs or [question]
    except Exception:
        return [question]


def retrieve(collection, subqueries, per_query=CANDIDATES_PER_SUBQUERY, where=None):
    seen, candidates = set(), []
    for sq in subqueries:
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=sq)["embedding"]
        res = collection.query(query_embeddings=[emb], n_results=per_query, where=where)
        for doc, meta, cid in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
            if cid not in seen:
                seen.add(cid)
                candidates.append({"text": doc, "meta": meta})
    return candidates


def rerank(question, candidates, k):
    if not candidates:
        return []
    scores = _reranker.predict([[question, c["text"]] for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:k]]


def get_similar_cases(client, subqueries, question, patient_id=None):
    try:
        col = client.get_collection(CASE_COLLECTION)
    except Exception:
        return []
    where = {"patient_id": patient_id} if patient_id else None
    cands = retrieve(col, subqueries, per_query=6, where=where)
    return rerank(question, cands, CASE_K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--patient", help="restrict case retrieval to one patient_id")
    args = ap.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        print('Usage: python ask.py "your clinical question" [--patient Patient_042]')
        return

    client = chromadb.PersistentClient(path=DB_DIR)
    books = client.get_collection(BOOK_COLLECTION)

    subqueries = split_question(question)
    print("Sub-queries:", subqueries, "\n")

    # Textbook facts
    book_top = rerank(question, retrieve(books, subqueries), FINAL_K)
    context = ""
    for c in book_top:
        context += f"[Source: {c['meta']['book']}, page {c['meta']['page']}]\n{c['text']}\n\n"

    # Similar past cases (the expert's precedent)
    cases = get_similar_cases(client, subqueries, question, args.patient)
    case_block = ""
    for c in cases:
        m = c["meta"]
        case_block += (f"[Case {m.get('patient_id','?')}, {m.get('date','?')} "
                       f"- anonymised, clinic record]\n{c['text']}\n\n")
    if cases:
        print(f"[i] Included {len(cases)} similar past case chunk(s).\n")

    prompt = f"""You are a clinical decision-support assistant for a doctor, especially a
less-experienced one. Blend two sources, kept DISTINCT:

  1. TEXTBOOK CONTEXT = medical FACTS. Cite book and page for every factual claim.
  2. PAST CASES = how THIS clinic's expert previously handled similar presentations.
     This is precedent and clinical judgment, NOT textbook fact. It may reflect this
     expert's style. Use it to show the junior doctor "how it's been approached here,"
     but never let a past case override a textbook fact or a dose.

RULES:
1. Rank any differential by LIKELIHOOD for this patient. LEAD WITH COMMON causes.
   Structure: MOST LIKELY / LESS COMMON / MUST-NOT-MISS.
2. When management is implied, include "TREATMENT OPTIONS": drug CLASSES and AGENTS the
   BOOKS describe, with strength/dose ONLY where the passages state them, each cited.
3. NEVER state a dose not in the textbook passages. If none is given, name the drug and
   say the dose must be confirmed from a prescribing reference.
4. Add a section "HOW THIS CLINIC HAS HANDLED SIMILAR CASES" summarising the relevant
   PAST CASES - clearly framed as precedent/judgment, anonymised, not as evidence or fact.
5. End TREATMENT OPTIONS with a reminder that the clinician selects and verifies the drug
   for the individual patient, checking contraindications and interactions.
6. Base all medical FACTS on textbooks; base precedent on past cases; never blur them.
7. Be clinical, structured, concise.

TEXTBOOK CONTEXT:
{context}

PAST CASES (anonymised clinic precedent - NOT fact):
{case_block or "[no similar past cases found]"}

QUESTION: {question}

ANSWER:"""

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False, keep_alive=KEEP_ALIVE,
        options={"temperature": 0.2, "num_predict": MAX_ANSWER_TOKENS},
        stream=True,
    )
    print("ANSWER:\n")
    for part in stream:
        print(part["message"]["content"], end="", flush=True)

    print("\n\n--- Textbook passages used ---")
    for c in book_top:
        print(f"- {c['meta']['book']}, page {c['meta']['page']}")
    if cases:
        print("--- Past cases referenced (anonymised) ---")
        for c in cases:
            print(f"- {c['meta'].get('patient_id','?')}, {c['meta'].get('date','?')}")


if __name__ == "__main__":
    main()
