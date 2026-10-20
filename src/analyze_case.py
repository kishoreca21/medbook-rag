"""
analyze_case.py — Step 3: in-house clinical analysis (Ollama, fully local).

Combines, per patient:
  - the transcript        transcripts/<id>.txt          (from Sarvam STT)
  - the VERIFIED Rx       prescriptions/<id>_rx.txt      (doctor-checked OCR)
into one teaching-oriented analysis      summaries/<id>_summary.txt

This step is 100% in-house: Sarvam never sees the analysis, diagnosis, or the
medicine-to-symptom mapping. Only the raw audio and raw image went to the cloud.

Cross-check rule (as agreed):
  - The analysis NOTES connections between prescribed drugs and discussed symptoms.
  - Where a prescribed drug has no obvious basis in the transcript, it does NOT flag
    or challenge it - the doctor's prescription is authoritative. It just records a
    neutral observation in a COMMENTS field for the doctor's awareness.

Inferred clinical reasoning is tagged [AI-INFERRED - CONFIRM OR REWRITE] for the
doctor to confirm before ingestion (same discipline as before).

SETUP: needs Ollama running + the books RAG (for grounding). No internet.

Usage:
    python analyze_case.py                 process all matched pairs
    python analyze_case.py --id Patient_042   one patient
"""

import os
import argparse
import datetime
import chromadb
import ollama
from sentence_transformers import CrossEncoder

# ---------------- Settings (mirror ask.py) ----------------
# Path anchoring (ironclad, CWD-independent):
#   HERE  = this script's folder = PyFiles
#   ROOT  = project root         = parent of PyFiles
# The vector DB + book PDFs live under PyFiles (with the code); the patient-data
# folders (transcripts / prescriptions / summaries) live under the project root,
# matching where transcribe_sarvam.py writes them.
HERE            = os.path.dirname(os.path.abspath(__file__))
ROOT            = os.path.dirname(HERE)
DB_DIR          = os.path.join(HERE, "vectordb")
BOOK_COLLECTION = "medical_books"
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "qwen3:4b"        # bigger model recommended for this reasoning step
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TRANSCRIPT_DIR  = os.path.join(ROOT, "transcripts")
RX_DIR          = os.path.join(ROOT, "prescriptions")
OUT_DIR         = os.path.join(ROOT, "summaries")
KEEP_ALIVE      = "30m"
MAX_TOKENS      = 2000
GROUNDING_K     = 6
RX_UNVERIFIED   = "STATUS: UNVERIFIED"   # marker meaning Rx not yet doctor-checked
# ----------------------------------------------------------

_reranker = CrossEncoder(RERANK_MODEL)


def read_file(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def find_transcript(patient_id):
    """Locate <patient_id>.txt anywhere under TRANSCRIPT_DIR, including dated
    subfolders (transcribe_sarvam.py writes to transcripts\\<date>\\<id>.txt,
    while prescriptions/summaries are flat). Returns the path or None."""
    target = f"{patient_id}.txt"
    if not os.path.isdir(TRANSCRIPT_DIR):
        return None
    for root, _dirs, files in os.walk(TRANSCRIPT_DIR):
        if target in files:
            return os.path.join(root, target)
    return None


def strip_header(text):
    """Drop a header block if present, return body.
    Handles both the long '====...' separator (40+ '=') used by Rx/summary files
    and the short '=== RAG RECORD ... ===' marker used by codemix transcripts."""
    if "=" * 40 in text:
        return text.split("=" * 40)[-1].strip()
    # codemix transcript: drop the single '=== RAG RECORD ... ===' header line
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("=== ") and lines[0].strip().endswith("==="):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def get_grounding(query_text):
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        col = client.get_collection(BOOK_COLLECTION)
    except Exception as e:
        print(f"      [i] book grounding unavailable ({e}).")
        return ""
    emb = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)["embedding"]
    res = col.query(query_embeddings=[emb], n_results=GROUNDING_K * 2)
    cands = [{"text": d, "meta": m}
             for d, m in zip(res["documents"][0], res["metadatas"][0])]
    if not cands:
        return ""
    scores = _reranker.predict([[query_text, c["text"]] for c in cands])
    top = [c for _, c in sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)][:GROUNDING_K]
    return "\n\n".join(f"[{c['meta'].get('book','?')}, p.{c['meta'].get('page','?')}]\n{c['text']}"
                       for c in top)


def build_prompt(transcript, rx, grounding):
    return f"""You are a psychiatric/clinical scribe building a TEACHING record so a future,
less-experienced doctor can learn how this expert reasons. Combine the interview
transcript and the doctor's VERIFIED prescription into one structured analysis.

INPUTS:
  - TRANSCRIPT: the consultation (may have minor transcription errors).
  - PRESCRIPTION: the doctor's verified medicines (authoritative - do NOT question it).
  - TEXTBOOK PASSAGES: to ground inferred reasoning.

RULES:
- Patient case and the doctor's plan/medicines come ONLY from the inputs.
- You MAY infer likely clinical reasoning, grounded in and citing the textbook passages.
  Tag EVERY inferred line [AI-INFERRED - CONFIRM OR REWRITE]. Never present it as fact.
- CROSS-CHECK (observational only): note where a prescribed medicine connects to a
  symptom or finding in the transcript. If a prescribed medicine has NO clear basis in
  the transcript, do NOT flag, warn, or question it - the doctor's prescription is
  correct by definition. Simply record a neutral note in COMMENTS, e.g. "Drug X
  prescribed; transcript did not discuss a clear indication (noted for awareness)."
- Do NOT invent symptoms, doses, or diagnoses. Mark ambiguities [UNCLEAR].
- Be concise and clinical.

Produce EXACTLY these sections:

PATIENT ID / DATE:
CHIEF COMPLAINT:
HISTORY OF PRESENT ILLNESS:
REPORTED SYMPTOMS:
RELEVANT HISTORY:
MEDICINES PRESCRIBED (verbatim from the verified prescription):
SYMPTOM-MEDICINE MAPPING (observational connections; neutral COMMENTS for any drug
  without a clear transcript basis - never a warning):
LIKELY CLINICAL REASONING ([AI-INFERRED - CONFIRM OR REWRITE], cite textbooks):
TEACHING POINTS ([AI-INFERRED - CONFIRM OR REWRITE]):
COMMENTS / NOTES FOR DOCTOR:
UNCERTAIN / NEEDS VERIFICATION:

--- TRANSCRIPT ---
{transcript or "[none]"}

--- VERIFIED PRESCRIPTION ---
{rx or "[none]"}

--- TEXTBOOK PASSAGES ---
{grounding or "[none retrieved]"}
"""


def analyze_one(patient_id):
    transcript = strip_header(read_file(find_transcript(patient_id)))
    rx_raw = read_file(os.path.join(RX_DIR, f"{patient_id}_rx.txt"))

    if rx_raw and RX_UNVERIFIED in rx_raw:
        return ("rx_unverified", None)
    rx = strip_header(rx_raw)

    if not transcript:
        return ("no_transcript", None)

    grounding = get_grounding((transcript + " " + rx)[:2000])
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_prompt(transcript, rx, grounding)}],
        think=False, keep_alive=KEEP_ALIVE,
        options={"temperature": 0.1, "num_predict": MAX_TOKENS},
    )
    out = resp["message"]["content"].strip()
    if "</think>" in out:
        out = out.split("</think>")[-1].strip()

    header = (f"=== CASE ANALYSIS (DRAFT - FOR DOCTOR REVIEW) ===\n"
              f"Patient ID: {patient_id}\nDate: {datetime.date.today().isoformat()}\n"
              f"STATUS: UNVERIFIED. Doctor confirms/rewrites every [AI-INFERRED] line\n"
              f"        and removes the tags before ingestion.\n"
              f"{'='*55}\n\n")
    out_path = os.path.join(OUT_DIR, f"{patient_id}_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + out + "\n")
    return ("ok", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="single patient id")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.id:
        ids = [args.id]
    else:
        # every transcript (searched recursively, incl. dated subfolders) that
        # has a matching prescription file
        if not os.path.isdir(TRANSCRIPT_DIR):
            print("No transcripts/ folder.")
            return
        ids = []
        for root, _dirs, files in os.walk(TRANSCRIPT_DIR):
            for f in files:
                if f.lower().endswith(".txt"):
                    pid = os.path.splitext(f)[0]
                    if os.path.exists(os.path.join(RX_DIR, f"{pid}_rx.txt")):
                        ids.append(pid)
        ids = sorted(set(ids))
        if not ids:
            print("No transcript+prescription pairs found.")
            print("Pairing is by filename stem: transcripts/<id>.txt  <->  "
                  "prescriptions/<id>_rx.txt")
            print("Name the prescription image with the SAME stem as the audio "
                  "segment (e.g. 06thJun_01.jpg -> 06thJun_01_rx.txt).")
            return

    print(f"[*] Analyzing {len(ids)} case(s)...\n")
    ok = skipped = 0
    for i, pid in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {pid}")
        status, out = analyze_one(pid)
        if status == "ok":
            print(f"   -> {out}")
            ok += 1
        elif status == "rx_unverified":
            print("   -> SKIPPED: prescription not yet verified by doctor "
                  "(still says UNVERIFIED). Verify the _rx.txt first.")
            skipped += 1
        else:
            print("   -> SKIPPED: transcript missing.")
            skipped += 1

    print(f"\n[*] Done. {ok} analysed, {skipped} skipped.")
    print("    Doctor reviews summaries/, confirms [AI-INFERRED] lines, THEN:")
    print("    python ingest_interview.py")


if __name__ == "__main__":
    main()
