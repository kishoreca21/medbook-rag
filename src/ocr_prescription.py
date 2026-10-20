r"""
ocr_prescription.py — MAIN. Read a prescription image, match every line against
your verified medicine DB, and write a doctor-review draft.

Pipeline:
    image ──► Sarvam Vision OCR ──► candidate lines
                                      │
                                      ▼
                         match each line vs medicines DB (fuzzy + alias memory)
                                      │
                                      ▼
            draft file: matched drugs (+salt/use), with REVIEW flags on anything
            below the confidence threshold and the top guesses for each.

Nothing here is trusted until the doctor verifies it — same discipline as the
rest of your pipeline. Low-confidence lines are flagged, never silently accepted.

Usage:
    python ocr_prescription.py path\to\image.jpg
    python ocr_prescription.py path\to\image.jpg --id Patient_018 --db All_Medicines_Salts_Filled.xlsx
    python ocr_prescription.py folder_of_images\        (whole folder)

Output: prescriptions/<id>_rx.txt   (DRAFT, STATUS: UNVERIFIED)
        prescriptions/<id>_rx.review.tsv   (machine-readable, for the learn step)
"""

import os
import argparse
import datetime

from sarvam_ocr import ocr_image, split_into_lines
from medicine_matcher import MedicineDB, AUTO_ACCEPT_SCORE, REVIEW_FLOOR

# Path anchoring (CWD-independent): HERE = PyFiles, ROOT = project root.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "prescriptions")
DB_NAME = "All_Medicines_Salts_Filled.xlsx"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".pdf")


def resolve_db(path):
    """Resolve the medicine DB xlsx without depending on the working directory.
    Tries, in order: the path as given, then ROOT, then HERE (PyFiles)."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in ("", ROOT, HERE):
        cand = path if base == "" else os.path.join(base, os.path.basename(path))
        if os.path.exists(cand):
            return cand
    return path  # let the caller raise a clear "not found"


def process_one(image_path, patient_id, db, language="hi-IN"):
    md = ocr_image(image_path, language=language)
    lines = split_into_lines(md)

    matched, review, nomatch = [], [], []
    rows_for_tsv = []
    for ln in lines:
        r = db.match_line(ln)
        st = r.get("status")
        if st == "matched":
            matched.append(r)
        elif st == "review":
            review.append(r)
        elif st == "no_match":
            nomatch.append(r)
        else:
            continue
        # tsv row for the learning step (raw -> chosen match, editable by human)
        rows_for_tsv.append("\t".join([
            r.get("raw", ""), st, r.get("match", ""),
            str(r.get("score", "")), r.get("salt", ""),
        ]))

    # ---- human-readable draft ----
    today = datetime.date.today().isoformat()
    header = (
        "=== PRESCRIPTION OCR (DRAFT - DOCTOR MUST VERIFY) ===\n"
        f"Patient ID : {patient_id}\n"
        f"Source     : {os.path.basename(image_path)}\n"
        f"Date       : {today}\n"
        "STATUS: UNVERIFIED. Confidence shown per line. Doctor confirms/corrects\n"
        f"        every line before use. Auto-accept >= {AUTO_ACCEPT_SCORE}; "
        f"review {REVIEW_FLOOR}-{AUTO_ACCEPT_SCORE}; below {REVIEW_FLOOR} = no match.\n"
        + "=" * 55 + "\n\n"
    )

    body = []
    body.append(f"MATCHED ({len(matched)}) — high confidence, still verify:")
    if matched:
        for r in matched:
            salt = f"  [{r['salt']}]" if r.get("salt") else ""
            body.append(f"  \u2713 {r['match']}{salt}   (score {r['score']}, read: \"{r['raw']}\")")
    else:
        body.append("  (none)")
    body.append("")

    body.append(f"REVIEW ({len(review)}) — likely, doctor must confirm:")
    if review:
        for r in review:
            cands = ", ".join(f"{c['name']} ({c['score']})" for c in r.get("candidates", [])[:3])
            body.append(f"  ? read \"{r['raw']}\"  -> best: {r['match']} (score {r['score']})")
            body.append(f"      other guesses: {cands}")
    else:
        body.append("  (none)")
    body.append("")

    body.append(f"NO MATCH ({len(nomatch)}) — not in DB or OCR too garbled:")
    if nomatch:
        for r in nomatch:
            cands = ", ".join(f"{c['name']} ({c['score']})" for c in r.get("candidates", [])[:3])
            body.append(f"  \u2717 read \"{r['raw']}\"   closest: {cands or '(nothing close)'}")
    else:
        body.append("  (none)")
    body.append("")
    body.append("--- RAW OCR (for reference) ---")
    body.append(md)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{patient_id}_rx.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(body) + "\n")

    # ---- machine-readable companion for the learning step ----
    tsv_path = os.path.join(OUT_DIR, f"{patient_id}_rx.review.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("raw_read\tstatus\tmatched_name\tscore\tsalt\n")
        f.write("\n".join(rows_for_tsv) + "\n")

    return out_path, len(matched), len(review), len(nomatch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="image file OR folder of images")
    ap.add_argument("--id", help="patient id (single-file; default = filename stem)")
    ap.add_argument("--db", default=DB_NAME, help="medicine database xlsx")
    ap.add_argument("--lang", default="hi-IN", help="OCR language (default hi-IN)")
    args = ap.parse_args()

    db_path = resolve_db(args.db)
    if not os.path.exists(db_path):
        print(f"[ERROR] DB not found: {args.db} "
              f"(looked in project root and PyFiles\\)")
        return
    db = MedicineDB(db_path)
    print(f"[*] Loaded DB: {len(db.records)} medicines, {len(db._choices)} match keys.\n")

    if os.path.isdir(args.path):
        imgs = sorted(f for f in os.listdir(args.path) if f.lower().endswith(IMG_EXTS))
        if not imgs:
            print(f"No images in {args.path}")
            return
        for i, fn in enumerate(imgs, 1):
            pid = os.path.splitext(fn)[0]
            print(f"[{i}/{len(imgs)}] {fn} (id={pid})")
            try:
                out, m, rv, nm = process_one(os.path.join(args.path, fn), pid, db, args.lang)
                print(f"    -> {out}   matched={m} review={rv} no_match={nm}")
            except Exception as e:
                print(f"    [!] FAILED: {e}")
    else:
        pid = args.id or os.path.splitext(os.path.basename(args.path))[0]
        print(f"[*] {os.path.basename(args.path)} (id={pid})")
        try:
            out, m, rv, nm = process_one(args.path, pid, db, args.lang)
            print(f"    -> {out}   matched={m} review={rv} no_match={nm}")
        except Exception as e:
            print(f"    [!] FAILED: {e}")

    print("\n[*] Done. DOCTOR verifies each prescriptions/*_rx.txt.")
    print("    To teach corrections back into the DB:  python learn_corrections.py <id>")


if __name__ == "__main__":
    main()
