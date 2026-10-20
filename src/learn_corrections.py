"""
learn_corrections.py — Teach the DB from doctor corrections (the memory loop).

This is what makes the system get smarter. When Sarvam misreads a drug and you
correct it, this records the raw misread as an ALIAS of the correct medicine in
your DB. Next time that same misread appears, the matcher snaps to it instantly.

IMPORTANT — Sarvam has no memory. It reads every image cold. This local alias
memory in your Excel is the ONLY learning in the system, which is good: it stays
private, on your machine, and under your control.

APPROVAL, not auto-learn. A wrong alias permanently teaches a wrong mapping —
exactly the dangerous silent error we guard against everywhere. So corrections
go through you explicitly here; nothing is learned automatically.

How to use
----------
After the doctor reviews prescriptions/<id>_rx.txt, edit the companion file
    prescriptions/<id>_rx.review.tsv
Set the "matched_name" column to the CORRECT DB medicine name for any line that
was wrong (leave correct ones as-is; blank it to skip). Then:

    python learn_corrections.py <id>

It will, for each corrected line, show the raw read -> correct medicine and ask
you to confirm before writing the alias. Type 'y' to teach it.

You can also teach a single pair directly:
    python learn_corrections.py --pair "Eitlax Md 0.25" "ETILAX MD 0.25"
"""

import os
import sys
import argparse

from medicine_matcher import MedicineDB

# Path anchoring (CWD-independent): HERE = PyFiles, ROOT = project root.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "prescriptions")
DEFAULT_DB = "All_Medicines_Salts_Filled.xlsx"


def resolve_db(path):
    """Resolve the medicine DB xlsx regardless of working directory:
    tries the given path, then project root, then PyFiles."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in ("", ROOT, HERE):
        cand = path if base == "" else os.path.join(base, os.path.basename(path))
        if os.path.exists(cand):
            return cand
    return path


def teach_pair(db, raw_read, correct_name, assume_yes=False):
    raw_read = raw_read.strip()
    correct_name = correct_name.strip()
    if not raw_read or not correct_name:
        return False
    # sanity: does correct_name exist in the DB?
    exists = any(rec["name"].strip().lower() == correct_name.lower() for rec in db.records)
    if not exists:
        print(f"  [skip] \"{correct_name}\" is not an exact DB medicine name. "
              f"Check spelling (must match the verbatim name column).")
        return False
    if not assume_yes:
        ans = input(f'  Teach:  "{raw_read}"  ->  {correct_name}  ?  [y/N] ').strip().lower()
        if ans != "y":
            print("    skipped.")
            return False
    ok = db.add_alias(correct_name, raw_read)
    print("    learned." if ok else "    [!] failed to write alias.")
    return ok


def from_review_file(db, patient_id, assume_yes=False):
    tsv = os.path.join(OUT_DIR, f"{patient_id}_rx.review.tsv")
    if not os.path.exists(tsv):
        print(f"No review file: {tsv}")
        return
    with open(tsv, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines or len(lines) < 2:
        print("Review file empty.")
        return
    header = lines[0].split("\t")
    try:
        i_raw = header.index("raw_read")
        i_name = header.index("matched_name")
        i_status = header.index("status")
    except ValueError:
        print("Review file header malformed.")
        return

    taught = 0
    for ln in lines[1:]:
        if not ln.strip():
            continue
        cols = ln.split("\t")
        raw = cols[i_raw] if len(cols) > i_raw else ""
        name = cols[i_name] if len(cols) > i_name else ""
        status = cols[i_status] if len(cols) > i_status else ""
        if not raw or not name:
            continue
        # Only worth teaching when the read differs from the chosen name
        # (i.e. it was a correction or a weak/no match the doctor resolved).
        if raw.strip().lower() == name.strip().lower():
            continue
        if teach_pair(db, raw, name, assume_yes):
            taught += 1
    print(f"\n[*] Done. Taught {taught} new alias(es). DB updated: {db.path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patient_id", nargs="?", help="patient id whose review.tsv to learn from")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--pair", nargs=2, metavar=("RAW", "CORRECT"),
                    help="teach a single misread->correct pair directly")
    ap.add_argument("--yes", action="store_true", help="skip per-line confirmation")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        args.db = resolve_db(args.db)
    if not os.path.exists(args.db):
        print(f"[ERROR] DB not found: {args.db} (looked in project root and PyFiles\\)")
        return
    db = MedicineDB(args.db)

    if args.pair:
        teach_pair(db, args.pair[0], args.pair[1], assume_yes=args.yes)
        return
    if not args.patient_id:
        print("Give a patient id (to learn from its review.tsv) or use --pair.")
        return
    from_review_file(db, args.patient_id, assume_yes=args.yes)


if __name__ == "__main__":
    main()
