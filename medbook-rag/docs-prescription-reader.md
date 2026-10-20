# Prescription Reader — OCR + Medicine-DB Matcher (with learning memory)

Reads a handwritten prescription image, extracts the medicine lines with **Sarvam
Vision** (Document Intelligence), and matches each line against **your verified
medicine database** using fuzzy matching. Low-confidence reads are flagged for the
doctor — never silently accepted. Doctor corrections are taught back into the DB so
the system gets better over time.

## Files

| File | What it does |
|---|---|
| `medicine_matcher.py` | Loads the DB, fuzzy-matches OCR lines, manages the alias memory. The brain. |
| `sarvam_ocr.py` | Calls Sarvam Vision (official Document Intelligence job flow) to OCR an image. |
| `ocr_prescription.py` | **MAIN.** image → OCR → match → writes a doctor-review draft. |
| `learn_corrections.py` | Teaches doctor corrections back into the DB as aliases (with approval). |
| `All_Medicines_Salts_Filled.xlsx` | **The database** (single source of truth). Verbatim bill names + salts. |

## Setup

```bash
pip install -U sarvamai rapidfuzz openpyxl
set SARVAM_API_KEY=your_key          # Windows
# or: export SARVAM_API_KEY=your_key  (mac/linux)
# or: put the key in apikey.txt next to these scripts
```

## Run

```bash
# one image
python ocr_prescription.py demo_2.jpeg --id Patient_018

# a whole folder
python ocr_prescription.py prescriptions_raw\06thJun\
```

Output → `prescriptions/<id>_rx.txt` (human draft, STATUS: UNVERIFIED) and
`prescriptions/<id>_rx.review.tsv` (machine-readable, used for learning).

The draft sorts every line into **MATCHED** (high confidence), **REVIEW** (likely —
doctor confirms), and **NO MATCH** (not in DB or OCR too garbled), with the top
guesses shown for the weak ones.

## The learning loop (this is the important part)

**Sarvam has no memory** — it reads every image cold and cannot learn from
corrections. The learning lives entirely in *your* DB, in the `Aliases` column.
That's by design: it stays local, private, and under your control.

When a drug is misread:

1. Doctor reviews `prescriptions/<id>_rx.txt`.
2. In `prescriptions/<id>_rx.review.tsv`, set the `matched_name` column to the
   correct DB medicine name for any wrong line.
3. Run:
   ```bash
   python learn_corrections.py Patient_018
   ```
   It asks you to confirm each correction, then writes the raw misread as an
   alias of the correct drug.

Next time that same misread appears, it matches **instantly at 100%**.

You can also teach one pair directly:
```bash
python learn_corrections.py --pair "Eitlax Md 0.25" "ETILAX MD 0.25"
```

## Safety design

- **Strict threshold.** Auto-accept only ≥ 88% similarity. A *missing* match is
  safe (it gets reviewed); a *wrong* match is dangerous, so we err toward flagging.
- **Approval before learning.** Corrections are confirmed by a human before
  becoming permanent aliases — a wrong alias would teach a wrong mapping forever.
- **Nothing trusted until verified.** Every draft is UNVERIFIED; the doctor signs
  off before anything flows downstream.
- **DB is separate from code.** Edit the Excel once; every run uses the new version.

## How it connects to analyze_case.py (your existing pipeline)

This step writes `prescriptions/<id>_rx.txt` — the **exact path and filename**
`analyze_case.py` already reads. So they connect with no changes. The flow:

1. `python ocr_prescription.py <image> --id <id>` → writes the draft (marked
   `STATUS: UNVERIFIED`).
2. **Doctor opens the `_rx.txt`, confirms/corrects the medicine list, and removes
   the `STATUS: UNVERIFIED` line.** (Optionally clean it to a plain medicine list.)
3. `python analyze_case.py --id <id>` → it sees the Rx is no longer UNVERIFIED and
   runs the clinical analysis combining transcript + verified Rx.

The `STATUS: UNVERIFIED` gate is deliberate: `analyze_case.py` **skips** any
prescription still marked UNVERIFIED, so an unreviewed OCR draft can never flow
into analysis by accident. The header separator (`=` × 55) matches the rest of
your pipeline so `strip_header` parses it cleanly.

Run order for one patient:
```
python transcribe_sarvam.py            # audio -> transcripts/<id>.txt
python ocr_prescription.py <image> --id <id>   # image -> prescriptions/<id>_rx.txt (DRAFT)
#   [doctor verifies the _rx.txt, removes UNVERIFIED]
python learn_corrections.py <id>       # (optional) teach any misreads back to DB
python analyze_case.py --id <id>       # transcript + Rx -> summaries/<id>_summary.txt
#   [doctor reviews summary, removes [AI-INFERRED] tags]
python ingest_interview.py             # -> RAG
```

## Tuning

In `medicine_matcher.py`:
- `AUTO_ACCEPT_SCORE` (default 88) — raise for stricter, lower for more auto-accepts.
- `REVIEW_FLOOR` (default 70) — below this a line is treated as no-match.

## Notes / limits

- Sarvam Document Intelligence caps PDFs/ZIPs at 10 pages.
- Param is `language` (e.g. `hi-IN`), not `language_code`; `output_format` must be
  `"md"` (not `"markdown"`). These are baked in correctly already.
- The DB still has rows flagged `FLAG` (doctor to confirm the salt). Matching works
  regardless — the flag is about the salt, not the name.
