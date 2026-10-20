"""
sarvam_ocr.py — Read a prescription image via Sarvam Vision (Document Intelligence).

Uses the OFFICIAL documented Document Intelligence job flow (verified against
docs.sarvam.ai), NOT a guessed endpoint:

    client.document_intelligence.create_job(language=..., output_format="md")
    job.upload_file(path) -> job.start() -> job.wait_until_complete()
    job.download_output(zip_path)

The ZIP contains your chosen format (md/html) PLUS a JSON with page-level data.
We read the markdown for the human-readable text.

Why Sarvam Vision: it's a 3B doc-intelligence model trained on handwritten text
across 22 Indian languages + English — well suited to Hindi+English prescriptions.

Gotchas baked in (from the docs, these are easy to get wrong):
  - parameter is `language` (e.g. "hi-IN"), NOT `language_code`
  - output_format must be "md" or "html" — "markdown" returns HTTP 400
  - PDF/ZIP capped at 10 pages

PRIVACY: only image pixels are uploaded. No patient name/metadata is attached.
Strip identifiers from the image before sending if any are visible.

SETUP:
    pip install -U sarvamai rapidfuzz openpyxl
    set SARVAM_API_KEY=your_key      (or put it in apikey.txt next to this file)
"""

import os
import io
import zipfile
import tempfile


def get_api_key():
    """Find the Sarvam key, in priority order:
      1. your existing project helper:  from sarvam_key import get_sarvam_key
         (this is what transcribe_sarvam.py / your original ocr_prescription.py use)
      2. the SARVAM_API_KEY environment variable
      3. a key file next to the scripts — case-insensitive, so APIKEY.txt,
         apikey.txt, ApiKey.txt etc. all work (Windows is case-insensitive anyway,
         but this also covers mac/linux)
    Returns the key string, or raises with a clear message."""
    # 1) reuse the project's existing loader if present (keeps ONE source of truth)
    try:
        from sarvam_key import get_sarvam_key
        k = (get_sarvam_key() or "").strip()
        if k:
            return k
    except Exception:
        pass

    # 2) environment variable
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if key:
        return key

    # 3) a key file next to the scripts (case-insensitive match)
    here = os.path.dirname(os.path.abspath(__file__))
    wanted = {"apikey.txt", "sarvam_key.txt", "sarvamkey.txt"}
    try:
        for fname in os.listdir(here):
            if fname.lower() in wanted:
                with open(os.path.join(here, fname), encoding="utf-8") as f:
                    k = f.read().strip()
                    if k:
                        return k
    except OSError:
        pass

    raise RuntimeError(
        "No Sarvam API key found. Do ONE of:\n"
        "  - keep your sarvam_key.py helper in the same folder (it already works for\n"
        "    your other scripts), or\n"
        "  - set the SARVAM_API_KEY environment variable, or\n"
        "  - place APIKEY.txt (one line) in the same folder as these scripts."
    )


def ocr_image(image_path, language="hi-IN", keep_zip=False):
    """Run Sarvam Document Intelligence on one image/PDF. Returns the extracted
    markdown text (string). Raises on hard API errors."""
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=get_api_key())

    # 1) create job. language=hi-IN handles the Hindi+English code-mix; the model
    #    detects English drug names within the Devanagari context.
    job = client.document_intelligence.create_job(
        language=language,
        output_format="md",
    )
    # 2) upload -> 3) start -> 4) wait
    job.upload_file(image_path)
    job.start()
    status = job.wait_until_complete()

    state = getattr(status, "job_state", None) or getattr(status, "status", "")
    if str(state).lower() not in ("completed", "partiallycompleted"):
        raise RuntimeError(f"Sarvam job did not complete (state={state}).")

    # 5) download ZIP and pull the markdown out of it
    tmp = tempfile.mkdtemp(prefix="sarvam_rx_")
    zip_path = os.path.join(tmp, "out.zip")
    job.download_output(zip_path)

    md_text = _extract_markdown_from_zip(zip_path)
    if not keep_zip:
        try:
            os.remove(zip_path)
        except OSError:
            pass

    # Clean before returning so callers never see a raw base64 image blob.
    md_text = clean_markdown(md_text)

    # If after cleaning there's essentially no readable text, the model failed
    # to OCR the page (it just handed back the image). Make that loud — a silent
    # empty/garbage result is how a bad scan slips into the matcher.
    if len(md_text.strip()) < 3:
        import sys
        print(
            "WARNING: Sarvam returned no readable text for this image "
            "(it likely embedded the raw image instead of transcribing it). "
            "Try a sharper/cropped scan, or set language to match the page "
            "(e.g. 'en-IN' if it's mostly English).",
            file=sys.stderr,
        )
    return md_text


def _extract_markdown_from_zip(zip_path):
    """Pull and concatenate all .md files from the output ZIP (page order)."""
    chunks = []
    with zipfile.ZipFile(zip_path) as z:
        md_names = sorted(n for n in z.namelist() if n.lower().endswith(".md"))
        if not md_names:
            # fall back to any .txt / .json text if md missing
            md_names = sorted(n for n in z.namelist()
                              if n.lower().endswith((".txt", ".json")))
        for n in md_names:
            with z.open(n) as f:
                chunks.append(f.read().decode("utf-8", errors="replace"))
    return "\n\n".join(chunks).strip()


def clean_markdown(md_text):
    """Remove non-text junk that Sarvam sometimes returns instead of (or
    alongside) real OCR text. The big one: when the model can't read the page
    it embeds the source image as a base64 data URI inside a markdown image
    tag, e.g.  ![Image](data:image/jpeg;base64,/9j/4AAQ...). That blob is tens
    of thousands of characters of gibberish and must NEVER reach the matcher.

    Strips:
      - data: URIs (base64 image payloads), with or without a markdown wrapper
      - markdown image embeds  ![alt](url)
      - bare markdown links     [text](url)  -> keeps the text
      - HTML <img ...> tags
    Returns the cleaned text.
    """
    import re
    # 1) kill any ![...](data:...) embed outright (the payload is huge)
    md_text = re.sub(r"!\[[^\]]*\]\(\s*data:[^)]*\)", " ", md_text,
                     flags=re.IGNORECASE | re.DOTALL)
    # 2) kill a bare data: URI even if it wasn't wrapped in markdown
    md_text = re.sub(r"data:image/[^\s)]+", " ", md_text,
                     flags=re.IGNORECASE)
    # 3) normal markdown image embeds (non-data) -> drop entirely
    md_text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md_text)
    # 4) markdown links [text](url) -> keep just the text
    md_text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md_text)
    # 5) HTML <img> tags
    md_text = re.sub(r"<img[^>]*>", " ", md_text, flags=re.IGNORECASE)
    return md_text


# A real medicine line is short. Anything past this is almost certainly a
# stray base64 chunk, a table dump, or paragraph noise — never a drug name.
MAX_CANDIDATE_LEN = 80


def split_into_lines(md_text):
    """Turn the OCR markdown into candidate medicine lines. Drops obvious
    non-medicine noise (headers, vitals, blank/markdown-table rows, image
    embeds, over-long blobs). This is a first pass — the matcher decides what's
    actually a drug."""
    import re
    md_text = clean_markdown(md_text)
    lines = []
    for raw in md_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # strip markdown decoration
        s = re.sub(r"^[#>*\-\|\s]+", "", s).strip()
        s = s.replace("|", " ").strip()
        if len(s) < 2:
            continue
        # a genuine drug line is short; long blobs are OCR/base64 noise
        if len(s) > MAX_CANDIDATE_LEN:
            continue
        # reject lines that are mostly non-letters (base64 fragments, symbols)
        letters = sum(c.isalpha() for c in s)
        if letters < max(2, len(s) // 2):
            continue
        # reject any line containing a long unbroken alphanumeric run — that's
        # the signature of a base64 chunk; real drug names have spaces/dashes
        if re.search(r"[A-Za-z0-9/+]{25,}", s):
            continue
        # skip lines that are clearly vitals/header (heuristic, tunable)
        low = s.lower()
        if any(tok in low for tok in ("bp -", "bp-", "wt -", "wt-", "weight",
                                      "age ", "sex ", "date ", "s.n", "s/n")):
            continue
        lines.append(s)
    return lines


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python sarvam_ocr.py <image_path> [language]")
        raise SystemExit(1)
    lang = sys.argv[2] if len(sys.argv) > 2 else "hi-IN"
    text = ocr_image(sys.argv[1], language=lang)
    print("=== RAW OCR MARKDOWN ===")
    print(text)
    print("\n=== CANDIDATE LINES ===")
    for ln in split_into_lines(text):
        print("  ", ln)
