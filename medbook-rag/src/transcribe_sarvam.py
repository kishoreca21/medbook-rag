"""
transcribe_sarvam.py — Cloud transcription via Sarvam Saaras v3 BATCH API.

Uses the Batch API (not the <30s REST call) because the interviews are minutes
long. Diarization (speaker labels) is currently OFF (with_diarization=False) -
the deliberate cost/call-saving choice for codemix interviews. The output
parser still reads speaker fields if they appear, but falls back cleanly to a
flat transcript when (as now) there are none. Flip with_diarization to True in
run_job() if you later want per-speaker labels.

Flow per run:
  1. anonymize all audio into a temp folder (renamed, no patient names leave machine)
  2. create_job(mode="transcribe") for native transcription
  3. create_job(mode="translate")  for English
  4. upload -> start -> wait -> download JSON outputs
  5. parse JSON into the dual-pane transcripts/<id>.txt your pipeline expects

Folders resolve relative to the project root (parent of this PyFiles folder).

SETUP:
    pip install -U sarvamai
    put your key in apikey.txt (one line) next to the scripts

Usage:
    python transcribe_sarvam.py
"""

import os
import json
import shutil
import tempfile

# Resolve folders relative to project root (parent of this script's folder)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_THIS_DIR)
def _proj(p):
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)

# ---------------- Settings ----------------
AUDIO_DIR   = r"recordings\07thJun"     # change the date each day
OUT_DIR     = r"transcripts\07thJun"
LANGUAGE    = "hi-IN"
MODEL       = "saaras:v3"
AUDIO_EXTS  = (".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".amr")
NUM_SPEAKERS = None                     # set an int (e.g. 2 or 3) if you know it; helps diarization
POLL_SECONDS = 5
JOB_TIMEOUT  = 1200                     # 20 min ceiling per job

from sarvam_key import get_sarvam_key
try:
    API_KEY = get_sarvam_key()
except Exception as _e:
    API_KEY = ""
    print(_e)
# ------------------------------------------


def anonymize_all(src_dir, files):
    """Copy each audio file into a temp dir renamed to Audio_NN, preserving a
    mapping back to the real (output) name. Returns (tmp_dir, [paths], {anon->realstem})."""
    tmp_dir = tempfile.mkdtemp(prefix="sarvam_anon_")
    paths, mapping = [], {}
    for i, fn in enumerate(sorted(files), 1):
        ext = os.path.splitext(fn)[1].lower()
        anon = f"Audio_{i:03d}{ext}"
        shutil.copy2(os.path.join(src_dir, fn), os.path.join(tmp_dir, anon))
        paths.append(os.path.join(tmp_dir, anon))
        mapping[anon] = os.path.splitext(fn)[0]   # real stem for output filename
    return tmp_dir, paths, mapping


def run_job(client, mode, paths):
    """Create a batch job in the given mode, run it, return {anon_audio_basename: local_json_path}.
    Diarization is off by default (see module docstring); set with_diarization=True below to enable."""
    kwargs = dict(model=MODEL, with_diarization=False, language_code=LANGUAGE)
    if mode:
        kwargs["mode"] = mode
    if NUM_SPEAKERS:
        kwargs["num_speakers"] = NUM_SPEAKERS

    job = client.speech_to_text_job.create_job(**kwargs)
    job.upload_files(paths)
    job.start()
    job.wait_until_complete(poll_interval=POLL_SECONDS, timeout=JOB_TIMEOUT)

    out_tmp = tempfile.mkdtemp(prefix="sarvam_out_")
    job.download_outputs(out_tmp)

    # List what actually landed on disk after download
    downloaded = {
        f: os.path.join(out_tmp, f)
        for f in os.listdir(out_tmp)
        if os.path.isfile(os.path.join(out_tmp, f))
    }
    print(f"      [debug] downloaded files in {out_tmp}: {list(downloaded.keys())}")

    # Build result map: anon audio basename -> local json path
    # The API may return full paths or bare names in input_file; normalise with basename.
    result = {}
    try:
        mappings = job.get_output_mappings()
        print(f"      [debug] output_mappings: {mappings}")
        for m in mappings:
            in_key  = os.path.basename(m.get("input_file",  ""))   # e.g. "Audio_001.wav"
            out_key = os.path.basename(m.get("output_file", ""))   # e.g. "Audio_001.json"
            candidate = os.path.join(out_tmp, out_key)
            if os.path.isfile(candidate):
                result[in_key] = candidate
            else:
                # output_file name from mapping doesn't match disk — try every downloaded file
                # by matching stem (Audio_001.wav -> Audio_001.json or 0.json etc.)
                in_stem = os.path.splitext(in_key)[0]
                matched = [p for name, p in downloaded.items()
                           if os.path.splitext(name)[0] == in_stem]
                if matched:
                    result[in_key] = matched[0]
                    print(f"      [debug] fuzzy-matched {in_key} -> {matched[0]}")
                else:
                    print(f"      [warn] no downloaded file found for mapping entry: {m}")
    except Exception as e:
        print(f"      [warn] get_output_mappings() failed ({e}); falling back to positional match")

    # Fallback: if mapping produced nothing (or API doesn't support it),
    # pair each uploaded path with downloaded files by sort order.
    if not result:
        sorted_inputs  = sorted(os.path.basename(p) for p in paths)
        sorted_outputs = sorted(downloaded.keys())
        print(f"      [debug] positional fallback: {sorted_inputs} -> {sorted_outputs}")
        for in_name, out_name in zip(sorted_inputs, sorted_outputs):
            result[in_name] = downloaded[out_name]

    return result, out_tmp


def parse_diarized_json(json_path):
    """Read a Sarvam batch output JSON and build a '[Speaker] text' transcript."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"[could not read output: {e}]"

    # Diarized output usually carries a list of turns/segments with speaker + transcript.
    for key in ("diarized_transcript", "output", "segments", "turns"):
        seg = data.get(key) if isinstance(data, dict) else None
        if isinstance(seg, dict):
            seg = seg.get("entries") or seg.get("turns") or seg.get("segments")
        if isinstance(seg, list) and seg:
            lines = []
            for s in seg:
                spk = s.get("speaker_id") or s.get("speaker") or "?"
                txt = (s.get("transcript") or s.get("text") or "").strip()
                if txt:
                    lines.append(f"[{spk}] {txt}")
            if lines:
                return "\n".join(lines)
    # fallback: flat transcript field
    if isinstance(data, dict):
        return (data.get("transcript") or data.get("text") or json.dumps(data)[:2000]).strip()
    return str(data)[:2000]


def main():
    if not API_KEY:
        print("ERROR: no Sarvam key. Put it in apikey.txt next to the scripts.")
        return
    try:
        from sarvamai import SarvamAI
    except ImportError:
        print("ERROR: SDK not installed.  pip install -U sarvamai")
        return

    src_dir = _proj(AUDIO_DIR)
    if not os.path.isdir(src_dir):
        print(f"Audio folder not found: {src_dir}")
        return
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(AUDIO_EXTS)]
    if not files:
        print(f"No audio files in '{src_dir}'.")
        return

    out_dir = _proj(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    client = SarvamAI(api_subscription_key=API_KEY)
    print(f"[*] Sarvam Saaras v3 BATCH (diarized) | {len(files)} file(s)\n")

    tmp_dir, paths, mapping = anonymize_all(src_dir, files)
    out_tmps = []
    try:
        print("[*] Job 1/2: native transcription (diarized)...")
        native_map, t1 = run_job(client, "transcribe", paths); out_tmps.append(t1)
        print("[*] Job 2/2: English translation (diarized)...")
        english_map, t2 = run_job(client, "translate", paths); out_tmps.append(t2)
    except Exception as e:
        print(f"[!] Batch job failed: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        for t in out_tmps: shutil.rmtree(t, ignore_errors=True)
        return

    # Build one dual-pane transcript per input file
    for anon_audio, real_stem in mapping.items():
        # anon_audio is already a bare filename (e.g. "Audio_001.wav"); use as-is.
        native_json = native_map.get(anon_audio)
        eng_json    = english_map.get(anon_audio)

        if not native_json:
            print(f"   [warn] no native JSON for {anon_audio}. Keys in map: {list(native_map.keys())}")
        if not eng_json:
            print(f"   [warn] no English JSON for {anon_audio}. Keys in map: {list(english_map.keys())}")

        native_pane  = parse_diarized_json(native_json)  if native_json  else "[no transcription]"
        english_pane = parse_diarized_json(eng_json)     if eng_json     else "[no translation]"

        out_path = os.path.join(out_dir, real_stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("=== HUMAN RECORD (DEVANAGARI) ===\n")
            f.write(native_pane + "\n\n")
            f.write("=== RAG RECORD (ENGLISH) ===\n")
            f.write(english_pane + "\n")
        print(f"   -> [SUCCESS] {out_path}")

    # cleanup all temp dirs (anonymized audio + downloaded outputs)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for t in out_tmps:
        shutil.rmtree(t, ignore_errors=True)

    print("\n[*] Done. Next:  python ocr_prescription.py / analyze_case.py")


if __name__ == "__main__":
    main()
