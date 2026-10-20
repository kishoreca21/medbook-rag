"""
sarvam_key.py — central place to load the Sarvam API key.

Order it tries:
  1. environment variable SARVAM_API_KEY  (best practice)
  2. a local file 'apikey.txt' sitting next to the scripts (easy + keeps key
     OUT of your .py files, so sharing/backing up code never leaks the key)

Create apikey.txt with ONE line: your key, nothing else.
NEVER commit apikey.txt to git or share it. Add it to .gitignore.
"""

import os

def get_sarvam_key():
    # 1. environment variable
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if key:
        return key
    # 2. apikey.txt next to this file (or in current dir)
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "apikey.txt"), "apikey.txt"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                k = f.read().strip()
            if k:
                return k
    raise RuntimeError(
        "No Sarvam key found. Create 'apikey.txt' (one line = your key) "
        "next to the scripts, OR run:  set SARVAM_API_KEY=your_key"
    )
