# medbook-rag — a private, local clinical AI for a small psychiatry practice

I built this for my father's psychiatry clinic in a Tier-2 town. The goal: capture how an experienced psychiatrist actually handles cases, so that a future, less-experienced doctor can learn from it, without patient data ever leaving the clinic.

> **This repository contains code and design only.** No patient recordings, transcripts, prescriptions, medicine database or vector index are included, and none should ever be committed. See `.gitignore`.

## The pipeline

```
consultation audio ──► transcribe_sarvam.py ──► transcripts/<id>.txt
prescription photo ──► ocr_prescription.py ──► prescriptions/<id>_rx.txt  (STATUS: UNVERIFIED)
                                    │
                          doctor reviews & signs off
                                    │
transcript + verified Rx ──► analyze_case.py (local LLM) ──► summaries/<id>_summary.txt
                                    │
                          doctor confirms [AI-INFERRED] lines
                                    │
                        ingest_interview.py ──► ChromaDB "patient_interviews"
psychiatry textbooks    ──► ingest_books.py   ──► ChromaDB "medical_books"
                                    │
                    ask.py ──► answer = textbook facts (cited) + how this clinic handled similar cases
```

| Step | Script | Runs where |
|---|---|---|
| Speech-to-text (Hindi/English, long audio) | `transcribe_sarvam.py` (Sarvam Saaras v3, batch API) | Cloud: raw audio only |
| Prescription OCR | `sarvam_ocr.py`, `ocr_prescription.py` (Sarvam Vision) | Cloud: raw image only |
| Fuzzy match to verified medicine DB | `medicine_matcher.py` (RapidFuzz) | Local |
| Learning loop from doctor corrections | `learn_corrections.py` | Local |
| Clinical analysis | `analyze_case.py` (Ollama + Qwen3 4B) | **Local, offline** |
| Retrieval-augmented Q&A | `ask.py` (ChromaDB + nomic-embed-text + cross-encoder reranker) | **Local, offline** |
| Audio splitting per patient | `split_audio.bat` (ffmpeg) | Local |

## Safety and privacy decisions

- **Analysis never leaves the machine.** Only raw audio and raw images touch a cloud API; diagnosis, reasoning and medicine-to-symptom mapping are done by a local model.
- **Nothing is trusted until a doctor verifies it.** OCR drafts are stamped `STATUS: UNVERIFIED`, and `analyze_case.py` refuses to process them until that line is removed.
- **A missing match is safe; a wrong match is dangerous.** Auto-accept only at ≥88% similarity; everything else goes to review.
- **AI reasoning is labelled.** Inferred lines carry `[AI-INFERRED - CONFIRM OR REWRITE]` until the doctor confirms them.
- **The prescription is authoritative.** The model notes connections between drugs and symptoms but never challenges a prescription.
- Designed with India's DPDP Act and the Mental Healthcare Act in mind: anonymised IDs (`Patient_042`), local storage, human sign-off.

More detail on the prescription reader: [docs-prescription-reader.md](docs-prescription-reader.md)

## Setup

```bash
pip install -U sarvamai rapidfuzz openpyxl chromadb ollama sentence-transformers pypdf
ollama pull qwen3:4b && ollama pull nomic-embed-text
set SARVAM_API_KEY=your_key      # Windows  (or put it in apikey.txt, which is git-ignored)
```

## Status

Working prototype on real clinic workflows (June 2026). Built with AI coding tools; prompt design, retrieval tuning and guardrails are my own.
