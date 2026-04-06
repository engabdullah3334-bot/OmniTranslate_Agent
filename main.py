"""
main.py — Translation Generator (7-Step Professional Pipeline)
==============================================================

Step 1 ─ User Input Phase   : File path (PDF/TXT) or raw text + target language
Step 2 ─ Text Extraction    : PDF → PyMuPDF | TXT/raw → chunked pages
Step 3 ─ LLaMA 3 Phase     : Detect source lang, normalize target lang, summarize
Step 4 ─ Save Extraction    : Write raw text to source_extraction.txt
Step 5 ─ Memory Handover    : Evict LLaMA from Ollama RAM + gc.collect()
Step 6 ─ NLLB Phase        : Load NLLB, read source_extraction.txt, translate
Step 7 ─ Output & Logging   : Save output.txt, log session, print success
"""

import gc
import json
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF

from orchestrator import ModelOrchestrator

# ─── File paths ───────────────────────────────────────────────────────────────
EXTRACTION_FILE = "source_extraction.txt"
OUTPUT_FILE = "output.txt"
OUTPUT_PDF = "output_translated.pdf"

# ___ Suppress warnings ________________________________________________________
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _header(text: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}")


def _step(n: int, title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  STEP {n} — {title}")
    print(f"{'─' * 60}")


def _progress(done: int, total: int) -> None:
    """Animated in-place progress bar."""
    pct = int(done / total * 100) if total else 100
    filled = pct // 5
    bar = "█" * filled + "░" * (20 - filled)
    print(f"\r  [{bar}] {pct:3d}%  ({done}/{total} pages)", end="", flush=True)
    if done == total:
        print()  # newline on completion


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SAVERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_txt(pages: list[dict], path: str) -> None:
    """Write translated pages to a plain-text file."""
    if os.path.exists(path):
        os.remove(path)
    with open(path, "a", encoding="utf-8") as f:
        for item in pages:
            text = item.get("translated_text", "").strip()
            if text:
                f.write(f"--- Page {item['page_index']} ---\n")
                f.write(text + "\n\n")


def _save_pdf(pages: list[dict], path: str) -> None:
    """Write translated text into a new PDF, one page per original page."""
    new_doc = fitz.open()
    for item in pages:
        pg = new_doc.new_page()
        pg.insert_text(
            (50, 50),
            item.get("translated_text") or "[No translation]",
            fontsize=12,
        )
    new_doc.save(path)
    new_doc.close()
    print(f"  [PDF] Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:

    _header("Translation Generator — LLaMA 3 Brain + NLLB Engine")

    # ── STEP 1: USER INPUT ────────────────────────────────────────────────────
    _step(1, "User Input Phase")

    input_source = input(
        "  File path (PDF/TXT), or press Enter to type text directly:\n  > "
    ).strip()

    target_lang_request = input(
        "\n  Target language?\n"
        "  (e.g. 'Translate to Arabic' or just 'arabic')\n  > "
    ).strip()

    if not target_lang_request:
        print("  [Error] No target language specified. Exiting.")
        sys.exit(1)

    # ── STEP 2: TEXT EXTRACTION ───────────────────────────────────────────────
    _step(2, "Text Extraction")

    doc = None
    input_type = "raw_text"

    if input_source and input_source.lower().endswith(".pdf") and os.path.exists(input_source):
        print(f"  [PDF] Reading: {input_source}")
        doc, pages = ModelOrchestrator.extract_pdf(input_source)
        input_type = "pdf"

    elif input_source and os.path.exists(input_source):
        print(f"  [TXT] Reading: {input_source}")
        raw = Path(input_source).read_text(encoding="utf-8")
        pages = ModelOrchestrator.extract_text(raw)
        input_type = "text_file"

    else:
        if input_source:
            print(f"  [Warning] File not found: '{input_source}'. Switching to raw-text mode.")
        raw_text = input("  Paste the text to translate:\n  > ")
        pages = ModelOrchestrator.extract_text(raw_text)
        input_source = "inline_text"

    if not pages:
        print("  [Error] No text found in input. Exiting.")
        sys.exit(1)

    print(f"  ✓ Loaded {len(pages)} page(s).")
    sample_text = pages[0]["original_text"]

    # ── STEP 3: LLAMA 3 — THE BRAIN ───────────────────────────────────────────
    _step(3, "LLaMA 3 Phase — Analyzing Request")

    # free_after=True → _flush_ollama(keep_alive=0) is called inside analyze_request
    analysis = ModelOrchestrator.analyze_request(
        user_request=target_lang_request,
        sample_text=sample_text,
        free_after=True,
    )

    print(f"\n  LLaMA analysis:\n{json.dumps(analysis, ensure_ascii=False, indent=4)}\n")

    # Resolve keys — support both prompt formats
    src_lang = (
        analysis.get("detected_source_lang")
        or analysis.get("source_language")
        or "unknown"
    )
    tgt_lang = (
        analysis.get("target_lang")
        or analysis.get("target_language")
        or "english"
    )

    # Let user correct if LLaMA misunderstood
    print(f"  Detected source language : {src_lang}")
    override = input("  Press Enter to confirm, or type correction: ").strip()
    if override:
        src_lang = override

    print(f"  Target language          : {tgt_lang}")
    override = input("  Press Enter to confirm, or type correction: ").strip()
    if override:
        tgt_lang = override

    # Store confirmed values back into analysis (used by log_request)
    analysis.update(
        detected_source_lang=src_lang,
        source_language=src_lang,
        target_lang=tgt_lang,
        target_language=tgt_lang,
    )

    # ── STEP 4: SAVE EXTRACTION → source_extraction.txt ──────────────────────
    _step(4, f"Saving Raw Extraction → {EXTRACTION_FILE}")

    ModelOrchestrator.save_extraction(pages, EXTRACTION_FILE)
    print(f"  ✓ {len(pages)} page(s) written to {EXTRACTION_FILE}")

    # ── STEP 5: MEMORY HANDOVER ───────────────────────────────────────────────
    _step(5, "Memory Handover (LLaMA eviction)")

    # LLaMA was already evicted inside analyze_request(free_after=True) via
    # keep_alive=0 REST call — confirmed here for clarity.
    print("  ✓ LLaMA evicted from Ollama RAM  (keep_alive = 0)")
    gc.collect()
    print("  ✓ Python heap flushed            (gc.collect)")
    print("  ✓ RAM is clear — NLLB may now load")

    # ── STEP 6: NLLB PHASE — TRANSLATION ─────────────────────────────────────
    _step(6, "NLLB Phase — Translation")

    print(f"  {src_lang}  →  {tgt_lang}\n")

    # Read pages from the handover file (clean interface between phases)
    pages_to_translate = ModelOrchestrator.load_extraction(EXTRACTION_FILE)

    # Clear output file before starting so it accumulates cleanly
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    try:
        translated_pages = ModelOrchestrator.translate_pages(
            pages_to_translate,
            source_language=src_lang,
            target_language=tgt_lang,
            progress_callback=_progress,
            output_file=OUTPUT_FILE,
        )
    except ValueError as exc:
        # Language not found in NLLB
        print(f"\n  [Error] Language mapping failed: {exc}")
        print("  Tip: Check LANG_CODES in translator_engine.py for valid names.")
        sys.exit(1)

    # ── STEP 7: OUTPUT & LOGGING ──────────────────────────────────────────────
    _step(7, "Output & Logging")

    # 7a — Save output.txt (Already done incrementally)
    print(f"  ✓ Translation incremental saving completed: {OUTPUT_FILE}")

    # 7b — Save PDF (only when input was a PDF)
    if input_type == "pdf" and doc:
        _save_pdf(translated_pages, OUTPUT_PDF)
        doc.close()

    # 7c — Append to translation_log.json
    ModelOrchestrator.log_request(analysis, input_source, len(pages))
    print("  ✓ Session appended to translation_log.json")

    # 7d — Final success banner
    print("\n" + "═" * 60)
    print("  ✅ Process Completed Successfully.")
    print(f"     Translation saved to {OUTPUT_FILE}")
    if input_type == "pdf":
        print(f"     PDF export saved to  {OUTPUT_PDF}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()

