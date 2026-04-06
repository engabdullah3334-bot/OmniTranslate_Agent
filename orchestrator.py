"""
orchestrator.py — Model Orchestrator
=====================================
Single source of truth for the two-phase pipeline:

  Phase 1 ┌─ LLaMA 3 (Ollama HTTP) ──────────────────────────────────────────┐
           │  analyze_request() → structured JSON                              │
           │  _flush_ollama()   → keep_alive=0  ← PROPER RAM release          │
           │  gc.collect()      → clear Python heap                            │
           └──────────────────────────────────────────────────────────────────┘
                                      ↓  RAM clear
  Phase 2 ┌─ NLLB-200-1.3B (lazy) ──────────────────────────────────────────┐
           │  translator_engine.load()   ← called only after Phase 1 is free  │
           │  translate_pages()          ← page-by-page with progress callback │
           └──────────────────────────────────────────────────────────────────┘

Key design decisions:
  - Ollama RAM is freed via the REST API (keep_alive=0), NOT taskkill.
    taskkill kills the whole service; keep_alive=0 only evicts the model.
  - NLLB is never imported at module level — translator_engine.load() is
    the explicit gate that prevents both models coexisting in RAM.
  - PDF pages are dereferenced after text extraction to avoid fitz leaks.
  - All requests are appended to translation_log.json atomically.
"""

import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF

import translator_engine

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")

LOG_FILE = Path(__file__).parent / "translation_log.json"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")


# ══════════════════════════════════════════════════════════════════════════════
class ModelOrchestrator:
    """
    Stateless orchestrator — all methods are @staticmethod.
    Import and call directly; no instantiation needed.

    Typical usage:
        pages    = ModelOrchestrator.extract_bytes(raw_bytes)
        analysis = ModelOrchestrator.analyze_request(request, pages[0]["original_text"])
        pages    = ModelOrchestrator.translate_pages(pages, **analysis, callback=cb)
        ModelOrchestrator.log_request(analysis, filename, len(pages))
    """

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — LLaMA Brain
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def analyze_request(
        user_request: str,
        sample_text: str,
        model: str = OLLAMA_MODEL,
        free_after: bool = True,
    ) -> dict:
        """
        Call LLaMA 3 to detect languages and parse the user's intent.

        Parameters
        ----------
        user_request : Natural-language request from the client.
        sample_text  : First ~800 chars of the document to translate.
        model        : Ollama model tag (default: llama3).
        free_after   : If True, evicts the model from Ollama RAM after the call.
                       Set False only when running multiple consecutive calls.

        Returns
        -------
        dict with keys: target_lang, detected_source_lang, summary, file_status
        """
        from langchain_community.llms import Ollama  # lazy import

        prompt = ModelOrchestrator._build_prompt(user_request, sample_text)
        logger.info("Phase 1 — LLaMA analyzing request...")

        llm = Ollama(model=model)
        try:
            raw = llm.invoke(prompt)
        finally:
            del llm  # release LangChain wrapper

        result = ModelOrchestrator._parse_json(raw)

        # Normalize language values immediately after parsing
        # so that messy LLaMA output like "Translate to Arabic" becomes "arabic"
        for key in ("target_lang", "target_language"):
            if result.get(key):
                result[key] = ModelOrchestrator.normalize_language_name(result[key])
        for key in ("detected_source_lang", "source_language"):
            if result.get(key):
                result[key] = ModelOrchestrator.normalize_language_name(result[key])

        if free_after:
            ModelOrchestrator._flush_ollama(model)

        gc.collect()  # sweep Python heap regardless

        tgt = result.get("target_lang") or result.get("target_language", "?")
        src = result.get("detected_source_lang") or result.get("source_language", "?")
        logger.info("Phase 1 done — src: %s | tgt: %s", src, tgt)
        return result

    @staticmethod
    def _build_prompt(user_request: str, sample_text: str) -> str:
        # NOTE: No leading indentation — LLMs are sensitive to whitespace in prompts.
        return (
            "You are an intelligent assistant for a professional translation pipeline.\n\n"
            f'User translation request: """{user_request}"""\n\n'
            f'Sample of the document (first 800 chars): """{sample_text[:800]}"""\n\n'
            "Tasks:\n"
            "1. From the user request, extract and normalize the TARGET language "
            "as a lowercase English name (e.g. 'arabic', 'swahili', 'french').\n"
            "2. Detect the SOURCE language of the sample text (lowercase English name).\n"
            "3. Write a one-sentence summary of the document content.\n"
            "4. Set file_status to 'extracted'.\n\n"
            "Respond ONLY with valid JSON, no extra text or markdown:\n"
            "{\n"
            '  "target_lang": "<target language in English, lowercase>",\n'
            '  "detected_source_lang": "<source language in English, lowercase>",\n'
            '  "summary": "<one-sentence summary>",\n'
            '  "file_status": "extracted"\n'
            "}"
        )

    @staticmethod
    def _flush_ollama(model: str) -> None:
        """
        Evict `model` from Ollama's RAM by setting keep_alive=0.

        This is the correct Windows-safe approach:
          - taskkill /IM ollama.exe  → kills the entire service (wrong)
          - keep_alive=0 via API    → evicts only this model (correct)
        """
        import requests  # lazy import — not needed unless flushing
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=15,
            )
            if resp.status_code == 200:
                logger.info("Phase 1 — Ollama model '%s' evicted from RAM.", model)
            else:
                logger.warning("Phase 1 — Ollama flush returned HTTP %s.", resp.status_code)
        except Exception as exc:
            logger.warning("Phase 1 — Could not reach Ollama to flush: %s", exc)

    @staticmethod
    def normalize_language_name(raw: str) -> str:
        """
        Normalize a potentially messy LLaMA language name to a clean lowercase name.

        Examples
        --------
        "Translate to Arabic"  →  "arabic"
        "translate into arabic" →  "arabic"
        "Arabic (MSA)"         →  "arabic"
        "ENGLISH"              →  "english"
        "jabanis"              →  "japanese"  (fuzzy)
        """
        import re
        import translator_engine  # local import to avoid circular dependency

        s = raw.strip().lower()

        # 1. Strip common leading boilerplate phrases
        for prefix in (
            "please translate to ",
            "translate into ",
            "translate to ",
            "translated to ",
            "translation to ",
            "translating to ",
            "output in ",
            "into ",
            "in ",
            "to ",
        ):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                break

        # 2. Strip trailing qualifiers like " language" or " (msa)"
        s = re.sub(r'\s+language$', '', s)
        s = re.sub(r'\s*\(.*?\)', '', s).strip()

        # 3. Direct match against known language names
        if s in translator_engine.LANG_CODES:
            return s

        # 4. Substring match — e.g. "modern standard arabic" → "arabic"
        for known in translator_engine.LANG_CODES:
            if known in s:
                return known

        # 5. Fuzzy character-overlap match for common typos (e.g. "jabanis" → "japanese")
        best_match, best_score = "", 0.0
        for known in translator_engine.LANG_CODES:
            if abs(len(s) - len(known)) > 4:
                continue  # length too different, skip
            overlap = sum(1 for a, b in zip(s, known) if a == b)
            score = overlap / max(len(s), len(known))
            if score > best_score:
                best_score, best_match = score, known
        if best_score >= 0.65:
            logger.info("Language fuzzy-matched '%s' → '%s' (%.0f%%)", s, best_match, best_score * 100)
            return best_match

        return s  # return cleaned string; NLLB will raise ValueError if truly unknown

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract the first valid JSON object from a raw LLM string."""
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
        # Fallback: find the JSON block inside potential markdown/prose
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("JSON parse failed. Raw response snippet: %s", raw[:300])
        return {
            "target_lang":          "english",
            "detected_source_lang": "unknown",
            # legacy keys — kept for backward compat with log_request
            "source_language":      "unknown",
            "target_language":      "english",
            "summary": "Could not parse LLaMA response.",
            "notes":   raw,
            "file_status": "error",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — NLLB Translator
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def translate_pages(
        pages: list[dict],
        source_language: str,
        target_language: str,
        progress_callback: Callable[[int, int], None] | None = None,
        output_file: str | None = None,
        **_ignored,   # absorbs extra keys from the analysis dict if unpacked
    ) -> list[dict]:
        """
        Translate all pages using NLLB.
        Calls translator_engine.load() lazily — safe to call multiple times.

        Parameters
        ----------
        pages             : List of page dicts (must have 'original_text').
        source_language   : Human-readable source language name.
        target_language   : Human-readable target language name.
        progress_callback : Optional callable(done: int, total: int).

        Returns
        -------
        Same list with 'translated_text' filled in.
        """
        logger.info("Phase 2 — Loading NLLB...")
        translator_engine.load()  # no-op if already loaded
        logger.info("Phase 2 — NLLB ready. Translating %d pages...", len(pages))

        total = sum(1 for p in pages if p["original_text"].strip())
        done  = 0

        for item in pages:
            if not item["original_text"].strip():
                item["translated_text"] = ""
                continue
                
            if output_file:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"--- Page {item['page_index']} ---\n")

            try:
                item["translated_text"] = translator_engine.translate(
                    item["original_text"],
                    source_language,
                    target_language,
                    output_file=output_file
                )
                
                if output_file:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write("\n\n")
                        
                logger.debug("Page %d translated.", item["page_index"])
            except Exception as exc:
                logger.error("Page %d error: %s", item["page_index"], exc)
                item["translated_text"] = f"[Translation error: {exc}]"

            done += 1
            if progress_callback:
                progress_callback(done, total)

        logger.info("Phase 2 done — %d/%d pages translated.", done, total)
        return pages

    # ══════════════════════════════════════════════════════════════════════════
    # PDF / TEXT EXTRACTION + SAVE / LOAD
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def extract_pdf(pdf_path: str) -> tuple[fitz.Document, list[dict]]:
        """
        Extract per-page text from a PDF file path.
        Page objects are dereferenced immediately to avoid fitz memory leaks.

        Returns (doc, pages) — caller must close doc after saving output.
        """
        doc = fitz.open(pdf_path)
        pages = ModelOrchestrator._extract_pages(doc)
        return doc, pages

    @staticmethod
    def extract_bytes(raw_bytes: bytes, filetype: str = "pdf") -> list[dict]:
        """
        Extract text from raw bytes (Streamlit uploaded files).
        Document is closed immediately — no leaks.
        """
        doc = fitz.open(stream=raw_bytes, filetype=filetype)
        pages = ModelOrchestrator._extract_pages(doc)
        doc.close()
        return pages

    @staticmethod
    def extract_text(raw_text: str, chunk_size: int = 1_500) -> list[dict]:
        """Split plain text into chunks compatible with NLLB's token limit."""
        chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]
        return [
            {"page_index": i, "original_text": c.strip(), "translated_text": ""}
            for i, c in enumerate(chunks)
            if c.strip()
        ]

    @staticmethod
    def save_extraction(pages: list[dict], output_path: str = "source_extraction.txt") -> None:
        """
        Write raw extracted page text to a file for inter-phase handover.
        This is the Task C output (PDF → source_extraction.txt).
        """
        path = Path(output_path)
        with path.open("w", encoding="utf-8") as f:
            for item in pages:
                text = item.get("original_text", "").strip()
                if text:
                    f.write(f"--- Page {item['page_index']} ---\n")
                    f.write(text + "\n\n")
        logger.info("Extraction saved → %s  (%d pages)", output_path, len(pages))

    @staticmethod
    def load_extraction(input_path: str = "source_extraction.txt") -> list[dict]:
        """
        Read source_extraction.txt back into the page-dict format used by
        translate_pages(). Pairs with save_extraction() for the memory handover.
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Extraction file not found: '{input_path}'. "
                "Did save_extraction() run successfully?"
            )

        pages: list[dict] = []
        current_index: int | None = None
        current_lines: list[str] = []

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("--- Page ") and line.endswith(" ---"):
                # Flush previous page
                if current_index is not None:
                    pages.append({
                        "page_index":    current_index,
                        "original_text": "\n".join(current_lines).strip(),
                        "translated_text": "",
                    })
                try:
                    current_index = int(line[9:-4])
                except ValueError:
                    current_index = len(pages)
                current_lines = []
            elif current_index is not None:
                current_lines.append(line)

        # Flush the last page
        if current_index is not None:
            pages.append({
                "page_index":    current_index,
                "original_text": "\n".join(current_lines).strip(),
                "translated_text": "",
            })

        logger.info("Loaded %d page(s) from %s", len(pages), input_path)
        return pages

    @staticmethod
    def _extract_pages(doc: fitz.Document) -> list[dict]:
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text").strip()
            pages.append({"page_index": i, "original_text": text, "translated_text": ""})
            page = None   # explicit dereference — fitz keeps internal refs otherwise
        return pages

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def log_request(
        analysis: dict,
        input_source: str,
        page_count: int,
        success: bool = True,
    ) -> None:
        """
        Append a structured log entry to translation_log.json.
        Accepts both the old keys (source_language / target_language) and
        the new keys (detected_source_lang / target_lang) so it works
        regardless of which prompt format was used.
        """
        entry = {
            "timestamp":       datetime.now().isoformat(),
            "input_source":    input_source,
            "page_count":      page_count,
            "source_language": (
                analysis.get("detected_source_lang")
                or analysis.get("source_language")
            ),
            "target_language": (
                analysis.get("target_lang")
                or analysis.get("target_language")
            ),
            "summary":  analysis.get("summary"),
            "notes":    analysis.get("notes", ""),
            "success":  success,
        }
        log_data: list = []
        if LOG_FILE.exists():
            try:
                log_data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log_data = []
        log_data.append(entry)
        LOG_FILE.write_text(
            json.dumps(log_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Logged: %s → %s | file: %s",
            entry["source_language"], entry["target_language"], input_source,
        )
