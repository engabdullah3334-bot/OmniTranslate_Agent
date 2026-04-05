# OmniTranslate Agent &nbsp;![Version](https://img.shields.io/badge/version-1.0.0-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-green) ![License](https://img.shields.io/badge/license-MIT-orange)

> **AI-Powered Document Translator (LLaMA + NLLB)**
>
> A high-performance translation engine that orchestrates Meta's LLaMA for structural analysis and NLLB-200 for precise, multi-lingual document translation. It preserves document formatting while delivering state-of-the-art translation accuracy across 200+ languages.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [File Structure](#-file-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Supported Languages](#-supported-languages)
- [Configuration](#%EF%B8%8F-configuration)
- [Memory Strategy](#-memory-strategy)
- [API Reference](#-api-reference)
- [Changelog](#-changelog)

---

## 🧠 Overview

OmniTranslate Agent is a two-phase AI pipeline that separates *understanding* from *translating*:

| Phase | Model | Role |
|-------|-------|------|
| **Phase 1** | LLaMA 3 (via Ollama) | Understands the request, detects source language, normalizes target language, summarizes document context |
| **Phase 2** | NLLB-200-distilled-1.3B (HuggingFace) | Performs neural machine translation, page by page |

The two models **never coexist in RAM** — LLaMA is fully evicted via Ollama's `keep_alive=0` API before NLLB loads. This makes the pipeline viable on consumer-grade hardware.

---

## 🏗 Architecture

```
User Input (PDF / TXT / raw text)
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 1 — User Input Phase            │
│  • File path OR inline text           │
│  • Target language (natural language) │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  STEP 2 — Text Extraction             │
│  • PDF  → PyMuPDF → pages[]          │
│  • TXT  → chunked pages[]            │
│  • Raw  → chunked pages[]            │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  STEP 3 — LLaMA 3 Phase (The Brain)  │  ← Ollama HTTP
│  • Detects source language            │
│  • Normalizes target language name    │
│  • Summarizes document context        │
│  • Returns structured JSON            │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  STEP 4 — Save Extraction             │
│  • Writes raw text → source_extraction.txt │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  STEP 5 — Memory Handover (CRITICAL)  │
│  • keep_alive=0 → evicts LLaMA       │
│  • gc.collect() → clears Python heap  │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  STEP 6 — NLLB Phase (The Workhorse) │  ← HuggingFace local
│  • Loads NllbTokenizer + model        │
│  • Reads source_extraction.txt        │
│  • Page-by-page translation           │
│  • Live progress bar                  │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  STEP 7 — Output & Logging            │
│  • output.txt (plain text)            │
│  • output_translated.pdf (if PDF in)  │
│  • translation_log.json (session log) │
└───────────────────────────────────────┘
```

---

## 📁 File Structure

```
translate_generator/
│
├── main.py                  # CLI entry point — 7-step pipeline
├── orchestrator.py          # ModelOrchestrator class (all pipeline logic)
├── translator_engine.py     # NLLB-200 engine (lazy-loaded, isolated)
│
├── source_extraction.txt    # Inter-phase handover file (auto-generated)
├── output.txt               # Final translated text (auto-generated)
├── output_translated.pdf    # Final translated PDF (if input was PDF)
├── translation_log.json     # Session history log (auto-generated)
│
├── CHANGELOG.md
├── .gitignore
└── README.md
```

### Key Files Explained

#### `main.py`
The CLI entry point. Implements the full 7-step pipeline, handles user prompts, progress display, and file I/O. Contains no model logic — delegates everything to `ModelOrchestrator`.

#### `orchestrator.py` — `ModelOrchestrator`
The single source of truth for the pipeline. All methods are `@staticmethod`. Responsibilities:

| Method | Description |
|--------|-------------|
| `analyze_request()` | Calls LLaMA 3, parses JSON response, normalizes language names |
| `normalize_language_name()` | Strips boilerplate ("translate to X" → "x"), fuzzy-matches typos |
| `translate_pages()` | Calls NLLB page-by-page with optional progress callback |
| `extract_pdf()` | Extracts text from a PDF file path (returns doc + pages) |
| `extract_bytes()` | Extracts text from raw bytes (Streamlit uploads) |
| `extract_text()` | Chunks plain text to NLLB token limits |
| `save_extraction()` | Writes pages to `source_extraction.txt` for inter-phase handover |
| `load_extraction()` | Reads `source_extraction.txt` back into `list[dict]` format |
| `log_request()` | Appends structured session data to `translation_log.json` |

#### `translator_engine.py`
The NLLB-200 engine module. Designed for **lazy loading** — the model is never imported at module level. Exposes a clean functional API:

| Function | Description |
|----------|-------------|
| `load()` | Loads `NllbTokenizer` + `AutoModelForSeq2SeqLM` into memory |
| `unload()` | Releases model, runs `gc.collect()` + CUDA cache clear |
| `translate()` | Translates a single text chunk |
| `is_loaded()` | Returns `True` if model is currently in memory |
| `resolve_lang_code()` | Converts human name → NLLB BCP-47 code |

---

## 🚀 Installation

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.10+ | Type hints used throughout |
| [Ollama](https://ollama.com) | Latest | Must be running with `llama3` pulled |
| CUDA (optional) | 11.8+ | Falls back to CPU automatically |

### 1. Clone the repository

```bash
git clone https://github.com/your-username/omnitranslate-agent.git
cd omnitranslate-agent
```

### 2. Create virtual environment

```bash
python -m venv venv_translate
# Windows
venv_translate\Scripts\activate
# Linux / macOS
source venv_translate/bin/activate
```

### 3. Install dependencies

```bash
pip install torch transformers pymupdf langchain-community requests
```

### 4. Pull LLaMA 3 via Ollama

```bash
ollama pull llama3
```

### 5. Set HuggingFace model cache (optional)

Edit `translator_engine.py` line 16:
```python
os.environ['HF_HOME'] = r'D:\Models\huggingface'  # change to your path
```

---

## 💻 Usage

### CLI (main.py)

```bash
python main.py
```

You will be prompted through 7 steps:

```
════════════════════════════════════════════════════════════
  Translation Generator — LLaMA 3 Brain + NLLB Engine
════════════════════════════════════════════════════════════

STEP 1 — User Input Phase
  File path (PDF/TXT), or press Enter to type text directly:
  > document.pdf

  Target language? (e.g. 'arabic' or 'Translate to French')
  > arabic
```

### Programmatic Use

```python
from orchestrator import ModelOrchestrator

# Extract pages
doc, pages = ModelOrchestrator.extract_pdf("document.pdf")

# Phase 1: LLaMA analysis
analysis = ModelOrchestrator.analyze_request(
    user_request="translate to arabic",
    sample_text=pages[0]["original_text"],
    free_after=True,   # evicts LLaMA from RAM before NLLB loads
)

# Save inter-phase handover file
ModelOrchestrator.save_extraction(pages, "source_extraction.txt")

# Phase 2: NLLB translation
pages_to_translate = ModelOrchestrator.load_extraction("source_extraction.txt")
translated = ModelOrchestrator.translate_pages(
    pages_to_translate,
    source_language=analysis["detected_source_lang"],
    target_language=analysis["target_lang"],
    progress_callback=lambda done, total: print(f"{done}/{total}"),
)

# Log session
ModelOrchestrator.log_request(analysis, "document.pdf", len(pages))
doc.close()
```

---

## 🌍 Supported Languages

OmniTranslate Agent supports **200+ languages** via NLLB-200. The following are pre-mapped in `LANG_CODES`:

| Language | NLLB Code | Language | NLLB Code |
|----------|-----------|----------|-----------|
| Arabic (MSA) | `arb_Arab` | English | `eng_Latn` |
| Arabic (Egypt) | `arz_Arab` | French | `fra_Latn` |
| Arabic (Morocco) | `ary_Arab` | Spanish | `spa_Latn` |
| Arabic (Gulf) | `afb_Arab` | German | `deu_Latn` |
| Arabic (Levantine) | `apc_Arab` | Russian | `rus_Cyrl` |
| Swahili | `swh_Latn` | Chinese (Simplified) | `zho_Hans` |
| Hausa | `hau_Latn` | Japanese | `jpn_Jpan` |
| Amharic | `amh_Ethi` | Hindi | `hin_Deva` |
| Somali | `som_Latn` | Turkish | `tur_Latn` |
| Yoruba | `yor_Latn` | Persian | `pes_Arab` |

> For the full NLLB-200 language list, see the [FLORES-200 README](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).

---

## ⚙️ Configuration

| Variable | File | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_BASE_URL` | `orchestrator.py` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `orchestrator.py` | `llama3` | LLaMA model tag |
| `MODEL_NAME` | `translator_engine.py` | `facebook/nllb-200-distilled-1.3B` | NLLB model |
| `HF_HOME` | `translator_engine.py` | `D:\Models\huggingface` | HuggingFace cache path |
| `LOG_FILE` | `orchestrator.py` | `translation_log.json` | Session log path |

---

## 💾 Memory Strategy

This is the core design principle of OmniTranslate Agent:

```
LLaMA 3 (≈4 GB RAM)          NLLB-200-1.3B (≈2.5 GB RAM)
───────────────────           ─────────────────────────────
 Load → Analyze → Evict   →   Load → Translate → (keep/unload)
        ↑                              ↑
  keep_alive=0 REST call         lazy load via load()
  + gc.collect()
```

**Why not run both at once?**  
LLaMA 3 + NLLB-200 combined would require ~6.5 GB+ of RAM/VRAM. Sequential execution keeps peak usage under 4 GB, making the tool viable on 8 GB systems.

**Eviction method:**  
`taskkill` is **not** used — it would kill the entire Ollama service. Instead, a `POST /api/generate` with `keep_alive: 0` is sent, which evicts only the active model from Ollama's RAM.

---

## 📖 API Reference

### `ModelOrchestrator.analyze_request()`

```python
@staticmethod
def analyze_request(
    user_request: str,       # e.g. "translate to arabic"
    sample_text: str,        # first ~800 chars of document
    model: str = "llama3",   # Ollama model tag
    free_after: bool = True, # evict LLaMA from RAM after call
) -> dict:
    # Returns:
    # {
    #   "target_lang": "arabic",
    #   "detected_source_lang": "english",
    #   "summary": "A one-sentence document summary.",
    #   "file_status": "extracted"
    # }
```

### `ModelOrchestrator.translate_pages()`

```python
@staticmethod
def translate_pages(
    pages: list[dict],                              # page dicts with "original_text"
    source_language: str,                           # e.g. "english"
    target_language: str,                           # e.g. "arabic"
    progress_callback: Callable[[int, int], None],  # optional (done, total)
) -> list[dict]:                                    # same list with "translated_text" filled
```

### `translator_engine.translate()`

```python
def translate(
    text: str,         # raw text chunk
    source_lang: str,  # human name or NLLB BCP-47 code
    target_lang: str,  # human name or NLLB BCP-47 code
) -> str:              # translated text
```

---

## 📜 Changelog

See [CHANGELOG.md](./CHANGELOG.md) for full version history.

### v1.0.0 — 2026-04-05
- Initial release with NLLB-200 integration and LLaMA 3 orchestration
- 7-step CLI pipeline with live progress bar
- PDF extraction and PDF output via PyMuPDF
- Memory-safe sequential model execution
- Language normalization + fuzzy matching for typo tolerance
- Session logging to `translation_log.json`
- 50+ pre-mapped languages in `LANG_CODES`

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT © 2026 OmniTranslate Agent
