"""
translator_engine.py
--------------------
NLLB Translation Engine — the pure translation workhorse.

Design: LAZY LOADING
  - Model is NOT loaded at import time.
  - Call load() explicitly after LLaMA 3 has been freed from memory.
  - This prevents both models from occupying RAM at the same time.
"""

import os
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

os.environ['HF_HOME'] = r'D:\Models\huggingface'

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

# ─── NLLB-200 language code map ──────────────────────────────────────────────
# Verified against the official NLLB-200 language list (Meta AI, 2022).
# Source: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
# Format: ISO-639-3 code + "_" + ISO-15924 script code
LANG_CODES: dict[str, str] = {
    # ── Arabic varieties ──────────────────────────────────────────────────────
    "arabic":                  "arb_Arab",   # Modern Standard Arabic  ← was WRONG (ara_Arab)
    "arabic (egypt)":          "arz_Arab",   # Egyptian Arabic
    "arabic (morocco)":        "ary_Arab",   # Moroccan Arabic
    "arabic (gulf)":           "afb_Arab",   # Gulf Arabic
    "arabic (levantine)":      "apc_Arab",   # North Levantine Arabic
    "arabic (mesopotamian)":   "acm_Arab",   # Iraqi Arabic
    "arabic (najdi)":          "ars_Arab",   # Najdi Arabic (Saudi)
    "arabic (tunisian)":       "aeb_Arab",   # Tunisian Arabic
    # ── Major world languages ─────────────────────────────────────────────────
    "english":                 "eng_Latn",
    "french":                  "fra_Latn",
    "spanish":                 "spa_Latn",
    "portuguese":              "por_Latn",
    "german":                  "deu_Latn",
    "italian":                 "ita_Latn",
    "dutch":                   "nld_Latn",
    "russian":                 "rus_Cyrl",
    "ukrainian":               "ukr_Cyrl",
    "polish":                  "pol_Latn",
    "chinese":                 "zho_Hans",   # Simplified Chinese
    "chinese (simplified)":    "zho_Hans",
    "chinese (traditional)":   "zho_Hant",
    "japanese":                "jpn_Jpan",
    "korean":                  "kor_Hang",
    "turkish":                 "tur_Latn",
    "persian":                 "pes_Arab",   # Western Persian / Farsi
    "hindi":                   "hin_Deva",
    "urdu":                    "urd_Arab",
    "bengali":                 "ben_Beng",
    "thai":                    "tha_Thai",
    "vietnamese":              "vie_Latn",
    "indonesian":              "ind_Latn",
    "malay":                   "zsm_Latn",
    "greek":                   "ell_Grek",
    "hebrew":                  "heb_Hebr",
    "czech":                   "ces_Latn",
    "swedish":                 "swe_Latn",
    "norwegian":               "nob_Latn",
    "danish":                  "dan_Latn",
    "finnish":                 "fin_Latn",
    "hungarian":               "hun_Latn",
    "romanian":                "ron_Latn",
    "catalan":                 "cat_Latn",
    # ── African languages ─────────────────────────────────────────────────────
    "hausa":                   "hau_Latn",
    "yoruba":                  "yor_Latn",
    "igbo":                    "ibo_Latn",
    "swahili":                 "swh_Latn",
    "somali":                  "som_Latn",
    "amharic":                 "amh_Ethi",
    "zulu":                    "zul_Latn",
    "xhosa":                   "xho_Latn",
    "shona":                   "sna_Latn",
}


def resolve_lang_code(lang_name: str) -> str:
    """
    Converts a human-readable language name to its NLLB code.
    Falls back to the input string as-is if not found (allows raw codes like 'hau_Latn').
    """
    return LANG_CODES.get(lang_name.strip().lower(), lang_name.strip())


def _get_lang_token_id(tokenizer, lang_code: str) -> int:
    """
    Resolve an NLLB BCP-47 language code to its token ID.

    Tries four methods in order (most reliable → least):
      1. lang_code_to_id       — NllbTokenizer (slow/Python) ← populated with use_fast=False
      2. added_tokens_encoder  — NllbTokenizerFast (Rust)
      3. get_vocab()           — full vocabulary scan (universal)
      4. convert_tokens_to_ids — last-resort generic call
    """
    # 1. Slow tokenizer attribute (populated when use_fast=False)
    lang_map = getattr(tokenizer, "lang_code_to_id", {})
    if lang_map and lang_code in lang_map:
        return lang_map[lang_code]

    # 2. Fast tokenizer added_tokens_encoder
    added = getattr(tokenizer, "added_tokens_encoder", {})
    if lang_code in added:
        return added[lang_code]

    # 3. Full vocabulary scan — works for any tokenizer variant
    vocab = tokenizer.get_vocab()
    if lang_code in vocab:
        return vocab[lang_code]

    # 4. Generic API fallback
    token_id = tokenizer.convert_tokens_to_ids(lang_code)
    if token_id != tokenizer.unk_token_id:
        return token_id

    # All methods failed — build a helpful error message from available vocab
    all_codes = set(lang_map.keys()) | set(added.keys()) | set(vocab.keys())
    nllb_codes = sorted(
        c for c in all_codes
        if len(c) >= 7 and "_" in c and c[3] == "_"
    )
    hint = ", ".join(nllb_codes[:20]) if nllb_codes else "(could not read tokenizer vocab)"
    raise ValueError(
        f"Unknown NLLB language code: '{lang_code}'.\n"
        f"Sample of valid codes: {hint}\n"
        "Check the LANG_CODES table in translator_engine.py or supply a raw BCP-47 code."
    )


# ─── Internal state (None until load() is called) ────────────────────────────
_tokenizer = None
_model = None


# ─── Public API ───────────────────────────────────────────────────────────────
def is_loaded() -> bool:
    """Returns True if the NLLB model is in memory."""
    return _model is not None


def load() -> None:
    """
    Explicitly load the NLLB model into memory.
    Call this AFTER LLaMA 3 has been freed to avoid RAM contention.
    Safe to call multiple times — does nothing if already loaded.

    Uses NllbTokenizer directly (not AutoTokenizer) so that get_vocab()
    returns the full vocabulary including all 200 language tokens.
    """
    global _tokenizer, _model
    if _model is not None:
        return  # already loaded

    print(f"[NLLB Engine] Loading {MODEL_NAME} ...")

    # NllbTokenizer (not AutoTokenizer) — exposes language codes via get_vocab()
    _tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)

    _model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,        # dtype= (not torch_dtype=, which is deprecated)
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    print("[NLLB Engine] Model loaded successfully.\n")

def unload() -> None:
    """
    Release the NLLB model from memory and clear GPU/CPU caches.
    """
    import gc
    global _tokenizer, _model
    _tokenizer = None
    _model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[NLLB Engine] Model unloaded and memory cleared.")


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate `text` from `source_lang` to `target_lang`.
    Raises RuntimeError if load() has not been called first.

    Parameters
    ----------
    text        : The text to translate.
    source_lang : Human-readable name or NLLB code for the source language.
    target_lang : Human-readable name or NLLB code for the target language.
    """
    if _model is None:
        raise RuntimeError(
            "NLLB model is not loaded. Call translator_engine.load() first."
        )

    src_code = resolve_lang_code(source_lang).strip()
    tgt_code = resolve_lang_code(target_lang).strip()

    # 1. Set source language and tokenize
    _tokenizer.src_lang = src_code
    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(_model.device)

    # 2. Get target language token ID via full vocab scan
    #    (only method that works for both NllbTokenizer and NllbTokenizerFast)
    vocab = _tokenizer.get_vocab()
    if tgt_code not in vocab:
        raise ValueError(
            f"Unknown NLLB language code: '{tgt_code}'.\n"
            "Check LANG_CODES table in translator_engine.py or pass a valid NLLB BCP-47 code."
        )
    target_lang_id = vocab[tgt_code]

    # 3. Generate translation
    translated_tokens = _model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id,
        max_length=512,
        num_beams=4,
    )
    return _tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

#____ for testing ____
if __name__ == "__main__":
    load()
    print(translate("Hello, how are you today?", "English", "Arabic"))
    unload()
    
