"""
Central settings for the RAG chatbot.

Every other module imports its settings from here, so there is one place to change
model names, chunk sizes, etc. Secrets (API keys) are read from `.env`, never hardcoded.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder above src/. Using an absolute path means `.env` is found
# no matter which folder you run the script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# LLM model registry
# Adding a model = adding one entry here. (Later, the UI model dropdown is built from this dict.)
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-oss-20b (Groq, hosted)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-20b",
        "api_key_env": "GROQ_API_KEY",
    },
    "GPT-6 Luna (OpenAI)": {
        "base_url": None,               # None = OpenAI default
        "model": "gpt-6-luna",
        "api_key_env": "OPENAI_API_KEY",
        # OpenAI's reasoning models reject custom temperatures (only the default, 1).
        # Optional flag; entries without it use LLM_TEMPERATURE.
        "supports_temperature": False,
    },
}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "GPT-6 Luna (OpenAI)")

# Low temperature = less creative, more factual answers.
LLM_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# PDF ingestion / OCR
# ---------------------------------------------------------------------------
OCR_MIN_CHARS = 50   # pages with less text than this are treated as scanned images
OCR_DPI = 300        # higher DPI = better OCR accuracy, but slower

# ---------------------------------------------------------------------------
# Chunking + retrieval
# ---------------------------------------------------------------------------
# LlamaIndex's SentenceSplitter counts TOKENS, not characters (1 token ≈ 4 characters),
# and table-like SDF text is token-dense, so 200 tokens ≈ 450 characters of text.
CHUNK_SIZE = 200      # tokens per chunk
CHUNK_OVERLAP = 40    # overlap keeps facts that span a chunk boundary findable
DEFAULT_K = 6         # how many chunks to retrieve per question (6 fits multi-batch answers)

# Re-ranking: vector search fetches a bigger pool of candidates, then a cross-encoder
# (which reads the question and each chunk together, so it judges relevance better than
# comparing two embeddings) picks the best DEFAULT_K of them for the LLM.
USE_RERANKER = True
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = 20  # chunks the reranker scores per question

# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------
CLASSIFIER_HEADER_CHARS = 400  # the "title area" at the top of a page, where keywords count double
USE_LLM_CLASSIFIER = True      # ask the LLM to label pages the keyword rules can't place

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_PDF = DATA_DIR / "sample-sdf-document (Project 6).pdf"


def get_api_key(model_name: str) -> str | None:
    """Return the API key for a model, or None if it is missing from `.env`."""
    key_env = MODELS[model_name]["api_key_env"]
    # Treat empty strings and the .env.example placeholders as "not set".
    key = os.getenv(key_env, "").strip()
    if not key or "your_key_here" in key:
        return None
    return key


if __name__ == "__main__":
    # Quick check: prints the settings and which models have keys configured.
    print("=== Settings ===")
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Sample PDF:       {SAMPLE_PDF} (exists: {SAMPLE_PDF.exists()})")
    print(f"Embedding model:  {EMBEDDING_MODEL}")
    print(f"Chunk size:       {CHUNK_SIZE} (overlap {CHUNK_OVERLAP})")
    print(f"Default k:        {DEFAULT_K}")
    print(f"Reranker:         {RERANK_MODEL if USE_RERANKER else 'off'} "
          f"({RERANK_CANDIDATES} candidates -> top {DEFAULT_K})")
    print(f"OCR threshold:    {OCR_MIN_CHARS} chars, {OCR_DPI} DPI")
    print(f"LLM temperature:  {LLM_TEMPERATURE}")
    print(f"Default model:    {DEFAULT_MODEL}")
    if DEFAULT_MODEL not in MODELS:
        print("  WARNING: DEFAULT_MODEL in .env doesn't match any entry in MODELS (typo?)")

    print("\n=== Models ===")
    for name, cfg in MODELS.items():
        # Only say "set"/"missing" — never print the key itself.
        if get_api_key(name):
            key_status = f"{cfg['api_key_env']} set"
        else:
            key_status = f"{cfg['api_key_env']} missing"
        print(f"- {name}")
        print(f"    model: {cfg['model']} | base_url: {cfg['base_url'] or 'OpenAI default'} | {key_status}")
