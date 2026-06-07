"""Central configuration. Model strings live ONLY here (PRD §1)."""
from __future__ import annotations

import os
from pathlib import Path

# Load .env (if present) so OPENAI_API_KEY is available from a local file.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv optional at runtime; env vars still work
    pass

# --- Models (do not use flagship models; cost discipline) ---
CHAT_MODEL = "gpt-4o-mini"          # vision-capable chat model
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536                    # dimensionality of text-embedding-3-small

# --- Multimodal fusion weight (text vs image caption), PRD §2.4 ---
W_TEXT = 0.5                        # 0.5/0.5 start

# --- Retry / backoff for the OpenAI wrapper ---
MAX_RETRIES = 6
BACKOFF_BASE = 0.5                  # seconds; exponential: base * 2**attempt
BACKOFF_CAP = 30.0                  # high enough to clear a low-RPM window

# --- Proactive client-side rate limiting (for tiny RPM quotas) ---
# Set OPENAI_RPM=3 to space requests ~21s apart and avoid 429 storms that would
# otherwise burn the quota. 0 (default) = no pacing.
OPENAI_RPM = float(os.environ.get("OPENAI_RPM", "0") or 0)
# spacing with a small safety margin so we stay strictly under the quota
MIN_REQUEST_INTERVAL = (60.0 / OPENAI_RPM) * 1.05 + 0.5 if OPENAI_RPM > 0 else 0.0

# --- Paths ---
PKG_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = PKG_DIR / "fixtures"
CATALOGUE_DIR = FIXTURES_DIR / "catalogue"
QUERIES_DIR = FIXTURES_DIR / "queries"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"

# Persisted artefacts
DATA_DIR = PKG_DIR / "data"
INDEX_PATH = DATA_DIR / "index.npz"          # numpy VectorIndex persistence
CHROMA_DIR = DATA_DIR / "chroma"             # Chroma persistent vector DB
CATALOGUE_JSON = DATA_DIR / "catalogue.json"  # enriched product metadata
ANALYTICS_DB = DATA_DIR / "analytics.db"      # sqlite analytics store

# Vector store backend: "chroma" (langchain-chroma, persistent) or "numpy".
# We embed once (batched) with the official OpenAI SDK and hand Chroma the
# precomputed vectors — Chroma never re-embeds, so the RPM budget is unaffected.
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "chroma")
CHROMA_COLLECTION = "catalogue"

DATA_DIR.mkdir(exist_ok=True)


def require_api_key() -> str:
    """Fail loudly if the key is missing (PRD §1). Returns the key."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file "
            "(see .env.example). This system makes REAL OpenAI calls."
        )
    return key
