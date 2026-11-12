"""Réglages de la brique « Veille par impact ».

Tout passe par des variables d'environnement (voir `.env.example`). Un fichier `.env`
à la racine du repo est chargé s'il existe, sans écraser les variables déjà définies.

Ce module s'appelle `settings` (et non `config`) pour ne pas entrer en conflit avec
`rag/config.py`, importé par le pont vers le RAG existant.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = ROOT_DIR / "rag"
DATA_DIR = ROOT_DIR / "data"
EVAL_DIR = ROOT_DIR / "eval"

try:  # python-dotenv est optionnel : sans lui, seules les variables du shell comptent
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "oui", "on"}


def _float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _list(name: str, default: str) -> list[str]:
    return [x.strip().lower() for x in os.getenv(name, default).split(",") if x.strip()]


# --- Modèles (100 % local via Ollama) ---------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "180"))

# --- Modes ---------------------------------------------------------------------------
DEMO_MODE = _bool("DEMO_MODE")    # rejoue les sorties figées de data/demo/ (aucun réseau)
DUMMY_MODE = _bool("DUMMY_MODE")  # réponses factices conformes au contrat (aucun calcul)

# --- Collecte --------------------------------------------------------------------------
NEWS_LANGUAGES = _list("NEWS_LANGUAGES", "french")  # "french", "english", "french,english"
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "").strip() or None
GDELT_URL = os.getenv("GDELT_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
GDELT_MIN_INTERVAL_S = float(os.getenv("GDELT_MIN_INTERVAL_S", "5"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "20"))
ARTICLE_TIMEOUT_S = float(os.getenv("ARTICLE_TIMEOUT_S", "8"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", DATA_DIR / "cache"))

# --- Filtre ----------------------------------------------------------------------------
FILTER_THRESHOLD = float(os.getenv("FILTER_THRESHOLD", "0.50"))
FILTER_THRESHOLD_BY_LANG = {
    "french": _float("FILTER_THRESHOLD_FR", None),
    "english": _float("FILTER_THRESHOLD_EN", None),
}


def threshold_for(language: str | None) -> float:
    """Seuil du filtre pour une langue (surcharge par langue, sinon seuil global)."""
    override = FILTER_THRESHOLD_BY_LANG.get((language or "").lower())
    return override if override is not None else FILTER_THRESHOLD


# --- Base vectorielle (même dossier que le RAG existant, collections distinctes) ------
CHROMA_DIR = Path(os.getenv("VEILLE_CHROMA_DIR", RAG_DIR / "chroma_db"))
FAMILIES_COLLECTION = "veille_familles"
COMPANIES_COLLECTION = "veille_entreprises"

# --- Étape 5 ---------------------------------------------------------------------------
MAX_COMPANIES_PER_NEWS = int(os.getenv("MAX_COMPANIES_PER_NEWS", "5"))

# --- API -------------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))  # 8000 reste au RAG histoire (rag/api.py)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def summary() -> dict:
    """Réglages effectifs, sans aucun secret (exposés par /health)."""
    return {
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "ollama_host": OLLAMA_HOST,
        "news_languages": NEWS_LANGUAGES,
        "filter_threshold": FILTER_THRESHOLD,
        "filter_threshold_by_lang": {k: v for k, v in FILTER_THRESHOLD_BY_LANG.items() if v is not None},
        "demo_mode": DEMO_MODE,
        "dummy_mode": DUMMY_MODE,
        "gnews_enabled": GNEWS_API_KEY is not None,
    }
