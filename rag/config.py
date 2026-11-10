"""Configuration partagée par les scripts (surcharge possible via variables d'environnement)."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Base vectorielle (ChromaDB persistante sur disque)
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "story")

# Source et découpage
STORY_PATH = Path(os.getenv("STORY_PATH", BASE_DIR / "story.txt"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # en caractères
CHUNK_OVERLAP_PARAGRAPHS = int(os.getenv("CHUNK_OVERLAP_PARAGRAPHS", "1"))

# Recherche
TOP_K = int(os.getenv("TOP_K", "4"))
