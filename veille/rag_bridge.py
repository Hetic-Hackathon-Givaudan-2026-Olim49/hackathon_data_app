"""Pont vers le RAG existant (`rag/`), réutilisé au lieu d'être dupliqué.

On réutilise :
- le client ChromaDB persistant (`rag.get_chroma_client`) et donc le même dossier
  `rag/chroma_db/`, dans des collections distinctes de la collection `story` ;
- le client Ollama (`rag._ollama`) pour les embeddings.

Différence assumée : les nouvelles collections utilisent `settings.EMBED_MODEL`
(bge-m3, multilingue) au lieu de `nomic-embed-text`. La collection `story` et
`rag/api.py` ne sont pas touchés et restent sur leur propre modèle.

Le module `rag/rag.py` fait `import config` : on ajoute donc `rag/` au `sys.path`
et on le charge sous le nom `rag_story` pour éviter toute collision avec le dossier
`rag/` vu comme paquet depuis la racine.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from functools import lru_cache
from types import ModuleType

from veille import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _rag() -> ModuleType:
    rag_dir = str(settings.RAG_DIR)
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)
    spec = importlib.util.spec_from_file_location("rag_story", settings.RAG_DIR / "rag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def chroma_client():
    """Client ChromaDB du RAG existant, pointé sur `settings.CHROMA_DIR`."""
    import chromadb

    if settings.CHROMA_DIR == settings.RAG_DIR / "chroma_db":
        return _rag().get_chroma_client()
    return chromadb.PersistentClient(path=str(settings.CHROMA_DIR))


def get_collection(name: str):
    """Collection cosinus étiquetée avec le modèle d'embeddings utilisé."""
    collection = chroma_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "embed_model": settings.EMBED_MODEL},
    )
    stored = (collection.metadata or {}).get("embed_model")
    if stored and stored != settings.EMBED_MODEL:
        log.warning(
            "La collection '%s' a été indexée avec '%s' mais EMBED_MODEL='%s' : "
            "réindexer (vecteurs incompatibles).", name, stored, settings.EMBED_MODEL,
        )
    return collection


def embed(texts: list[str]) -> list[list[float]]:
    """Embeddings via le client Ollama du RAG existant, avec `settings.EMBED_MODEL`.

    bge-m3 n'utilise pas de préfixe requête/document (contrairement à nomic-embed-text).
    """
    if not texts:
        return []
    prefix = "search_document: " if "nomic" in settings.EMBED_MODEL else ""
    response = _rag()._ollama.embed(model=settings.EMBED_MODEL, input=[prefix + t for t in texts])
    return response["embeddings"]
