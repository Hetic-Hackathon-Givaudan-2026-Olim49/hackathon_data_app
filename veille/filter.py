"""Étape 2 — filtre par similarité d'embeddings (aucun LLM génératif).

Pour chaque actualité : similarité cosinus entre l'embedding du titre (+ description si
disponible) et les descriptions de référence des 4 familles (data/families.json).
Score d'une famille = maximum sur ses descriptions, dans les langues activées.
L'actualité est retenue si le meilleur score dépasse le seuil de sa langue ET le score
des contre-exemples `hors_perimetre` (tech, sport, politique, énergie…).

Les embeddings de référence sont stockés dans la collection ChromaDB `veille_familles`
(réutilisation du RAG existant) et recalculés seulement si le texte change.

    python -m veille.filter --reindex          # recalcule les références
    python -m veille.filter "Titre à tester"   # affiche les scores
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from functools import lru_cache
from typing import Callable

import numpy as np

from veille import settings
from veille.models import FAMILLES, Actualite, ResultatFiltre

log = logging.getLogger(__name__)

FAMILIES_PATH = settings.DATA_DIR / "families.json"

Embedder = Callable[[list[str]], list[list[float]]]


HORS_PERIMETRE = "hors_perimetre"


def load_families() -> dict:
    """Les 4 familles + les contre-exemples `hors_perimetre`."""
    data = json.loads(FAMILIES_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k in FAMILLES or k == HORS_PERIMETRE}


def reference_rows(families: dict, languages: list[str]) -> list[tuple[str, str, str, str]]:
    """(id, famille, langue, texte) pour chaque description des langues demandées."""
    rows = []
    for famille, spec in families.items():
        for lang, texts in spec["descriptions"].items():
            if lang not in languages:
                continue
            for i, text in enumerate(texts):
                rows.append((f"{famille}-{lang}-{i}", famille, lang, text))
    return rows


class FamilyFilter:
    """Filtre réutilisable ; l'embedder et le stockage sont injectables (tests)."""

    def __init__(self, references: list[tuple[str, str, list[float]]], embedder: Embedder):
        # references : (famille, langue, vecteur)
        if not references:
            raise ValueError("Aucune description de référence pour les langues demandées.")
        self.families = [r[0] for r in references]
        matrix = np.asarray([r[2] for r in references], dtype=np.float32)
        self.matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        self.embedder = embedder

    @staticmethod
    def text_of(item: Actualite) -> str:
        return f"{item.titre}\n{item.description}" if item.description else item.titre

    def score_vectors(self, vectors: list[list[float]]) -> list[dict[str, float]]:
        v = np.asarray(vectors, dtype=np.float32)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        sims = v @ self.matrix.T  # (n_actualites, n_references)
        out = []
        for row in sims:
            scores: dict[str, float] = {}
            for famille, s in zip(self.families, row):
                scores[famille] = max(scores.get(famille, -1.0), float(s))
            out.append({k: round(v, 4) for k, v in scores.items()})
        return out

    def apply(self, items: list[Actualite]) -> list[ResultatFiltre]:
        if not items:
            return []
        vectors = self.embedder([self.text_of(i) for i in items])
        results = []
        for item, scores in zip(items, self.score_vectors(vectors)):
            hors = scores.get(HORS_PERIMETRE, -1.0)
            familles = {k: v for k, v in scores.items() if k != HORS_PERIMETRE}
            famille, score = max(familles.items(), key=lambda kv: kv[1])
            seuil = settings.threshold_for(item.langue)
            results.append(ResultatFiltre(
                retenue=score >= seuil and score > hors, famille=famille, score=round(score, 4),
                seuil=seuil, scores_par_famille=scores,
            ))
        return results


def _text_hash(text: str) -> str:
    return hashlib.sha1(f"{settings.EMBED_MODEL}|{text}".encode()).hexdigest()[:16]


def sync_references(languages: list[str], force: bool = False) -> list[tuple[str, str, list[float]]]:
    """Charge (et met à jour si besoin) les embeddings de référence dans ChromaDB."""
    from veille import rag_bridge

    rows = reference_rows(load_families(), languages)
    collection = rag_bridge.get_collection(settings.FAMILIES_COLLECTION)
    ids = [r[0] for r in rows]
    existing = collection.get(ids=ids, include=["metadatas", "embeddings"]) if ids else {"ids": []}
    metadatas = existing.get("metadatas")
    embeddings = existing.get("embeddings")  # tableau numpy : pas de test de vérité direct
    known = {
        i: (metadatas[n] or {}, embeddings[n]) for n, i in enumerate(existing["ids"])
    } if metadatas is not None and embeddings is not None else {}
    stale = [r for r in rows if force or r[0] not in known or known[r[0]][0].get("hash") != _text_hash(r[3])]
    if stale:
        log.info("Filtre : calcul de %d embeddings de référence (%s).", len(stale), settings.EMBED_MODEL)
        vectors = rag_bridge.embed([r[3] for r in stale])
        collection.upsert(
            ids=[r[0] for r in stale],
            documents=[r[3] for r in stale],
            embeddings=vectors,
            metadatas=[{"famille": r[1], "langue": r[2], "hash": _text_hash(r[3])} for r in stale],
        )
        for r, vec in zip(stale, vectors):
            known[r[0]] = ({"famille": r[1], "langue": r[2]}, vec)
    return [(r[1], r[2], list(known[r[0]][1])) for r in rows]


@lru_cache(maxsize=4)
def get_filter(languages: tuple[str, ...]) -> FamilyFilter:
    from veille import rag_bridge

    return FamilyFilter(sync_references(list(languages)), rag_bridge.embed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Teste le filtre de l'étape 2.")
    parser.add_argument("titres", nargs="*", help="titres à évaluer")
    parser.add_argument("--reindex", action="store_true", help="recalcule les embeddings de référence")
    parser.add_argument("--langue", default=None, help="langue des titres (défaut : 1re de NEWS_LANGUAGES)")
    args = parser.parse_args()

    languages = tuple(settings.NEWS_LANGUAGES)
    if args.reindex:
        sync_references(list(languages), force=True)
    filt = get_filter(languages)
    from veille.collect.base import news_id

    lang = args.langue or languages[0]
    items = [Actualite(id=news_id(f"https://cli/{i}"), titre=t, url=f"https://cli/{i}", langue=lang)
             for i, t in enumerate(args.titres)]
    for item, res in zip(items, filt.apply(items)):
        flag = "RETENUE" if res.retenue else "écartée"
        hors = res.scores_par_famille.get(HORS_PERIMETRE, 0)
        print(f"[{flag}] {res.score:.3f} (hors {hors:.3f}) {res.famille:<32} {item.titre}")


if __name__ == "__main__":
    main()
