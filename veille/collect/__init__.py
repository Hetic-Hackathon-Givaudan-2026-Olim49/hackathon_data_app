"""Étape 1 — collecte : une requête par famille et par langue, fusion, déduplication.

Stratégie de sources :
- GDELT est la source principale ;
- si GDELT échoue pour une requête (limite de débit, panne), Google News RSS prend le relais
  pour CETTE requête (même interface, même cache, même déduplication) ;
- GNews (si GNEWS_API_KEY est défini) s'ajoute aux résultats pour enrichir les descriptions.
"""

from __future__ import annotations

import logging

from veille import settings
from veille.collect.base import NewsSource, deduplicate
from veille.models import Actualite

log = logging.getLogger(__name__)


def default_queries(language: str) -> list[str]:
    """Requêtes par défaut des 4 familles (data/families.json) pour une langue."""
    from veille.filter import HORS_PERIMETRE, load_families

    return [spec["requetes_gdelt"][language] for name, spec in load_families().items()
            if name != HORS_PERIMETRE and language in spec.get("requetes_gdelt", {})]


def get_sources() -> tuple[NewsSource, NewsSource | None, list[NewsSource]]:
    """(principale, repli, complémentaires)."""
    from veille.collect.gdelt import GdeltSource
    from veille.collect.googlenews import GoogleNewsSource

    extra: list[NewsSource] = []
    if settings.GNEWS_API_KEY:
        try:
            from veille.collect.gnews import GNewsSource

            extra.append(GNewsSource(settings.GNEWS_API_KEY))
        except ImportError:
            log.info("Connecteur GNews non disponible dans cette version.")
    return GdeltSource(), GoogleNewsSource(), extra


def _search(source: NewsSource, q: str, language: str, timespan: str, max_records: int) -> list[Actualite]:
    return source.search(q, language, timespan, max_records)


def collect(query: str | None, languages: list[str], timespan: str, max_records: int,
            sources: tuple[NewsSource, NewsSource | None, list[NewsSource]] | None = None,
            ) -> tuple[list[Actualite], list[str]]:
    """Retourne (actualités dédupliquées, erreurs/avertissements). Une panne n'arrête pas la collecte."""
    primary, fallback, extra = sources if sources is not None else get_sources()
    items: list[Actualite] = []
    errors: list[str] = []
    for language in languages:
        for q in ([query] if query else default_queries(language)):
            try:
                items.extend(_search(primary, q, language, timespan, max_records))
            except Exception as exc:  # limite de débit, réseau, requête refusée…
                log.warning("Collecte %s (%s) en échec : %s", primary.name, language, exc)
                errors.append(f"{primary.name}/{language} : {exc}")
                if fallback is not None:
                    try:
                        items.extend(_search(fallback, q, language, timespan, max_records))
                        errors[-1] += f" → repli {fallback.name} OK"
                    except Exception as exc2:
                        log.error("Repli %s (%s) en échec : %s", fallback.name, language, exc2)
                        errors.append(f"{fallback.name}/{language} : {exc2}")
            for source in extra:
                try:
                    items.extend(_search(source, q, language, timespan, max_records))
                except Exception as exc:
                    log.warning("Collecte %s (%s) en échec : %s", source.name, language, exc)
                    errors.append(f"{source.name}/{language} : {exc}")
    unique = deduplicate(items)
    log.info("Collecte : %d articles, %d après déduplication.", len(items), len(unique))
    return unique, errors
