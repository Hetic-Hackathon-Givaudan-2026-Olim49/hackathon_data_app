"""Connecteur GDELT DOC 2.0 (gratuit, sans clé).

Doc : https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- mode=artlist, format=json ; la langue est filtrée dans la requête (`sourcelang:french`) ;
- GDELT demande au plus ~1 requête / 5 s : limiteur global + backoff si GDELT le signale
  (il répond alors HTTP 200 avec un message texte au lieu du JSON) ;
- les réponses sont mises en cache sur disque (TTL configurable).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

import httpx

from veille import settings
from veille.collect.base import NewsSource, news_id
from veille.collect.cache import DiskCache
from veille.models import Actualite

log = logging.getLogger(__name__)

_RATE_LIMIT_MARKER = "limit requests"


class GdeltError(RuntimeError):
    pass


class RateLimiter:
    """Garantit un intervalle minimal entre deux appels, partagé par tous les threads."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval_s - time.monotonic()
            if delay > 0:
                log.debug("GDELT : attente %.1f s (limite de débit).", delay)
                time.sleep(delay)
            self._last = time.monotonic()


_LIMITER = RateLimiter(settings.GDELT_MIN_INTERVAL_S)


def _parse_date(raw: str | None) -> datetime | None:
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) if raw else None
    except ValueError:
        return None


def parse_articles(payload: dict) -> list[Actualite]:
    """Convertit la réponse JSON `artlist` de GDELT en `Actualite`."""
    items = []
    for art in payload.get("articles") or []:
        url, title = art.get("url"), (art.get("title") or "").strip()
        if not url or not title:
            continue
        items.append(Actualite(
            id=news_id(url),
            titre=title,
            url=url,
            media=art.get("domain") or "inconnu",
            date=_parse_date(art.get("seendate")),
            langue=(art.get("language") or "inconnu").lower(),
            pays_media=art.get("sourcecountry") or "inconnu",
            source_api="gdelt",
        ))
    return items


class GdeltSource(NewsSource):
    name = "gdelt"

    def __init__(self, client: httpx.Client | None = None, cache: DiskCache | None = None,
                 limiter: RateLimiter | None = None, max_retries: int = 1):
        self.client = client or httpx.Client(timeout=settings.HTTP_TIMEOUT_S,
                                             headers={"User-Agent": "veille-par-impact/2.0"})
        self.cache = cache if cache is not None else DiskCache(settings.CACHE_DIR / "gdelt", ttl_s=3600)
        self.limiter = limiter or _LIMITER
        self.max_retries = max_retries

    def build_params(self, query: str, language: str, timespan: str, max_records: int) -> dict:
        full_query = f"{query} sourcelang:{language}" if language else query
        return {
            "query": full_query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(max(max_records, 1), 250)),
            "timespan": timespan,
            "sort": "datedesc",
        }

    def _fetch(self, params: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                response = self.client.get(settings.GDELT_URL, params=params)
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise GdeltError(f"GDELT injoignable : {exc}") from exc
                log.warning("GDELT : erreur réseau (%s), nouvel essai.", exc.__class__.__name__)
                continue
            text = response.text.strip()
            if response.status_code == 429 or _RATE_LIMIT_MARKER in text[:200].lower():
                backoff = settings.GDELT_MIN_INTERVAL_S * 2 ** (attempt + 1)
                if attempt == self.max_retries:
                    raise GdeltError("GDELT : limite de débit dépassée, réessayez dans une minute.")
                log.warning("GDELT : limite de débit, attente %.0f s (essai %d/%d).",
                            backoff, attempt + 1, self.max_retries)
                time.sleep(backoff)
                continue
            if response.status_code >= 400:
                raise GdeltError(f"GDELT HTTP {response.status_code} : {text[:200]}")
            if not text:
                return {"articles": []}  # GDELT renvoie un corps vide quand il n'y a aucun résultat
            try:
                return response.json()
            except ValueError as exc:
                # Requête refusée (mot trop court, syntaxe…) : GDELT répond en texte brut
                raise GdeltError(f"GDELT a refusé la requête : {text[:200]}") from exc
        raise GdeltError("GDELT : échec après plusieurs essais.")  # pragma: no cover

    def search(self, query: str, language: str, timespan: str = "7d", max_records: int = 50) -> list[Actualite]:
        params = self.build_params(query, language, timespan, max_records)
        key = DiskCache.key("gdelt", params)
        payload = self.cache.get(key)
        if payload is None:
            log.info("GDELT ← %s (%s, %s)", query[:80], language, timespan)
            payload = self._fetch(params)
            self.cache.set(key, payload)
        else:
            log.info("GDELT (cache) ← %s (%s, %s)", query[:80], language, timespan)
        items = parse_articles(payload)
        log.info("GDELT → %d articles", len(items))
        return items
