"""Connecteur de repli Google News RSS (gratuit, sans clé).

Utilisé uniquement quand GDELT échoue (limite de débit, panne) pour une requête donnée.
Limites : le lien fourni est une redirection Google (le texte de l'article n'est donc
généralement pas récupérable → repli sur le titre) ; usage personnel / non commercial
selon les conditions de Google News.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx

from veille import settings
from veille.collect.base import NewsSource, news_id
from veille.collect.cache import DiskCache
from veille.models import Actualite

log = logging.getLogger(__name__)

RSS_URL = "https://news.google.com/rss/search"
_EDITIONS = {
    "french": {"hl": "fr", "gl": "FR", "ceid": "FR:fr"},
    "english": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
}


def to_when(timespan: str) -> str:
    """Convertit une fenêtre GDELT (24h, 7d, 2w, 3m, 30min) en opérateur Google `when:`."""
    match = re.fullmatch(r"(\d+)(min|h|d|w|m)", timespan)
    if not match:
        return "7d"
    n, unit = int(match.group(1)), match.group(2)
    if unit == "min":
        return "1h"
    if unit == "h":
        return f"{n}h"
    return f"{n * {'d': 1, 'w': 7, 'm': 30}[unit]}d"


def parse_rss(xml_text: str, language: str) -> list[Actualite]:
    root = ET.fromstring(xml_text)
    items = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        if not title or not link:
            continue
        source = node.find("source")
        media_name = source.text.strip() if source is not None and source.text else None
        media_url = source.get("url") if source is not None else None
        if media_name and title.endswith(f" - {media_name}"):
            title = title[: -len(media_name) - 3].strip()
        try:
            date = parsedate_to_datetime(node.findtext("pubDate") or "")
        except (TypeError, ValueError):
            date = None
        media = urlsplit(media_url).netloc.removeprefix("www.") if media_url else (media_name or "inconnu")
        items.append(Actualite(
            id=news_id(link), titre=title, url=link, media=media, date=date,
            langue=language, source_api="googlenews",
        ))
    return items


class GoogleNewsSource(NewsSource):
    name = "googlenews"

    def __init__(self, client: httpx.Client | None = None, cache: DiskCache | None = None):
        self.client = client or httpx.Client(timeout=settings.HTTP_TIMEOUT_S, follow_redirects=True,
                                             headers={"User-Agent": "Mozilla/5.0 (compatible; veille-par-impact/2.0)"})
        self.cache = cache if cache is not None else DiskCache(settings.CACHE_DIR / "googlenews", ttl_s=3600)

    def search(self, query: str, language: str, timespan: str = "7d", max_records: int = 50) -> list[Actualite]:
        edition = _EDITIONS.get(language, _EDITIONS["french"])
        params = {"q": f"{query} when:{to_when(timespan)}", **edition}
        key = DiskCache.key("googlenews", params)
        xml_text = self.cache.get(key)
        if xml_text is None:
            log.info("Google News ← %s (%s, %s)", query[:80], language, timespan)
            response = self.client.get(RSS_URL, params=params)
            response.raise_for_status()
            xml_text = response.text
            self.cache.set(key, xml_text)
        items = parse_rss(xml_text, language)[:max_records]
        log.info("Google News → %d articles", len(items))
        return items
