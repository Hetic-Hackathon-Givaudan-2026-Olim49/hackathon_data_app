"""Récupération du texte d'un article (uniquement pour les actualités retenues par le filtre).

Extraction volontairement simple : paragraphes <p> de la balise <article> (ou de la page),
timeout court, et repli sur le titre (+ description) en cas d'échec : paywall, blocage
anti-robot, JavaScript obligatoire, etc.
"""

from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

from veille import settings
from veille.collect.cache import DiskCache
from veille.models import Actualite

log = logging.getLogger(__name__)

MAX_CHARS = 6000
MIN_PARAGRAPH = 40
_cache = DiskCache(settings.CACHE_DIR / "articles", ttl_s=7 * 24 * 3600)
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; veille-par-impact/2.0)"}


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    root = soup.find("article") or soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in root.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p) >= MIN_PARAGRAPH)
    return text[:MAX_CHARS]


def fallback_text(item: Actualite) -> str:
    return "\n".join(x for x in (item.titre, item.description) if x)


def fetch_article_text(item: Actualite) -> tuple[str, str]:
    """Retourne (texte, origine) avec origine ∈ {"article", "titre"}."""
    if "news.google.com" in item.url:  # redirection JavaScript : la page n'est pas l'article
        return fallback_text(item), "titre"
    key = DiskCache.key("article", item.url)
    cached = _cache.get(key)
    if cached:
        return cached["texte"], cached["origine"]
    try:
        response = httpx.get(item.url, timeout=settings.ARTICLE_TIMEOUT_S, follow_redirects=True, headers=_HEADERS)
        response.raise_for_status()
        text = extract_text(response.text)
    except (httpx.HTTPError, ValueError) as exc:
        log.info("Texte indisponible pour %s (%s) : repli sur le titre.", item.media, exc.__class__.__name__)
        text = ""
    if len(text) < 200:  # page vide, paywall, consentement cookies…
        result = (fallback_text(item), "titre")
    else:
        result = (text, "article")
    _cache.set(key, {"texte": result[0], "origine": result[1]})
    return result
