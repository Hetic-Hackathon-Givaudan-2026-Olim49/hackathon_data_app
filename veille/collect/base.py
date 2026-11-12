"""Interface commune des connecteurs d'actualités + outils partagés (id, normalisation)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from abc import ABC, abstractmethod
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from veille.models import Actualite

# Paramètres de suivi retirés avant de comparer deux URL
_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|xtor|at_|ref$|cmp$)", re.I)


class NewsSource(ABC):
    """Connecteur d'actualités. Toute nouvelle source (presse interne, flux RSS…) implémente `search`."""

    name: str = "abstract"

    @abstractmethod
    def search(self, query: str, language: str, timespan: str, max_records: int) -> list[Actualite]:
        """Retourne les actualités correspondant à `query` dans `language` sur `timespan` (ex. 7d)."""


def normalize_url(url: str) -> str:
    """Forme canonique d'une URL : schéma/hôte en minuscules, sans www, sans suivi, sans fragment."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.").removeprefix("m.")
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k)])
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, query, ""))


def normalize_title(title: str) -> str:
    """Titre comparable : minuscules, sans accents ni ponctuation, sans suffixe « - Média »."""
    title = re.split(r"\s[|\-–—]\s(?=[^|\-–—]{2,40}$)", title.strip())[0]  # « Titre - Le Monde »
    title = "".join(c for c in unicodedata.normalize("NFD", title) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", title.lower())).strip()


def news_id(url: str) -> str:
    return hashlib.sha1(normalize_url(url).encode()).hexdigest()[:16]


def deduplicate(items: list[Actualite]) -> list[Actualite]:
    """Supprime les doublons par URL normalisée ET par titre normalisé (on garde la 1re occurrence)."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[Actualite] = []
    for item in items:
        url_key = normalize_url(item.url)
        title_key = normalize_title(item.titre)
        if url_key in seen_urls or (title_key and title_key in seen_titles):
            continue
        seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        unique.append(item)
    return unique
