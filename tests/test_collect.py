"""Tests du connecteur (étape 1) : parsing GDELT / Google News, déduplication, limite de débit, repli."""

import time

import httpx
import pytest

from veille.collect import collect
from veille.collect.base import NewsSource, deduplicate, normalize_title, normalize_url
from veille.collect.cache import DiskCache
from veille.collect.gdelt import GdeltError, GdeltSource, RateLimiter, parse_articles
from veille.collect.googlenews import parse_rss, to_when
from veille.models import Actualite

GDELT_PAYLOAD = {"articles": [
    {"url": "https://www.example.fr/a?utm_source=x", "title": "Le prix du cacao s'envole", "seendate": "20261110T081500Z",
     "domain": "example.fr", "language": "French", "sourcecountry": "France"},
    {"url": "https://example.fr/sans-titre", "title": ""},
]}


def _item(url, titre):
    return Actualite(id=url, titre=titre, url=url)


def test_parse_articles():
    items = parse_articles(GDELT_PAYLOAD)
    assert len(items) == 1  # l'article sans titre est ignoré
    a = items[0]
    assert a.media == "example.fr" and a.langue == "french" and a.date.year == 2026 and a.source_api == "gdelt"


def test_normalize_url_and_title():
    assert normalize_url("http://www.Example.fr/a/?utm_source=x&id=3#top") == "https://example.fr/a?id=3"
    assert normalize_title("Le prix du Cacao s'envole ! - Le Monde") == "le prix du cacao s envole"


def test_deduplicate_by_url_and_title():
    items = [
        _item("https://example.fr/a", "Le prix du cacao s'envole"),
        _item("https://www.example.fr/a/?utm_medium=rss", "Autre titre"),          # même URL
        _item("https://autre.fr/b", "Le prix du cacao s'envole - Autre Média"),     # même titre
        _item("https://autre.fr/c", "Un autre sujet"),
    ]
    assert [i.url for i in deduplicate(items)] == ["https://example.fr/a", "https://autre.fr/c"]


def test_rate_limiter_spacing():
    limiter = RateLimiter(0.2)
    start = time.monotonic()
    for _ in range(3):
        limiter.wait()
    assert time.monotonic() - start >= 0.4


def _gdelt(handler, tmp_path, retries=1):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GdeltSource(client=client, cache=DiskCache(tmp_path), limiter=RateLimiter(0), max_retries=retries)


def test_gdelt_search_uses_language_and_cache(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=GDELT_PAYLOAD)

    source = _gdelt(handler, tmp_path)
    assert len(source.search("cacao", "french", "7d", 10)) == 1
    assert "sourcelang:french" in calls[0].url.params["query"]
    assert calls[0].url.params["mode"] == "artlist" and calls[0].url.params["format"] == "json"
    source.search("cacao", "french", "7d", 10)
    assert len(calls) == 1  # deuxième appel servi par le cache


def test_gdelt_rate_limit_message_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("veille.collect.gdelt.time.sleep", lambda s: None)
    source = _gdelt(lambda r: httpx.Response(200, text="Please limit requests to one every 5 seconds"), tmp_path)
    with pytest.raises(GdeltError):
        source.search("cacao", "french")


def test_gdelt_empty_body_means_no_result(tmp_path):
    assert _gdelt(lambda r: httpx.Response(200, text=""), tmp_path).search("cacao", "french") == []


RSS = """<rss><channel><item><title>Vanille : les prix flambent - Le Média</title>
<link>https://news.google.com/rss/articles/abc</link><pubDate>Tue, 10 Nov 2026 08:00:00 GMT</pubDate>
<source url="https://www.lemedia.fr">Le Média</source></item></channel></rss>"""


def test_parse_google_news_rss():
    [item] = parse_rss(RSS, "french")
    assert item.titre == "Vanille : les prix flambent" and item.media == "lemedia.fr"
    assert item.source_api == "googlenews" and item.date.day == 10


def test_to_when():
    assert to_when("7d") == "7d" and to_when("2w") == "14d" and to_when("24h") == "24h" and to_when("x") == "7d"


class _Fixed(NewsSource):
    def __init__(self, name, items=None, fail=False):
        self.name, self.items, self.fail = name, items or [], fail

    def search(self, query, language, timespan, max_records):
        if self.fail:
            raise RuntimeError("limite de débit")
        return list(self.items)


def test_collect_falls_back_when_primary_fails():
    fallback = _Fixed("secours", [_item("https://x.fr/1", "Titre un")])
    items, errors = collect("cacao", ["french"], "7d", 10, sources=(_Fixed("gdelt", fail=True), fallback, []))
    assert [i.url for i in items] == ["https://x.fr/1"]
    assert "repli secours OK" in errors[0]
