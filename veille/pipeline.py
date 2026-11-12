"""Orchestration du pipeline (étapes 1 à 5) et assemblage des fiches finales."""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup
from fastapi import HTTPException

from veille import analyze as step3
from veille import demo, match as step4, opportunities as step5, settings
from veille.collect import collect
from veille.collect.article import fetch_article_text
from veille.collect.base import news_id
from veille.filter import get_filter, load_families
from veille.llm import LLMError, LLMStats, LLMUnavailable
from veille.models import (
    Actualite,
    AnalyzeRequest,
    AnalyzeResponse,
    Diagnostic,
    EntrepriseConcernee,
    FicheOpportunite,
    HypotheseOpportunite,
    Mode,
    OpportunitiesResponse,
    ResultatFiltre,
    Signal,
    Source,
    Analyse,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
# Entrée de /analyze
# ----------------------------------------------------------------------------------------
def _media_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.") or "inconnu"


def _title_from_url(url: str) -> str:
    try:
        response = httpx.get(url, timeout=settings.ARTICLE_TIMEOUT_S, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0 (compatible; veille-par-impact/2.0)"})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Impossible de récupérer l'URL ({exc.__class__.__name__}). "
                                 "Fournissez plutôt `actualite` avec le titre.") from exc
    soup = BeautifulSoup(response.text, "html.parser")
    og = soup.find("meta", property="og:title")
    title = (og.get("content") if og else None) or (soup.title.string if soup.title else None)
    if not title or len(title.strip()) < 3:
        raise HTTPException(422, "Titre introuvable sur la page. Fournissez `actualite` avec le titre.")
    return title.strip()


def build_item(req: AnalyzeRequest) -> tuple[Actualite, str | None]:
    """(actualité, texte fourni éventuel)."""
    if req.actualite:
        a = req.actualite
        url = str(a.url)
        return Actualite(
            id=news_id(url), titre=a.titre, url=url, media=a.media or _media_of(url), date=a.date,
            langue=(a.langue or settings.NEWS_LANGUAGES[0]).lower(), source_api="manuel",
        ), a.texte
    url = str(req.url)
    return Actualite(
        id=news_id(url), titre=_title_from_url(url), url=url, media=_media_of(url),
        langue=settings.NEWS_LANGUAGES[0], source_api="manuel",
    ), None


def _filter_languages(extra: list[str] | None = None) -> tuple[str, ...]:
    available = {lang for name, spec in load_families().items() for lang in spec["descriptions"]}
    langs = list(dict.fromkeys(settings.NEWS_LANGUAGES + (extra or [])))
    return tuple(sorted(l for l in langs if l in available)) or ("french",)


# ----------------------------------------------------------------------------------------
# Fiches
# ----------------------------------------------------------------------------------------
def _slug(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def build_fiches(item: Actualite, filtre: ResultatFiltre, analysis: Analyse,
                 companies: list[EntrepriseConcernee], opps: list[HypotheseOpportunite]) -> list[FicheOpportunite]:
    by_name = {c.nom: c for c in companies}
    fiches = []
    for opp in opps:
        company = by_name[opp.entreprise]
        fiches.append(FicheOpportunite(
            id=f"{item.id}--{_slug(company.nom)}",
            signal=Signal(titre=item.titre, media=item.media, date=item.date, url=item.url),
            famille=filtre.famille,
            score_filtre=filtre.score,
            ce_que_ca_implique=analysis.hypotheses.resume,
            faits=analysis.faits,
            chaine_impact=analysis.hypotheses.chaine_impact,
            entreprise=company,
            opportunite=opp,
            question_a_poser=opp.question_client,
            confiance=opp.confiance,
            sources=[
                Source(type="article", libelle=f"{item.media} — {item.titre}", url=item.url),
                Source(type="base_entreprises", libelle=f"Référence {company.nom} (Open Food/Beauty Facts)",
                       url=company.source),
            ],
        ))
    return fiches


def select_diverse(kept: list[tuple[Actualite, ResultatFiltre]], n: int) -> list[tuple[Actualite, ResultatFiltre]]:
    """Les n meilleures actualités en alternant les familles (la meilleure de chaque famille d'abord),
    pour qu'une famille très représentée (ex. tendances parfum) n'occupe pas toutes les analyses."""
    by_family: dict[str, list] = {}
    for item, filtre in sorted(kept, key=lambda x: -x[1].score):
        by_family.setdefault(filtre.famille or "", []).append((item, filtre))
    queues = sorted(by_family.values(), key=lambda q: -q[0][1].score)
    selected = []
    while len(selected) < n and any(queues):
        for queue in queues:
            if queue and len(selected) < n:
                selected.append(queue.pop(0))
    return selected


def cap_confidence(value: str) -> str:
    return "moyen" if value == "élevé" else value


# ----------------------------------------------------------------------------------------
# Étapes 3 → 5 pour une actualité retenue
# ----------------------------------------------------------------------------------------
def run_on_item(item: Actualite, filtre: ResultatFiltre, text: str | None, reference: list[dict],
                diag: Diagnostic) -> tuple[Analyse | None, list[EntrepriseConcernee], list[FicheOpportunite]]:
    stats = LLMStats()
    try:
        if text is None:
            text, origine = fetch_article_text(item)
            log.info("Texte de « %s » : %s (%d car.)", item.titre[:60], origine, len(text))
        else:
            origine = "fourni"
        diag.texte_analyse = origine
        analysis = step3.analyze(item, filtre.famille, text, stats, diag.rejets_validation)
        companies = step4.match(analysis.hypotheses, reference, limit=settings.MAX_COMPANIES_PER_NEWS,
                                title=item.titre, article_url=item.url, famille=filtre.famille)
        log.info("Jointure : %d entreprise(s) pour « %s »", len(companies), item.titre[:60])
        opps = step5.hypothesize(item.titre, analysis, companies, reference, stats, diag.rejets_validation)
        if origine == "titre":  # règle déterministe : un titre seul ne justifie jamais une confiance élevée
            analysis.hypotheses.confiance = cap_confidence(analysis.hypotheses.confiance)
            for opp in opps:
                opp.confiance = cap_confidence(opp.confiance)
        return analysis, companies, build_fiches(item, filtre, analysis, companies, opps)
    except LLMUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc
    except LLMError as exc:
        log.error("LLM : %s", exc)
        diag.erreurs.append(str(exc))
        return None, [], []
    finally:
        diag.json_valides += stats.valides
        diag.json_invalides += stats.invalides
        diag.erreurs.extend(stats.erreurs)


# ----------------------------------------------------------------------------------------
# Points d'entrée de l'API
# ----------------------------------------------------------------------------------------
def analyze(req: AnalyzeRequest, mode: Mode = "live") -> AnalyzeResponse:
    if mode == "demo":
        return demo.replay_analyze(req)
    item, text = build_item(req)
    diag = Diagnostic()
    try:
        filtre = get_filter(_filter_languages([item.langue])).apply([item])[0]
    except Exception as exc:  # Ollama / ChromaDB indisponibles
        raise HTTPException(502, f"Filtre indisponible (embeddings {settings.EMBED_MODEL}) : {exc}") from exc
    response = AnalyzeResponse(mode=mode, actualite=item, filtre=filtre, diagnostic=diag)
    if not filtre.retenue and not req.forcer:
        log.info("Écartée par le filtre (%.3f < %.2f ou hors périmètre) : %s", filtre.score, filtre.seuil, item.titre)
        return response
    analysis, companies, fiches = run_on_item(item, filtre, text, step4.load_reference(), diag)
    response.analyse, response.entreprises, response.fiches = analysis, companies, fiches
    return response


def opportunities(query: str | None, timespan: str, languages: list[str], max_articles: int,
                  max_fiches: int, max_actualites: int = 5, mode: Mode = "live") -> OpportunitiesResponse:
    if mode == "demo":
        return demo.replay_opportunities(query, timespan, languages)
    diag = Diagnostic()
    items, errors = collect(query, languages, timespan, max_articles)
    diag.erreurs.extend(errors)
    try:
        results = get_filter(_filter_languages(languages)).apply(items)
    except Exception as exc:
        raise HTTPException(502, f"Filtre indisponible (embeddings {settings.EMBED_MODEL}) : {exc}") from exc
    kept = sorted(((i, r) for i, r in zip(items, results) if r.retenue), key=lambda x: -x[1].score)
    log.info("Filtre : %d/%d retenues ; analyse des %d meilleures.", len(kept), len(items), min(len(kept), max_actualites))
    reference = step4.load_reference()
    fiches: list[FicheOpportunite] = []
    for item, filtre in select_diverse(kept, max_actualites):
        fiches.extend(run_on_item(item, filtre, None, reference, diag)[2])
        if len(fiches) >= max_fiches:
            break
    return OpportunitiesResponse(
        mode=mode, requete=query, periode=timespan, langues=languages, genere_le=datetime.now(timezone.utc),
        nb_collectees=len(items), nb_retenues=len(kept), fiches=fiches[:max_fiches], diagnostic=diag,
    )
