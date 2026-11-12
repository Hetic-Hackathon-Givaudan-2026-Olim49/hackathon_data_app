"""Étape 4 — identification des entreprises, sans LLM : jointure déterministe.

Entrées : catégories et pays issus de l'étape 3 (vocabulaire contrôlé).
Règles :
- une entreprise doit partager AU MOINS UNE catégorie avec l'analyse ;
- si l'analyse cite des pays (UE développée en pays membres), l'entreprise doit être
  commercialisée dans au moins l'un d'eux ; si le pays est « MONDE » ou absent, le critère
  pays est neutre (score réduit) ;
- score = Σ catégories communes (1 + part lissée des produits de l'entreprise dans la catégorie)
          + 1 si pays commun (0,5 si MONDE / non localisé).
  Part lissée = n / (échantillon + 20) : un petit échantillon (peu de produits référencés)
  ne domine pas le classement. (Un score fondé sur la seule part a été essayé puis écarté :
  il pénalise par construction les groupes diversifiés comme Nestlé ; voir README, section 7.)
Chaque entreprise retournée porte la `raison` (critères qui ont matché, avec effectifs)
et la `source` (page Open Food Facts / Open Beauty Facts qui permet de le vérifier).

Famille « matières premières » : les pays cités sont des pays d'origine (ex. Côte d'Ivoire pour le
cacao), pas des marchés ; le critère pays y est neutre, comme pour MONDE.

Règle complémentaire : une entreprise de la base CITÉE dans le titre (nom ou marque, ≥ 4 caractères)
est rattachée en priorité (+2), avec l'article comme source.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata

from veille import settings, vocab
from veille.models import AnalyseImpact, EntrepriseConcernee

log = logging.getLogger(__name__)

MONDE = "MONDE"
SMOOTHING = 20
# Familles où les pays cités sont des pays de PRODUCTION (origine d'une matière première), pas des
# marchés : le critère pays y est neutre, comme pour MONDE (ex. récolte de cacao en Côte d'Ivoire).
FAMILIES_WITHOUT_MARKET_COUNTRY = {"matieres_premieres_aromatiques"}


def load_reference() -> list[dict]:
    """Base d'entreprises indexée dans ChromaDB (RAG) ; repli sur data/companies.json."""
    from veille.companies import load_companies

    try:
        from veille import rag_bridge

        client = rag_bridge.chroma_client()
        if settings.COMPANIES_COLLECTION in {c.name for c in client.list_collections()}:
            data = client.get_collection(settings.COMPANIES_COLLECTION).get(include=["metadatas"])
            companies = [json.loads(m["data"]) for m in data["metadatas"] or []]
            if companies:
                return companies
        log.warning("Collection '%s' vide : lancez `python -m veille.companies index`. Repli sur le JSON.",
                    settings.COMPANIES_COLLECTION)
    except Exception as exc:  # ChromaDB indisponible : le JSON suffit à la jointure
        log.warning("ChromaDB indisponible (%s) : repli sur data/companies.json.", exc)
    return load_companies()


def norm_text(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w]+", " ", text).strip()


def known_names(reference: list[dict]) -> dict[str, str]:
    """Nom ou marque normalisé (≥ 4 caractères, pour éviter « lu », « fa », « mac ») → entreprise."""
    names = {}
    for company in reference:
        for alias in [company["nom"], *company.get("marques", [])]:
            key = norm_text(alias.replace("-", " "))
            if len(key) >= 4:
                names[key] = company["nom"]
    return names


def mentioned(text: str, reference: list[dict]) -> dict[str, str]:
    """Entreprises citées dans `text` : {nom de l'entreprise: alias trouvé}."""
    padded = f" {norm_text(text)} "
    found: dict[str, str] = {}
    for alias, owner in known_names(reference).items():
        if f" {alias} " in padded:
            found.setdefault(owner, alias)
    return found


def _source_for(company: dict, categories: list[str]) -> str:
    """Page de référence la plus pertinente : OBF pour la beauté, OFF pour l'alimentaire."""
    # La base qui justifie la catégorie la plus pertinente (la 1re, dans l'ordre de l'analyse)
    beauty = bool(categories) and vocab.categories()[categories[0]]["division"] == "Fragrance & Beauty"
    wanted = "Open Beauty Facts" if beauty else "Open Food Facts"
    sources = company.get("sources") or []
    for s in sources:
        if s["base"] == wanted:
            return s["url"]
    return sources[0]["url"] if sources else "inconnu"


def match(analysis: AnalyseImpact, companies: list[dict], limit: int | None = None,
          title: str | None = None, article_url: str | None = None,
          famille: str | None = None) -> list[EntrepriseConcernee]:
    cited = mentioned(title, companies) if title else {}
    wanted_cats = [c for c in analysis.categories_produits if c in vocab.categories()]
    if not wanted_cats and not cited:
        return []
    origin_only = famille in FAMILIES_WITHOUT_MARKET_COUNTRY
    worldwide = origin_only or not analysis.pays or MONDE in analysis.pays
    wanted_countries = set() if origin_only else vocab.expand_countries([p for p in analysis.pays if p != MONDE])

    results = []
    for company in companies:
        total = max(company.get("produits_echantillonnes", 0), 1)
        smoothed = total + SMOOTHING
        cat_hits = [c for c in wanted_cats if c in company.get("categories", {})]
        country_hits = sorted(wanted_countries & set(company.get("pays", {})))
        if company["nom"] in cited:
            results.append(EntrepriseConcernee(
                nom=company["nom"],
                raison=f"Citée dans l'actualité (« {cited[company['nom']]} »)"
                       + (f" ; catégories communes : {', '.join(cat_hits)}." if cat_hits else "."),
                source=article_url or _source_for(company, cat_hits),
                categories=cat_hits, pays=country_hits,
                score=round(2 + len(cat_hits) + (1.0 if country_hits else 0.0), 3),
            ))
            continue
        if not cat_hits:
            continue
        if wanted_countries and not country_hits and not worldwide:
            continue

        score = sum(1 + company["categories"][c] / smoothed for c in cat_hits)
        score += 1.0 if country_hits else 0.5

        cats_txt = ", ".join(
            f"{vocab.categories()[c]['libelle']} ({company['categories'][c]} produits sur {total} échantillonnés)"
            for c in cat_hits
        )
        if country_hits:
            pays_txt = f"commercialisée en {', '.join(country_hits)}"
        else:
            pays_txt = ("pays cités = origine de la matière première, impact sur tous les marchés" if origin_only
                        else "pays non précisé par l'actualité (impact mondial)")
        results.append(EntrepriseConcernee(
            nom=company["nom"],
            raison=f"Catégories communes : {cats_txt} ; {pays_txt}.",
            source=_source_for(company, cat_hits),
            categories=cat_hits,
            pays=country_hits,
            score=round(score, 3),
        ))
    results.sort(key=lambda e: (-e.score, e.nom))
    return results[:limit] if limit else results
