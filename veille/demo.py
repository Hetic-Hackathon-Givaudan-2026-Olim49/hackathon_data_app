"""Mode démo : sorties précalculées, rejouées sans réseau ni LLM.

Construction (une fois, avec réseau et Ollama) :
    python -m veille.demo build --periode 7d --max-actualites 6

Produit :
- data/demo_news.json            actualités réelles collectées (figées) ;
- data/demo/analyze/<id>.json    réponse /analyze pour CHAQUE actualité figée (filtre pour toutes,
                                 analyse complète pour les retenues analysées) ;
- data/demo/opportunities.json   réponse /opportunities.
Avec DEMO_MODE=true, l'API ne fait que relire ces fichiers.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

from fastapi import HTTPException

from veille import settings
from veille.collect.base import news_id
from veille.models import Actualite, AnalyzeRequest, AnalyzeResponse, Diagnostic, OpportunitiesResponse

log = logging.getLogger(__name__)

DEMO_DIR = settings.DATA_DIR / "demo"
ANALYZE_DIR = DEMO_DIR / "analyze"
OPPORTUNITIES_PATH = DEMO_DIR / "opportunities.json"
NEWS_PATH = settings.DATA_DIR / "demo_news.json"


def _write(path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------------------
# Relecture (DEMO_MODE=true)
# ----------------------------------------------------------------------------------------
def replay_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    url = str(req.actualite.url if req.actualite else req.url)
    path = ANALYZE_DIR / f"{news_id(url)}.json"
    if not path.exists():
        raise HTTPException(404, "Mode démo : cette actualité ne fait pas partie du jeu figé. "
                                 "Utilisez une URL de data/demo_news.json (ou GET /opportunities).")
    response = AnalyzeResponse.model_validate_json(path.read_text(encoding="utf-8"))
    response.mode = "demo"
    return response


def replay_opportunities(query: str | None, timespan: str, languages: list[str]) -> OpportunitiesResponse:
    if not OPPORTUNITIES_PATH.exists():
        raise HTTPException(503, "Mode démo : jeu figé absent. Lancez `python -m veille.demo build`.")
    response = OpportunitiesResponse.model_validate_json(OPPORTUNITIES_PATH.read_text(encoding="utf-8"))
    response.mode = "demo"
    return response


# ----------------------------------------------------------------------------------------
# Construction (live)
# ----------------------------------------------------------------------------------------
def build(query: str | None, timespan: str, languages: list[str], max_articles: int, max_actualites: int) -> None:
    from veille import match as step4
    from veille.collect import collect
    from veille.filter import get_filter
    from veille.pipeline import _filter_languages, run_on_item, select_diverse

    diag = Diagnostic()
    items, errors = collect(query, languages, timespan, max_articles)
    diag.erreurs.extend(errors)
    NEWS_PATH.write_text(json.dumps({
        "_doc": f"Actualités réelles collectées le {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC "
                f"(requête={query!r}, période={timespan}, langues={languages}). Jeu figé pour DEMO_MODE.",
        "actualites": [json.loads(i.model_dump_json()) for i in items],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("%d actualités figées dans %s", len(items), NEWS_PATH)

    results = get_filter(_filter_languages(languages)).apply(items)
    ranked = sorted(zip(items, results), key=lambda x: -x[1].score)
    to_analyze = {i.id for i, _ in select_diverse([x for x in ranked if x[1].retenue], max_actualites)}
    done = 0
    reference = step4.load_reference()
    fiches = []
    for item, filtre in ranked:
        item_diag = Diagnostic()
        response = AnalyzeResponse(mode="demo", actualite=item, filtre=filtre, diagnostic=item_diag)
        if item.id in to_analyze:
            done += 1
            log.info("Analyse (%d/%d) [%s] : %s", done, len(to_analyze), filtre.famille, item.titre[:80])
            analysis, companies, item_fiches = run_on_item(item, filtre, None, reference, item_diag)
            response.analyse, response.entreprises, response.fiches = analysis, companies, item_fiches
            fiches.extend(item_fiches)
            diag.json_valides += item_diag.json_valides
            diag.json_invalides += item_diag.json_invalides
            diag.rejets_validation.extend(item_diag.rejets_validation)
            diag.erreurs.extend(item_diag.erreurs)
        _write(ANALYZE_DIR / f"{item.id}.json", response)

    _write(OPPORTUNITIES_PATH, OpportunitiesResponse(
        mode="demo", requete=query, periode=timespan, langues=languages, genere_le=datetime.now(timezone.utc),
        nb_collectees=len(items), nb_retenues=sum(r.retenue for r in results), fiches=fiches, diagnostic=diag,
    ))
    log.info("Démo figée : %d actualités, %d analysées, %d fiches → %s", len(items), len(to_analyze), len(fiches), DEMO_DIR)


def load_news() -> list[Actualite]:
    data = json.loads(NEWS_PATH.read_text(encoding="utf-8"))
    return [Actualite.model_validate(x) for x in data["actualites"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Construit le jeu de démo figé.")
    parser.add_argument("action", choices=["build"])
    parser.add_argument("--q", default=None, help="mots-clés (défaut : requêtes des 4 familles)")
    parser.add_argument("--periode", default="7d")
    parser.add_argument("--langues", default=",".join(settings.NEWS_LANGUAGES))
    parser.add_argument("--max-articles", type=int, default=30)
    parser.add_argument("--max-actualites", type=int, default=6, help="actualités retenues analysées par le LLM")
    args = parser.parse_args()
    build(args.q, args.periode, [x.strip() for x in args.langues.split(",")], args.max_articles, args.max_actualites)


if __name__ == "__main__":
    main()
