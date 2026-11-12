"""API « Veille par impact ».

Lancement (depuis la racine du repo) :
    python -m veille.api
    # ou : uvicorn veille.api:app --host 0.0.0.0 --port 8001
Swagger : http://localhost:8001/docs   — contrat brut : /openapi.json

Modes (variables d'environnement, ou `?dummy=true` par requête) :
- DUMMY_MODE=true : réponses factices conformes au contrat, aucun calcul ;
- DEMO_MODE=true  : rejoue les sorties précalculées de data/demo/, sans réseau ;
- sinon           : pipeline live (GDELT + Ollama).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from veille import __version__, check_env, dummy, settings
from veille.models import AnalyzeRequest, AnalyzeResponse, HealthResponse, Mode, OpportunitiesResponse

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)-7s %(name)s — %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("veille.api")

# Exemples OpenAPI générés à partir du mode dummy : ils sont donc toujours conformes au contrat.
_EX_ANALYZE = dummy.analyze(AnalyzeRequest(url="https://example.org/dummy/article")).model_dump(mode="json")
_EX_OPPORTUNITIES = dummy.opportunities("édulcorant", "7d", ["french"]).model_dump(mode="json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    report = check_env.check()
    check_env.log_report(report)
    app.state.env_report = report
    log.info("API prête — mode %s — réglages : %s", _default_mode(), settings.summary())
    yield


app = FastAPI(
    title="Veille par impact — API",
    version=__version__,
    lifespan=lifespan,
    description=(
        "Brique Data du hackathon Givaudan (thématique 01). Détecte les actualités qui impactent "
        "les produits des clients **sans forcément les citer**, puis propose des hypothèses "
        "d'opportunités sourcées.\n\n"
        "Pipeline : **1. collecte** (GDELT) → **2. filtre** (embeddings, sans LLM) → "
        "**3. analyse implicite** (LLM local) → **4. jointure entreprises** (déterministe) → "
        "**5. hypothèses d'opportunités** (LLM local).\n\n"
        "Chaque fiche contient ses `sources` (article + base d'entreprises). Les champs `faits` "
        "sont vérifiables ; `chaine_impact` et `opportunite` sont des hypothèses avec une `confiance`.\n\n"
        "**Mode dummy** : ajoutez `?dummy=true` à n'importe quel endpoint (ou `DUMMY_MODE=true`) "
        "pour obtenir une réponse factice conforme au contrat.\n\n"
        "**Via ngrok** : ajoutez l'en-tête `ngrok-skip-browser-warning: 1` aux appels programmatiques."
    ),
    openapi_tags=[
        {"name": "veille", "description": "Pipeline de veille par impact."},
        {"name": "système", "description": "État du service."},
    ],
)

# Le front de l'équipe Tech peut être servi depuis n'importe quel domaine (ngrok, localhost…).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _default_mode() -> Mode:
    if settings.DUMMY_MODE:
        return "dummy"
    return "demo" if settings.DEMO_MODE else "live"


def _mode(dummy_param: bool) -> Mode:
    return "dummy" if dummy_param else _default_mode()


def _pipeline():
    """Import tardif : le mode dummy ne dépend ni d'Ollama ni de ChromaDB."""
    try:
        from veille import pipeline
    except ImportError as exc:
        raise HTTPException(501, f"Pipeline indisponible ({exc}). Utilisez `?dummy=true` en attendant.") from exc
    return pipeline


@app.get("/health", response_model=HealthResponse, tags=["système"])
def health(dummy: bool = Query(False, description="Ignoré : présent pour l'homogénéité des appels.")):
    """État du service : Ollama, modèles, mode actif, réglages effectifs (sans secret)."""
    report = check_env.check()
    collections: dict[str, int] = {}
    if report["ollama_joignable"] and _default_mode() == "live":
        try:
            from veille import rag_bridge

            client = rag_bridge.chroma_client()
            names = {c.name for c in client.list_collections()}
            for name in (settings.FAMILIES_COLLECTION, settings.COMPANIES_COLLECTION):
                collections[name] = client.get_collection(name).count() if name in names else 0
        except Exception as exc:  # ChromaDB absente ou corrompue : on le signale sans planter
            report["messages"].append(f"ChromaDB illisible : {exc}")
    ok = report["ok"] or _default_mode() != "live"
    return HealthResponse(
        statut="ok" if ok else "degrade",
        mode=_default_mode(),
        version=__version__,
        ollama={k: report[k] for k in ("ollama_host", "ollama_joignable", "modeles_presents", "modeles_manquants")},
        reglages=settings.summary(),
        collections=collections,
        messages=report["messages"],
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["veille"],
    responses={200: {"content": {"application/json": {"example": _EX_ANALYZE}}},
               422: {"description": "Requête invalide (ni `actualite` ni `url`, ou les deux)."},
               502: {"description": "Ollama ou la source d'actualité a échoué."}},
)
def analyze(req: AnalyzeRequest, dummy: bool = Query(False, description="Réponse factice conforme au contrat.")):
    """Analyse **une** actualité (objet `actualite` ou simple `url`) à travers les étapes 2 à 5.

    Si le filtre l'écarte, la réponse contient le score mais pas d'analyse (sauf `forcer=true`).
    """
    mode = _mode(dummy)
    if mode == "dummy":
        return dummy_module.analyze(req)
    start = time.perf_counter()
    response = _pipeline().analyze(req, mode=mode)
    response.diagnostic.duree_s = round(time.perf_counter() - start, 2)
    return response


@app.get(
    "/opportunities",
    response_model=OpportunitiesResponse,
    tags=["veille"],
    responses={200: {"content": {"application/json": {"example": _EX_OPPORTUNITIES}}},
               502: {"description": "Ollama ou GDELT a échoué."}},
)
def opportunities(
    q: str | None = Query(None, description="Mots-clés GDELT. Vide : requêtes par défaut des 4 familles.",
                          examples=["édulcorant"]),
    periode: str = Query("7d", pattern=r"^\d+(min|h|d|w|m)$", description="Fenêtre GDELT : 24h, 3d, 7d, 2w…"),
    langues: str | None = Query(None, description="Ex. `french,english`. Défaut : NEWS_LANGUAGES."),
    max_articles: int = Query(50, ge=1, le=250, description="Articles collectés par requête et par langue."),
    max_fiches: int = Query(20, ge=1, le=100),
    max_actualites: int = Query(5, ge=1, le=30, description="Actualités retenues analysées par le LLM "
                                "(≈ 1 min chacune en local). Les autres restent comptées dans `nb_retenues`."),
    dummy: bool = Query(False, description="Réponse factice conforme au contrat."),
):
    """Pipeline complet sur une requête et une période : collecte → filtre → analyse → jointure → hypothèses.

    En mode live, compter environ une minute par actualité analysée (LLM local). En mode démo, la réponse
    figée est renvoyée telle quelle (les paramètres sont ignorés).
    """
    lang_list = [x.strip().lower() for x in langues.split(",")] if langues else settings.NEWS_LANGUAGES
    mode = _mode(dummy)
    if mode == "dummy":
        return dummy_module.opportunities(q, periode, lang_list)
    start = time.perf_counter()
    response = _pipeline().opportunities(q, periode, lang_list, max_articles, max_fiches, max_actualites, mode=mode)
    response.diagnostic.duree_s = round(time.perf_counter() - start, 2)
    return response


# Le paramètre de requête s'appelle `dummy` : on garde un alias vers le module.
dummy_module = dummy


def main() -> None:
    import uvicorn

    uvicorn.run(
        "veille.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        proxy_headers=True,          # derrière ngrok : respecte X-Forwarded-Proto/Host
        forwarded_allow_ips="*",
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
