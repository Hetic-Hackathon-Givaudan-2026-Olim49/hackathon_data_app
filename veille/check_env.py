"""Vérifie l'environnement local : Ollama joignable et modèles présents.

Lancé automatiquement au démarrage de l'API (sans bloquer : l'API démarre en mode
dégradé et /health indique quoi faire). Utilisable aussi seul :

    python -m veille.check_env      # code de sortie 0 si tout est prêt, 1 sinon
"""

from __future__ import annotations

import logging
import sys

import httpx

from veille import settings

log = logging.getLogger(__name__)


def _same_model(wanted: str, available: str) -> bool:
    """`bge-m3` correspond à `bge-m3:latest` ; `qwen3:8b` doit correspondre exactement."""
    if ":" not in wanted:
        wanted += ":latest"
    if ":" not in available:
        available += ":latest"
    return wanted == available


def check() -> dict:
    """Retourne {ok, ollama_joignable, modeles_presents, modeles_manquants, messages}."""
    required = {"LLM_MODEL": settings.LLM_MODEL, "EMBED_MODEL": settings.EMBED_MODEL}
    result = {
        "ok": False,
        "ollama_host": settings.OLLAMA_HOST,
        "ollama_joignable": False,
        "modeles_presents": [],
        "modeles_manquants": [],
        "messages": [],
    }
    try:
        response = httpx.get(f"{settings.OLLAMA_HOST.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        available = [m["name"] for m in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError) as exc:
        result["messages"].append(
            f"Ollama ne répond pas sur {settings.OLLAMA_HOST} ({exc.__class__.__name__}). "
            "Installez-le (https://ollama.com) puis lancez-le (`ollama serve` ou l'application), "
            "ou corrigez OLLAMA_HOST."
        )
        return result

    result["ollama_joignable"] = True
    for var, model in required.items():
        if any(_same_model(model, a) for a in available):
            result["modeles_presents"].append(model)
        else:
            result["modeles_manquants"].append(model)
            result["messages"].append(f"Modèle {var}='{model}' absent : exécutez `ollama pull {model}`.")
    result["ok"] = not result["modeles_manquants"]
    return result


def log_report(report: dict) -> None:
    if report["ok"]:
        log.info("Ollama OK (%s) — modèles : %s", report["ollama_host"], ", ".join(report["modeles_presents"]))
        return
    for message in report["messages"]:
        log.warning(message)
    if settings.DEMO_MODE or settings.DUMMY_MODE:
        log.warning("DEMO_MODE / DUMMY_MODE actif : l'API fonctionne quand même (sorties précalculées).")
    else:
        log.warning("Mode live dégradé : /analyze et /opportunities échoueront tant que ce n'est pas corrigé.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = check()
    log_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
