"""Appel LLM local (Ollama) avec sortie JSON contrainte et validée par Pydantic.

- `format` = JSON Schema du modèle Pydantic attendu (Ollama contraint la génération) ;
- mode « thinking » de Qwen3 désactivé (`think=False`) : réponses plus courtes et plus rapides ;
- température basse (LLM_TEMPERATURE, 0.1 par défaut) ;
- une nouvelle tentative si la validation Pydantic échoue, avec l'erreur renvoyée au modèle ;
- cache disque des réponses valides (clé = modèle + messages + schéma).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from veille import settings
from veille.collect.cache import DiskCache

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client = ollama.Client(host=settings.OLLAMA_HOST, timeout=settings.LLM_TIMEOUT_S)
_cache = DiskCache(settings.CACHE_DIR / "llm")


class LLMError(RuntimeError):
    """Sortie invalide après la nouvelle tentative."""


class LLMUnavailable(LLMError):
    """Ollama injoignable ou modèle absent."""


@dataclass
class LLMStats:
    """Compteurs remontés dans le `Diagnostic` de l'API (évaluation : % de JSON valides)."""

    valides: int = 0
    invalides: int = 0
    erreurs: list[str] = field(default_factory=list)


def schema_with_enums(model: type[BaseModel], enums: dict[str, list[str]] | None = None) -> dict:
    """JSON Schema du modèle, en imposant une liste de valeurs à certains champs de type liste.

    Ex. `{"categories_produits": [...]}` force le LLM à choisir dans le vocabulaire contrôlé.
    """
    schema = copy.deepcopy(model.model_json_schema())
    for name, values in (enums or {}).items():
        prop = schema.get("properties", {}).get(name)
        if prop and prop.get("type") == "array":
            prop["items"] = {"type": "string", "enum": values}
    return schema


def generate(model: type[T], messages: list[dict], stats: LLMStats | None = None,
             enums: dict[str, list[str]] | None = None, label: str = "llm", schema: dict | None = None) -> T:
    """Appelle le LLM et retourne une instance validée de `model`.

    `schema` : JSON Schema imposé (par défaut celui de `model`, complété par `enums`).
    """
    stats = stats if stats is not None else LLMStats()
    schema = schema if schema is not None else schema_with_enums(model, enums)
    key = DiskCache.key(settings.LLM_MODEL, messages, schema)
    cached = _cache.get(key)
    if cached is not None:
        try:
            result = model.model_validate(cached)
            stats.valides += 1
            log.info("%s : réponse en cache.", label)
            return result
        except ValidationError:
            log.warning("%s : entrée de cache invalide, régénération.", label)

    attempt_messages = list(messages)
    for attempt in (1, 2):
        try:
            response = _client.chat(
                model=settings.LLM_MODEL,
                messages=attempt_messages,
                format=schema,
                think=False,
                options={"temperature": settings.LLM_TEMPERATURE, "num_ctx": 8192},
            )
        except ollama.ResponseError as exc:
            raise LLMUnavailable(f"Ollama a refusé la requête ({settings.LLM_MODEL}) : {exc}") from exc
        except Exception as exc:  # connexion refusée, timeout…
            raise LLMUnavailable(f"Ollama injoignable sur {settings.OLLAMA_HOST} : {exc}") from exc

        content = response.message.content or ""
        try:
            result = model.model_validate_json(content)
        except ValidationError as exc:
            stats.invalides += 1
            short = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])
            log.warning("%s : JSON invalide (essai %d/2) : %s", label, attempt, short)
            stats.erreurs.append(f"{label} : JSON invalide (essai {attempt}) : {short}")
            attempt_messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Ta réponse ne respecte pas le schéma ({short}). "
                                            "Renvoie uniquement un JSON valide et complet."},
            ]
            continue
        stats.valides += 1
        _cache.set(key, json.loads(result.model_dump_json()))
        return result
    raise LLMError(f"{label} : sortie invalide après 2 essais.")
