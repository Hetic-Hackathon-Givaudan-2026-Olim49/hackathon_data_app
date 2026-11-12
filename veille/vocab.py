"""Accès au vocabulaire contrôlé (data/vocab.json) et normalisation des sorties LLM."""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache

from veille import settings

VOCAB_PATH = settings.DATA_DIR / "vocab.json"


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def categories() -> dict[str, dict]:
    return load()["categories"]


def countries() -> dict[str, str]:
    return load()["pays"]


def regions() -> dict[str, list[str]]:
    return load().get("regions", {})


def _key(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.strip().lower()


def normalize_categories(values: list[str]) -> list[str]:
    """Garde uniquement les identifiants connus (tolère libellés et casse)."""
    by_label = {_key(v["libelle"]): k for k, v in categories().items()}
    out = []
    for value in values:
        key = _key(value)
        cat = key if key in categories() else by_label.get(key)
        if cat and cat not in out:
            out.append(cat)
    return out


def normalize_countries(values: list[str]) -> list[str]:
    """Codes ISO2 connus (tolère noms français) ; UE et MONDE conservés tels quels."""
    by_name = {_key(name): code for code, name in countries().items()}
    out = []
    for value in values:
        code = value.strip().upper()
        code = code if code in countries() else by_name.get(_key(value))
        if code and code not in out:
            out.append(code)
    return out


def expand_countries(codes: list[str]) -> set[str]:
    """UE → pays membres. MONDE est laissé tel quel (traité à part par la jointure)."""
    out: set[str] = set()
    for code in codes:
        out.update(regions().get(code, [code]))
    return out
