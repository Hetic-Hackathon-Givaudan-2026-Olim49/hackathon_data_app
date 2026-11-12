"""Cache disque très simple (un fichier JSON par clé) pour les réponses réseau et LLM."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DiskCache:
    def __init__(self, directory: Path, ttl_s: float | None = None):
        self.directory = Path(directory)
        self.ttl_s = ttl_s  # None = pas d'expiration

    @staticmethod
    def key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        if self.ttl_s is not None and time.time() - path.stat().st_mtime > self.ttl_s:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Entrée de cache illisible (%s) : ignorée.", exc)
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = self._path(key).with_suffix(".tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            tmp.replace(self._path(key))
        except OSError as exc:  # disque plein, droits… : le cache est facultatif
            log.warning("Écriture du cache impossible (%s).", exc)
