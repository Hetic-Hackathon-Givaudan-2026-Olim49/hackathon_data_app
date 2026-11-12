"""Génère eval/reference_set.json à partir de eval/news_to_annotate.json (fourni par l'équipe).

    python eval/build_reference_set.py

- Les champs source (id, titre, date, media, url, langue) sont recopiés.
- Les champs d'annotation sont créés VIDES. Ce script ne les remplit JAMAIS : ils constituent
  la vérité terrain et sont complétés uniquement par l'équipe (voir README, section 7).
- Si reference_set.json existe déjà, les annotations existantes sont CONSERVÉES : seules les
  nouvelles actualités sont ajoutées. Rien n'est écrasé.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SOURCE = EVAL_DIR / "news_to_annotate.json"
TARGET = EVAL_DIR / "reference_set.json"

SOURCE_FIELDS = ["id", "titre", "date", "media", "url", "langue"]
ANNOTATION_FIELDS = {
    "pertinente": "",              # "oui" | "non"
    "famille": "",                 # une des 4 familles, ou "" si non pertinente
    "entreprises_attendues": [],   # noms tels qu'ils figurent dans data/companies.json
    "opportunite_attendue": "",    # texte libre
    "annotateur": "",              # prénom ou initiales
    "commentaire": "",
}
INSTRUCTIONS = {
    "regle": "Vérité terrain : à remplir UNIQUEMENT par l'équipe, sans regarder les sorties du système.",
    "pertinente": "oui | non — l'actualité peut-elle impacter les produits d'un client F&F ?",
    "famille": "reglementation_sensorielle | matieres_premieres_aromatiques | tendances_gout_odorat | "
               "mouvements_entreprises_ff (vide si pertinente = non)",
    "entreprises_attendues": "liste de noms de data/companies.json (peut être vide)",
    "opportunite_attendue": "texte libre : l'opportunité que vous attendriez",
    "annotateur": "initiales ; une entrée sans `pertinente` est ignorée par run_eval.py",
}


def load_source() -> list[dict]:
    if not SOURCE.exists():
        sys.exit(f"Fichier introuvable : {SOURCE}\n"
                 "Déposez-y la liste fournie par l'équipe : [{id, titre, date, media, url, langue}, ...]")
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("actualites", [])
    missing = [i for i, x in enumerate(items) if not x.get("id") or not x.get("titre") or not x.get("url")]
    if missing:
        sys.exit(f"Entrées sans id/titre/url aux positions : {missing[:10]}")
    return items


def main() -> None:
    items = load_source()
    existing = {}
    if TARGET.exists():
        for entry in json.loads(TARGET.read_text(encoding="utf-8")).get("entrees", []):
            existing[str(entry["id"])] = entry

    entries, added = [], 0
    for item in items:
        key = str(item["id"])
        if key in existing:
            entry = existing[key]
            for f in SOURCE_FIELDS:  # met à jour les champs source, jamais les annotations
                entry[f] = item.get(f, entry.get(f, ""))
        else:
            entry = {f: item.get(f, "") for f in SOURCE_FIELDS}
            entry.update(json.loads(json.dumps(ANNOTATION_FIELDS)))
            added += 1
        entries.append(entry)

    TARGET.write_text(json.dumps({"_instructions": INSTRUCTIONS, "entrees": entries}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    annotated = sum(1 for e in entries if e.get("pertinente") in ("oui", "non"))
    print(f"{TARGET} : {len(entries)} entrées ({added} nouvelles, {annotated} déjà annotées).")


if __name__ == "__main__":
    main()
