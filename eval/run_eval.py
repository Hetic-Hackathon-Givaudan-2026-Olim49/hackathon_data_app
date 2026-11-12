"""Évalue le pipeline sur le jeu de référence annoté par l'équipe.

    python eval/run_eval.py                                  # API locale (http://localhost:8001)
    python eval/run_eval.py --api https://xxxx.ngrok-free.app --verifier-urls

Pour chaque entrée ANNOTÉE de eval/reference_set.json (pertinente = oui/non ; les autres
sont ignorées), appelle POST /analyze, puis calcule :
- précision du filtre (et rappel, pour information) ;
- exactitude de la famille sur les vrais positifs annotés avec une famille ;
- % d'entreprises justifiées par une source vérifiable (raison non vide + URL http(s),
  et URL qui répond si --verifier-urls) ;
- % de sorties JSON valides (appels LLM validés par Pydantic / appels totaux) ;
- couverture des entreprises attendues (pour information).
Exporte les hypothèses dans eval/hypotheses_to_rate.csv (colonne `note_humaine` à remplir :
plausible / non plausible) et les métriques dans eval/results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parent
REFERENCE = EVAL_DIR / "reference_set.json"
CSV_OUT = EVAL_DIR / "hypotheses_to_rate.csv"
RESULTS_OUT = EVAL_DIR / "results.json"
FAMILIES = EVAL_DIR.parent / "data" / "families.json"
HEADERS = {"ngrok-skip-browser-warning": "1"}


def pct(num: float, den: float) -> float | None:
    return round(100 * num / den, 1) if den else None


def annotated_entries() -> list[dict]:
    if not REFERENCE.exists():
        sys.exit(f"{REFERENCE} introuvable : lancez d'abord `python eval/build_reference_set.py`.")
    entries = json.loads(REFERENCE.read_text(encoding="utf-8"))["entrees"]
    kept = [e for e in entries if str(e.get("pertinente", "")).strip().lower() in ("oui", "non")]
    check_families(kept)
    print(f"{len(kept)}/{len(entries)} entrées annotées (les autres sont ignorées).")
    return kept


class AnnotationError(ValueError):
    """Annotation incompatible avec le pipeline (valeur hors vocabulaire)."""


def check_families(entries: list[dict]) -> None:
    """Toute famille inconnue est une erreur : sinon elle compterait silencieusement comme fausse."""
    families = json.loads(FAMILIES.read_text(encoding="utf-8"))
    allowed = {k for k in families if not k.startswith("_") and k != "hors_perimetre"} | {""}
    bad = [(e.get("id", "?"), e.get("famille")) for e in entries if (e.get("famille") or "") not in allowed]
    if bad:
        raise AnnotationError(
            "Famille(s) inconnue(s) dans reference_set.json : "
            + ", ".join(f"{i} → {f!r}" for i, f in bad)
            + f". Valeurs acceptées : {', '.join(sorted(allowed - {''}))} (ou vide si pertinente = non)."
        )


def request_body(entry: dict) -> dict:
    actualite = {"titre": entry["titre"], "url": entry["url"]}
    for field in ("media", "langue"):
        if entry.get(field):
            actualite[field] = entry[field]
    if entry.get("date"):
        try:
            actualite["date"] = datetime.fromisoformat(str(entry["date"]).replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass  # date dans un format inattendu : on l'omet plutôt que d'échouer
    return {"actualite": actualite}


def url_ok(client: httpx.Client, url: str, cache: dict) -> bool:
    if url not in cache:
        try:
            cache[url] = client.get(url, timeout=20, follow_redirects=True).status_code < 400
        except httpx.HTTPError:
            cache[url] = False
    return cache[url]


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation du pipeline sur le jeu de référence.")
    parser.add_argument("--api", default="http://localhost:8001")
    parser.add_argument("--timeout", type=float, default=600, help="secondes par appel /analyze")
    parser.add_argument("--verifier-urls", action="store_true", help="vérifie que chaque URL source répond")
    args = parser.parse_args()

    entries = annotated_entries()
    if not entries:
        sys.exit("Aucune entrée annotée : rien à évaluer.")

    client = httpx.Client(base_url=args.api.rstrip("/"), timeout=args.timeout, headers=HEADERS)
    checker = httpx.Client(headers=HEADERS)
    tp = fp = fn = tn = 0
    famille_ok = famille_total = 0
    companies_total = companies_justified = 0
    json_valid = json_invalid = 0
    expected_total = expected_found = 0
    errors, rows, url_cache = [], [], {}

    for n, entry in enumerate(entries, 1):
        truth = entry["pertinente"].strip().lower() == "oui"
        print(f"[{n}/{len(entries)}] {entry['titre'][:80]}")
        try:
            r = client.post("/analyze", json=request_body(entry))
            r.raise_for_status()
            resp = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            errors.append({"id": entry["id"], "erreur": str(exc)[:300]})
            print(f"   ! erreur : {exc}")
            continue

        predicted = resp["filtre"]["retenue"]
        tp += predicted and truth
        fp += predicted and not truth
        fn += (not predicted) and truth
        tn += (not predicted) and not truth
        if predicted and truth and entry.get("famille"):
            famille_total += 1
            famille_ok += resp["filtre"]["famille"] == entry["famille"]

        diag = resp.get("diagnostic", {})
        json_valid += diag.get("json_valides", 0)
        json_invalid += diag.get("json_invalides", 0)

        returned = {c["nom"]: c for c in resp.get("entreprises", [])}
        for company in returned.values():
            companies_total += 1
            justified = bool(company.get("raison")) and str(company.get("source", "")).startswith("http")
            if justified and args.verifier_urls:
                justified = url_ok(checker, company["source"], url_cache)
            companies_justified += justified
        expected = entry.get("entreprises_attendues") or []
        expected_total += len(expected)
        expected_found += sum(1 for name in expected if name in returned)

        for fiche in resp.get("fiches", []):
            opp = fiche["opportunite"]
            rows.append({
                "id_actualite": entry["id"], "titre": entry["titre"], "url": entry["url"],
                "famille_detectee": resp["filtre"]["famille"], "entreprise": opp["entreprise"],
                "raison_rattachement": fiche["entreprise"]["raison"], "type_opportunite": opp["type_opportunite"],
                "division_givaudan": opp["division_givaudan"], "hypothese": opp["hypothese"],
                "fenetre_action": opp["fenetre_action"], "question_client": opp["question_client"],
                "confiance": opp["confiance"],
                "faits": " | ".join(f["enonce"] for f in fiche["faits"]),
                "sources": " | ".join(s["url"] for s in fiche["sources"]),
                "opportunite_attendue (annotation)": entry.get("opportunite_attendue", ""),
                "note_humaine": "", "commentaire_noteur": "",
            })

    metrics = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api": args.api,
        "entrees_evaluees": len(entries) - len(errors),
        "erreurs_api": len(errors),
        "filtre": {
            "vrais_positifs": tp, "faux_positifs": fp, "faux_negatifs": fn, "vrais_negatifs": tn,
            "precision_pct": pct(tp, tp + fp), "rappel_pct": pct(tp, tp + fn),
            "famille_exacte_pct": pct(famille_ok, famille_total),
        },
        "entreprises": {
            "retournees": companies_total, "justifiees_source_verifiable": companies_justified,
            "justifiees_pct": pct(companies_justified, companies_total),
            "urls_verifiees_en_ligne": args.verifier_urls,
            "attendues_retrouvees_pct": pct(expected_found, expected_total),
        },
        "json": {"valides": json_valid, "invalides": json_invalid, "valides_pct": pct(json_valid, json_valid + json_invalid)},
        "hypotheses_exportees": len(rows),
        "erreurs": errors,
    }
    RESULTS_OUT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig : lisible dans Excel
        fieldnames = list(rows[0]) if rows else ["id_actualite", "note_humaine"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    f, e, j = metrics["filtre"], metrics["entreprises"], metrics["json"]
    print("\n=== Résultats ===")
    print(f"Précision du filtre      : {f['precision_pct']} %  (rappel {f['rappel_pct']} % ; TP={tp} FP={fp} FN={fn} TN={tn})")
    print(f"Famille exacte           : {f['famille_exacte_pct']} %")
    print(f"Entreprises justifiées   : {e['justifiees_pct']} %  ({companies_justified}/{companies_total})")
    print(f"JSON valides             : {j['valides_pct']} %  ({json_valid}/{json_valid + json_invalid})")
    print(f"Entreprises attendues    : {e['attendues_retrouvees_pct']} % retrouvées")
    print(f"→ {RESULTS_OUT}\n→ {CSV_OUT} ({len(rows)} hypothèses à noter : plausible / non plausible)")


if __name__ == "__main__":
    main()
