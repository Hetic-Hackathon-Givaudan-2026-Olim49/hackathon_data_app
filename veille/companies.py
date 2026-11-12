"""Base de référence des entreprises (étape 4) : construction et indexation.

    python -m veille.companies build   # interroge Open Food Facts / Open Beauty Facts → data/companies.json
    python -m veille.companies index   # indexe data/companies.json dans ChromaDB (collection veille_entreprises)

Pour chaque groupe de data/companies_seed.json, on échantillonne ses produits (par marque)
et on en déduit :
- ses catégories (vocabulaire contrôlé data/vocab.json, via les étiquettes de catégories OFF/OBF) ;
- ses pays de commercialisation (étiquettes de pays OFF/OBF → ISO 3166-1 alpha-2).
Chaque catégorie et chaque pays garde son nombre de produits et l'URL qui permet de le vérifier.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from datetime import date
from urllib.parse import quote

import httpx

from veille import settings, vocab
from veille.collect.cache import DiskCache

log = logging.getLogger(__name__)

SEED_PATH = settings.DATA_DIR / "companies_seed.json"
COMPANIES_PATH = settings.DATA_DIR / "companies.json"
UA = {"User-Agent": "veille-par-impact/2.0 (hackathon Givaudan; usage non commercial)"}

OFF_SEARCH = "https://search.openfoodfacts.org/search"
OBF_SEARCH = "https://world.openbeautyfacts.org/api/v2/search"
PAGE_SIZE = 100
MIN_SHARE = 0.03   # une catégorie / un pays doit représenter au moins 3 % de l'échantillon
MIN_PRODUCTS = 3

# Étiquettes de pays OFF/OBF → ISO2 (pays du vocabulaire contrôlé uniquement)
OFF_COUNTRIES = {
    "en:france": "FR", "en:germany": "DE", "en:italy": "IT", "en:spain": "ES", "en:portugal": "PT",
    "en:belgium": "BE", "en:netherlands": "NL", "en:switzerland": "CH", "en:united-kingdom": "GB",
    "en:ireland": "IE", "en:poland": "PL", "en:sweden": "SE", "en:denmark": "DK", "en:austria": "AT",
    "en:greece": "GR", "en:turkey": "TR", "en:russia": "RU", "en:united-states": "US", "en:canada": "CA",
    "en:mexico": "MX", "en:brazil": "BR", "en:argentina": "AR", "en:china": "CN", "en:japan": "JP",
    "en:south-korea": "KR", "en:india": "IN", "en:indonesia": "ID", "en:thailand": "TH", "en:vietnam": "VN",
    "en:philippines": "PH", "en:australia": "AU", "en:morocco": "MA", "en:algeria": "DZ", "en:tunisia": "TN",
    "en:egypt": "EG", "en:nigeria": "NG", "en:cote-d-ivoire": "CI", "en:ghana": "GH", "en:kenya": "KE",
    "en:south-africa": "ZA", "en:madagascar": "MG", "en:saudi-arabia": "SA", "en:united-arab-emirates": "AE",
}

BASES = {
    "off": {"libelle": "Open Food Facts", "page": "https://world.openfoodfacts.org/brand/{}"},
    "obf": {"libelle": "Open Beauty Facts", "page": "https://world.openbeautyfacts.org/brand/{}"},
}


def _fetch_off(client: httpx.Client, brands: list[str]) -> list[dict]:
    # Syntaxe groupée : `brands_tags:"a" OR brands_tags:"b"` renvoie une erreur 500 côté serveur.
    query = "brands_tags:(" + " OR ".join(f'"{b}"' for b in brands) + ")"
    r = client.get(OFF_SEARCH, params={"q": query, "fields": "code,categories_tags,countries_tags",
                                       "page_size": PAGE_SIZE})
    r.raise_for_status()
    return r.json().get("hits", [])


def _fetch_obf(client: httpx.Client, brands: list[str]) -> list[dict]:
    r = client.get(OBF_SEARCH, params={"brands_tags": "|".join(brands), "fields": "code,categories_tags,countries_tags",
                                       "page_size": PAGE_SIZE, "sort_by": "unique_scans_n"})
    r.raise_for_status()
    return r.json().get("products", [])


def profile(products: list[dict]) -> tuple[Counter, Counter, Counter]:
    """(catégories vocab, pays ISO2, étiquettes OFF non rattachées) sur un échantillon de produits."""
    tag_to_cat = {tag: cat for cat, spec in vocab.categories().items() for tag in spec.get("off", [])}
    cats, countries, unmapped = Counter(), Counter(), Counter()
    for p in products:
        tags = p.get("categories_tags") or []
        found = {tag_to_cat[t] for t in tags if t in tag_to_cat}
        cats.update(found)
        if not found:
            unmapped.update(tags[:3])
        countries.update({OFF_COUNTRIES[c] for c in p.get("countries_tags") or [] if c in OFF_COUNTRIES})
    return cats, countries, unmapped


def _keep(counter: Counter, total: int) -> dict[str, int]:
    return {k: n for k, n in counter.most_common() if n >= MIN_PRODUCTS and n / max(total, 1) >= MIN_SHARE}


_raw_cache = DiskCache(settings.CACHE_DIR / "companies")  # produits bruts : relance sans re-télécharger


def build() -> list[dict]:
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))["groupes"]
    client = httpx.Client(timeout=60, headers=UA)
    companies, unmapped_all = [], Counter()
    for group in seed:
        products_by_base: dict[str, list[dict]] = {}
        for base in group["bases"]:
            key = DiskCache.key(base, group["marques"], PAGE_SIZE)
            cached = _raw_cache.get(key)
            if cached:
                products_by_base[base] = cached
                continue
            for attempt in (1, 2, 3):
                try:
                    fetch = _fetch_off if base == "off" else _fetch_obf
                    products_by_base[base] = fetch(client, group["marques"])
                    _raw_cache.set(key, products_by_base[base])
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    code = getattr(getattr(exc, "response", None), "status_code", "")
                    log.warning("%s / %s : %s %s (essai %d)", group["nom"], base, exc.__class__.__name__, code, attempt)
                    time.sleep(15 * attempt)
            time.sleep(12)  # OFF/OBF limitent les recherches (~10 par minute, moins en charge)
        all_products = [p for ps in products_by_base.values() for p in ps]
        cats, countries, unmapped = profile(all_products)
        unmapped_all.update(unmapped)
        total = len(all_products)
        sources = [
            {"base": BASES[b]["libelle"], "url": BASES[b]["page"].format(quote(group["marques"][0])),
             "produits_echantillonnes": len(ps)}
            for b, ps in products_by_base.items()
        ]
        company = {
            "nom": group["nom"],
            "marques": group["marques"],
            "categories": _keep(cats, total),
            "pays": _keep(countries, total),
            "produits_echantillonnes": total,
            "sources": sources,
            "date_extraction": date.today().isoformat(),
        }
        log.info("%-32s %3d produits | catégories %s | pays %s", group["nom"], total,
                 list(company["categories"])[:5], list(company["pays"])[:8])
        companies.append(company)
    COMPANIES_PATH.write_text(json.dumps({
        "_doc": "Généré par `python -m veille.companies build` à partir d'Open Food Facts / Open Beauty Facts "
                "(licence ODbL). Ne pas éditer à la main : modifier data/companies_seed.json puis relancer.",
        "entreprises": companies,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Étiquettes non rattachées les plus fréquentes : %s", unmapped_all.most_common(25))
    log.info("→ %s (%d entreprises)", COMPANIES_PATH, len(companies))
    return companies


def load_companies() -> list[dict]:
    return json.loads(COMPANIES_PATH.read_text(encoding="utf-8"))["entreprises"]


def document_of(company: dict) -> str:
    cats = ", ".join(vocab.categories()[c]["libelle"] for c in company["categories"] if c in vocab.categories())
    pays = ", ".join(vocab.countries().get(p, p) for p in company["pays"])
    return f"{company['nom']} — catégories : {cats or 'inconnues'} — pays : {pays or 'inconnus'}"


def index() -> int:
    """Indexe la base dans ChromaDB (réutilise le RAG). Les listes sont stockées en JSON dans les métadonnées."""
    from veille import rag_bridge

    companies = load_companies()
    collection = rag_bridge.get_collection(settings.COMPANIES_COLLECTION)
    docs = [document_of(c) for c in companies]
    collection.upsert(
        ids=[f"ent-{i:03d}" for i in range(len(companies))],
        documents=docs,
        embeddings=rag_bridge.embed(docs),
        metadatas=[{"nom": c["nom"], "data": json.dumps(c, ensure_ascii=False)} for c in companies],
    )
    log.info("Collection '%s' : %d entreprises indexées.", settings.COMPANIES_COLLECTION, collection.count())
    return collection.count()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Base de référence des entreprises.")
    parser.add_argument("action", choices=["build", "index"])
    args = parser.parse_args()
    build() if args.action == "build" else index()


if __name__ == "__main__":
    main()
