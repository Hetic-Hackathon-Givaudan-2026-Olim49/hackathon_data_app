"""Étape 3 — analyse implicite (LLM local), en deux appels :

(a) extraction des FAITS sourcés : uniquement ce que dit l'article, avec citation ;
(b) HYPOTHÈSES d'impact : chaîne fait → conséquence → impact sensoriel, catégories
    produits et pays (vocabulaire contrôlé imposé par JSON Schema), ingrédients/notes,
    niveau de confiance.

Contrôles après coup (sans LLM) :
- un fait dont la citation ne se retrouve pas dans le texte source est rejeté ;
- un maillon de la chaîne d'impact qui ne s'appuie sur aucun fait extrait est rejeté ;
- catégories et pays sont ramenés au vocabulaire contrôlé.

Les exemples few-shot sont FICTIFS et marqués comme tels.
"""

from __future__ import annotations

import json
import re
import unicodedata

from pydantic import BaseModel, Field

from veille import llm, vocab
from veille.llm import LLMStats
from veille.models import INCONNU, Actualite, Analyse, AnalyseImpact, Fait


# ----------------------------------------------------------------------------------------
# Schémas vus par le LLM (sans les champs que l'on remplit nous-mêmes, comme l'URL)
# ----------------------------------------------------------------------------------------
class FaitLLM(BaseModel):
    enonce: str = Field(description="Le fait, en français.")
    citation: str = Field(description="Extrait copié mot pour mot de l'article.")


class FaitsLLM(BaseModel):
    faits: list[FaitLLM]


# ----------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------
SYSTEM_FAITS = """Tu es analyste de veille pour un fabricant d'arômes et de parfums.
Tâche : extraire les FAITS présents dans l'article fourni. Règles strictes :
- Un fait = une information explicitement écrite dans l'article (qui, quoi, où, quand, combien).
- Aucune déduction, aucune opinion, aucune connaissance extérieure.
- `citation` : extrait copié MOT POUR MOT de l'article (dans sa langue d'origine), 5 à 30 mots.
- `enonce` : le fait reformulé en français, même si l'article est en anglais.
- 1 à 6 faits, les plus utiles pour comprendre l'impact sur des produits alimentaires,
  boissons, cosmétiques ou parfums.
- Si l'article ne contient que son titre, extrais les faits du titre.
Réponds uniquement en JSON."""

FEWSHOT_FAITS = [
    (
        "Titre : Exemple fictif — La Commission européenne interdit l'additif E171 dans les aliments\n"
        "Texte : La Commission européenne a adopté mardi l'interdiction du dioxyde de titane (E171) "
        "comme additif alimentaire. Les fabricants disposent de six mois pour écouler leurs stocks. "
        "L'additif était surtout utilisé pour blanchir des confiseries et des pâtisseries.",
        {"faits": [
            {"enonce": "La Commission européenne interdit le dioxyde de titane (E171) comme additif alimentaire.",
             "citation": "La Commission européenne a adopté mardi l'interdiction du dioxyde de titane (E171) comme additif alimentaire"},
            {"enonce": "Les fabricants ont six mois pour écouler leurs stocks.",
             "citation": "Les fabricants disposent de six mois pour écouler leurs stocks"},
            {"enonce": "L'E171 servait surtout à blanchir confiseries et pâtisseries.",
             "citation": "L'additif était surtout utilisé pour blanchir des confiseries et des pâtisseries"},
        ]},
    ),
    (
        "Titre : Example (fictitious) — Vanilla prices jump 40% after cyclone hits Madagascar\n"
        "Texte : Vanilla prices jump 40% after cyclone hits Madagascar",
        {"faits": [
            {"enonce": "Les prix de la vanille ont augmenté de 40 %.", "citation": "Vanilla prices jump 40%"},
            {"enonce": "Un cyclone a frappé Madagascar.", "citation": "after cyclone hits Madagascar"},
        ]},
    ),
]


def _system_impact() -> str:
    cats = "\n".join(f"- {k} : {v['libelle']}" for k, v in vocab.categories().items())
    return f"""Tu es analyste de veille pour un fabricant d'arômes et de parfums (Givaudan).
À partir des FAITS fournis (et seulement d'eux), formule des HYPOTHÈSES d'impact sur les produits
des clients (agroalimentaire, boissons, cosmétique, parfumerie, entretien). Règles :
- `chaine_impact` : 1 à 3 maillons. `fait` reprend un des faits fournis (ne crée aucun fait) ;
  `consequence` = conséquence business probable ; `impact_sensoriel` = effet sur le goût, l'odeur,
  la texture ou la formulation.
- `categories_produits` : choisis UNIQUEMENT parmi ces identifiants :
{cats}
- `pays` : codes ISO 3166-1 alpha-2 (FR, DE, US…), "UE" pour toute l'Union européenne,
  "MONDE" si non localisé. Seulement les pays où les produits sont touchés.
- `ingredients_notes` : ingrédients ou notes aromatiques/olfactives touchés (en français).
- `confiance` : "élevé" si l'impact découle directement des faits, "moyen" s'il faut une étape
  de raisonnement, "spéculatif" sinon.
- Si une information manque, écris "inconnu" ou laisse la liste vide. N'invente rien.
- `resume` : une ou deux phrases en français : ce que ça implique pour les clients.
Réponds uniquement en JSON, en français."""


FEWSHOT_IMPACT = [
    (
        "Titre : Exemple fictif — La Commission européenne interdit l'additif E171 dans les aliments\n"
        "Famille détectée : reglementation_sensorielle\n"
        "Faits :\n0. La Commission européenne interdit le dioxyde de titane (E171) comme additif alimentaire.\n"
        "1. Les fabricants ont six mois pour écouler leurs stocks.\n"
        "2. L'E171 servait surtout à blanchir confiseries et pâtisseries.",
        {
            "resume": "Les confiseurs et pâtissiers vendant dans l'UE doivent reformuler sous six mois sans E171.",
            "chaine_impact": [{
                "fait": "La Commission européenne interdit le dioxyde de titane (E171) comme additif alimentaire.",
                "consequence": "Reformulation obligatoire des produits blanchis à l'E171 vendus dans l'UE.",
                "impact_sensoriel": "Changement d'aspect et parfois de texture ; besoin d'ajuster l'enrobage et le profil aromatique.",
            }],
            "categories_produits": ["confiserie_chocolat", "biscuits_snacks"],
            "pays": ["UE"],
            "ingredients_notes": ["dioxyde de titane", "enrobages"],
            "confiance": "élevé",
        },
    ),
    (
        "Titre : Example (fictitious) — Vanilla prices jump 40% after cyclone hits Madagascar\n"
        "Famille détectée : matieres_premieres_aromatiques\n"
        "Faits :\n0. Les prix de la vanille ont augmenté de 40 %.\n1. Un cyclone a frappé Madagascar.",
        {
            "resume": "La hausse du prix de la vanille pèse sur les produits vanillés ; certains clients chercheront des alternatives.",
            "chaine_impact": [{
                "fait": "Les prix de la vanille ont augmenté de 40 %.",
                "consequence": "Hausse du coût des produits à la vanille naturelle, pression pour substituer ou réduire.",
                "impact_sensoriel": "Risque de profil vanillé moins riche si la dose baisse ; besoin d'arômes vanille alternatifs.",
            }],
            "categories_produits": ["glaces_desserts", "produits_laitiers", "confiserie_chocolat", "parfumerie_fine"],
            "pays": ["MONDE"],
            "ingredients_notes": ["vanille"],
            "confiance": "moyen",
        },
    ),
]


def _messages(system: str, fewshot: list[tuple[str, dict]], user: str) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for question, answer in fewshot:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    messages.append({"role": "user", "content": user})
    return messages


# ----------------------------------------------------------------------------------------
# Contrôles après coup
# ----------------------------------------------------------------------------------------
def _words(text: str) -> list[str]:
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", text)


def overlap(fragment: str, source: str) -> float:
    """Part des mots de `fragment` présents dans `source` (0 à 1)."""
    words = _words(fragment)
    if not words:
        return 0.0
    vocab_source = set(_words(source))
    return sum(w in vocab_source for w in words) / len(words)


CITATION_MIN_OVERLAP = 0.7
MAILLON_MIN_OVERLAP = 0.5


# ----------------------------------------------------------------------------------------
# Étape 3
# ----------------------------------------------------------------------------------------
def extract_facts(item: Actualite, text: str, stats: LLMStats, rejects: list[str]) -> list[Fait]:
    user = f"Titre : {item.titre}\nTexte : {text}"
    raw = llm.generate(FaitsLLM, _messages(SYSTEM_FAITS, FEWSHOT_FAITS, user), stats, label="3a-faits")
    source = f"{item.titre}\n{text}"
    facts = []
    for f in raw.faits:
        if overlap(f.citation, source) < CITATION_MIN_OVERLAP:
            rejects.append(f"3a : fait rejeté (citation introuvable dans l'article) : {f.enonce[:120]}")
            continue
        facts.append(Fait(enonce=f.enonce, citation=f.citation, source_url=item.url))
    return facts


def infer_impact(item: Actualite, famille: str | None, facts: list[Fait], stats: LLMStats,
                 rejects: list[str]) -> AnalyseImpact:
    if not facts:
        return AnalyseImpact(resume=INCONNU, confiance=INCONNU)
    listed = "\n".join(f"{i}. {f.enonce}" for i, f in enumerate(facts))
    user = f"Titre : {item.titre}\nFamille détectée : {famille or INCONNU}\nFaits :\n{listed}"
    enums = {"categories_produits": list(vocab.categories()), "pays": list(vocab.countries())}
    impact = llm.generate(AnalyseImpact, _messages(_system_impact(), FEWSHOT_IMPACT, user), stats,
                          enums=enums, label="3b-impact")
    facts_text = "\n".join(f.enonce for f in facts)
    kept = []
    for maillon in impact.chaine_impact:
        if overlap(maillon.fait, facts_text) < MAILLON_MIN_OVERLAP:
            rejects.append(f"3b : maillon rejeté (ne reprend aucun fait extrait) : {maillon.fait[:120]}")
            continue
        kept.append(maillon)
    impact.chaine_impact = kept
    impact.categories_produits = vocab.normalize_categories(impact.categories_produits)
    impact.pays = vocab.normalize_countries(impact.pays)
    return impact


def analyze(item: Actualite, famille: str | None, text: str, stats: LLMStats, rejects: list[str]) -> Analyse:
    facts = extract_facts(item, text, stats, rejects)
    return Analyse(faits=facts, hypotheses=infer_impact(item, famille, facts, stats, rejects))
