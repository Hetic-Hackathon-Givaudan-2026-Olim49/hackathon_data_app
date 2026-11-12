"""Étape 5 — hypothèses d'opportunités (LLM local), un seul appel par actualité.

Le LLM reçoit les faits sourcés, les hypothèses d'impact et les entreprises retenues par
la jointure (MAX_COMPANIES_PER_NEWS, 5 par défaut). Le champ `entreprise` est contraint par
JSON Schema à la liste fournie.

Contrôles après coup (sans LLM) — une opportunité est rejetée si :
- l'entreprise n'est pas dans les entrées ;
- elle ne s'appuie sur aucun fait d'entrée (`faits_utilises` vide ou indice inexistant) ;
- son texte cite une autre entreprise/marque connue de la base qui n'est pas dans les entrées.

Entreprise citée dans le titre : si le premier appel ne produit aucune hypothèse valide pour une
entreprise de la jointure nommée dans le titre (ex. L'Oréal pour « L'Oréal rachète Kering Beauté »),
un second appel ciblé sur ces seules entreprises l'impose (JSON Schema : `minItems`). Les mêmes
contrôles s'appliquent ; si elle reste rejetée, c'est tracé dans `rejets_validation` plutôt que comblé
par un texte fabriqué.
"""

from __future__ import annotations

from veille import llm
from veille.analyze import _messages
from veille.llm import LLMStats
from veille.match import known_names as _known_names
from veille.match import mentioned
from veille.match import norm_text as _norm
from veille.models import Analyse, EntrepriseConcernee, HypotheseOpportunite, HypothesesOpportunites

SYSTEM = """Tu es un expert business de Givaudan (arômes et parfums) qui prépare les rendez-vous
des commerciaux. Pour CHAQUE entreprise fournie, formule une hypothèse d'opportunité née de l'actualité.
Règles strictes :
- `entreprise` : exactement un des noms fournis. N'invente aucune entreprise, n'en cite pas d'autre.
- Appuie-toi uniquement sur les faits numérotés fournis ; `faits_utilises` = leurs numéros.
  N'ajoute aucun fait, chiffre ou date absent des entrées.
- `type_opportunite` : reformulation, lancement, substitution_ingredient ou expansion_geographique.
- `division_givaudan` : "Taste & Wellbeing" (alimentaire, boissons, santé) ou "Fragrance & Beauty"
  (parfumerie, cosmétique, hygiène, entretien).
- `hypothese` : 1 à 2 phrases au conditionnel (c'est une hypothèse, pas un fait).
- `fenetre_action` : estimation courte (ex. « 0-3 mois », « avant l'entrée en vigueur »), ou "inconnu".
- `question_client` : une question ouverte et concrète à poser au client, sans jargon interne.
- `confiance` : "élevé", "moyen" ou "spéculatif".
- Si une entreprise n'a pas d'opportunité crédible, ne la mets pas dans la liste.
Réponds uniquement en JSON, en français."""

FEWSHOT = [
    (
        "Actualité : Exemple fictif — La Commission européenne interdit l'additif E171 dans les aliments\n"
        "Faits :\n0. La Commission européenne interdit le dioxyde de titane (E171) comme additif alimentaire.\n"
        "1. Les fabricants ont six mois pour écouler leurs stocks.\n"
        "Impact : Les confiseurs vendant dans l'UE doivent reformuler sous six mois sans E171.\n"
        "Entreprises :\n- Exemple Confiserie SA : Catégories communes : Confiserie et chocolat ; commercialisée en FR, DE.",
        {"opportunites": [{
            "entreprise": "Exemple Confiserie SA",
            "type_opportunite": "reformulation",
            "division_givaudan": "Taste & Wellbeing",
            "hypothese": "Exemple Confiserie SA devrait reformuler ses confiseries blanchies à l'E171 et pourrait "
                         "avoir besoin d'ajuster leur profil aromatique.",
            "fenetre_action": "0-6 mois (avant la fin de l'écoulement des stocks)",
            "question_client": "Quelles références de votre gamme utilisent encore l'E171, et où en est leur reformulation ?",
            "faits_utilises": [0, 1],
            "confiance": "élevé",
        }]},
    ),
    (
        "Actualité : Example (fictitious) — Vanilla prices jump 40% after cyclone hits Madagascar\n"
        "Faits :\n0. Les prix de la vanille ont augmenté de 40 %.\n1. Un cyclone a frappé Madagascar.\n"
        "Impact : La hausse du prix de la vanille pèse sur les produits vanillés.\n"
        "Entreprises :\n- Exemple Glaces SAS : Catégories communes : Glaces et desserts ; pays non précisé.\n"
        "- Exemple Beauté SA : Catégories communes : Parfumerie fine ; pays non précisé.",
        {"opportunites": [
            {"entreprise": "Exemple Glaces SAS", "type_opportunite": "substitution_ingredient",
             "division_givaudan": "Taste & Wellbeing",
             "hypothese": "Exemple Glaces SAS pourrait chercher à sécuriser ou substituer une partie de sa vanille naturelle.",
             "fenetre_action": "0-3 mois", "question_client": "Comment la hausse du prix de la vanille affecte-t-elle vos gammes vanillées ?",
             "faits_utilises": [0], "confiance": "moyen"},
            {"entreprise": "Exemple Beauté SA", "type_opportunite": "substitution_ingredient",
             "division_givaudan": "Fragrance & Beauty",
             "hypothese": "Exemple Beauté SA pourrait envisager des alternatives pour ses notes vanillées si le coût pèse.",
             "fenetre_action": "inconnu", "question_client": "Vos parfums à note vanille sont-ils exposés à cette hausse ?",
             "faits_utilises": [0, 1], "confiance": "spéculatif"},
        ]},
    ),
]




CITED_RULE = """
Entreprises citées dans le titre : l'actualité les concerne directement. Produis OBLIGATOIREMENT au
moins une hypothèse pour CHACUNE d'elles, en t'appuyant sur les faits fournis (confiance "spéculatif"
si le lien avec les arômes et parfums est indirect)."""


def _schema_enums(companies: list[EntrepriseConcernee], min_items: int = 0) -> dict:
    schema = HypothesesOpportunites.model_json_schema()
    schema["$defs"]["HypotheseOpportunite"]["properties"]["entreprise"]["enum"] = [c.nom for c in companies]
    if min_items:
        schema["properties"]["opportunites"]["minItems"] = min_items
    return schema


def validate(opps: list[HypotheseOpportunite], companies: list[EntrepriseConcernee], n_facts: int,
             reference: list[dict], rejects: list[str]) -> list[HypotheseOpportunite]:
    allowed = {c.nom for c in companies}
    known = _known_names(reference)
    kept, seen = [], set()
    for opp in opps:
        if opp.entreprise not in allowed:
            rejects.append(f"5 : entreprise hors entrées rejetée : {opp.entreprise}")
            continue
        if not opp.faits_utilises or any(i < 0 or i >= n_facts for i in opp.faits_utilises):
            rejects.append(f"5 : opportunité sans fait d'entrée valide rejetée ({opp.entreprise}) : {opp.faits_utilises}")
            continue
        text = f" {_norm(opp.hypothese + ' ' + opp.question_client)} "
        intruders = {owner for alias, owner in known.items() if f" {alias} " in text and owner not in allowed}
        if intruders:
            rejects.append(f"5 : opportunité citant une entreprise hors entrées rejetée ({opp.entreprise}) : {sorted(intruders)}")
            continue
        if opp.entreprise in seen:
            continue
        seen.add(opp.entreprise)
        kept.append(opp)
    return kept


def _user_prompt(title: str, analysis: Analyse, companies: list[EntrepriseConcernee]) -> str:
    facts = "\n".join(f"{i}. {f.enonce}" for i, f in enumerate(analysis.faits))
    chain = "\n".join(f"- {m.fait} → {m.consequence} → {m.impact_sensoriel}" for m in analysis.hypotheses.chaine_impact)
    firms = "\n".join(f"- {c.nom} : {c.raison}" for c in companies)
    return (f"Actualité : {title}\nFaits :\n{facts}\nImpact : {analysis.hypotheses.resume}\n"
            f"Chaîne d'impact (hypothèses) :\n{chain or '- inconnu'}\nEntreprises :\n{firms}")


def hypothesize(title: str, analysis: Analyse, companies: list[EntrepriseConcernee], reference: list[dict],
                stats: LLMStats, rejects: list[str]) -> list[HypotheseOpportunite]:
    if not companies or not analysis.faits:
        return []
    result = llm.generate(HypothesesOpportunites, _messages(SYSTEM, FEWSHOT, _user_prompt(title, analysis, companies)),
                          stats, label="5-opportunites", schema=_schema_enums(companies))
    kept = validate(result.opportunites, companies, len(analysis.faits), reference, rejects)

    # Garantie : au moins une hypothèse pour chaque entreprise de la jointure citée dans le titre.
    cited = mentioned(title, reference)
    covered = {o.entreprise for o in kept}
    missing = [c for c in companies if c.nom in cited and c.nom not in covered]
    if missing:
        retry = llm.generate(HypothesesOpportunites,
                             _messages(SYSTEM + CITED_RULE, FEWSHOT, _user_prompt(title, analysis, missing)),
                             stats, label="5-opportunites-citees", schema=_schema_enums(missing, len(missing)))
        extra = validate(retry.opportunites, missing, len(analysis.faits), reference, rejects)
        kept = extra + kept  # l'entreprise citée en premier, comme dans la jointure
        for company in missing:
            if company.nom not in {o.entreprise for o in extra}:
                rejects.append(f"5 : aucune hypothèse valide pour l'entreprise citée dans le titre : {company.nom}")
    return kept
