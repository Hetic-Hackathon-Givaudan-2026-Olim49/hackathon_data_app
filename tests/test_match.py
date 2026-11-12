"""Tests de la jointure déterministe (étape 4) et des contrôles anti-invention de l'étape 5."""

from veille.match import match
from veille.models import AnalyseImpact, EntrepriseConcernee, HypotheseOpportunite
from veille.opportunities import validate

COMPANIES = [
    {"nom": "Boissons SA", "marques": ["boissonsa"], "categories": {"boissons_sans_alcool": 60}, "pays": {"FR": 50, "US": 10},
     "produits_echantillonnes": 100, "sources": [{"base": "Open Food Facts", "url": "https://world.openfoodfacts.org/brand/boissonsa"}]},
    {"nom": "Glaces SAS", "marques": ["glacesas"], "categories": {"glaces_desserts": 40, "boissons_sans_alcool": 5},
     "pays": {"DE": 30}, "produits_echantillonnes": 100,
     "sources": [{"base": "Open Food Facts", "url": "https://world.openfoodfacts.org/brand/glacesas"}]},
    {"nom": "Beauté SA", "marques": ["beautesa"], "categories": {"parfumerie_fine": 20}, "pays": {"FR": 20},
     "produits_echantillonnes": 50, "sources": [{"base": "Open Food Facts", "url": "https://off/x"},
                                                {"base": "Open Beauty Facts", "url": "https://world.openbeautyfacts.org/brand/beautesa"}]},
]


def _analysis(cats, pays):
    return AnalyseImpact(categories_produits=cats, pays=pays)


def test_category_and_country_required():
    result = match(_analysis(["boissons_sans_alcool"], ["FR"]), COMPANIES)
    assert [c.nom for c in result] == ["Boissons SA"]  # Glaces SAS : catégorie commune mais pas en FR
    company = result[0]
    assert "Boissons sans alcool" in company.raison and "FR" in company.raison
    assert company.source.startswith("https://world.openfoodfacts.org/")
    assert company.pays == ["FR"]


def test_eu_region_expands_to_members():
    names = [c.nom for c in match(_analysis(["boissons_sans_alcool"], ["UE"]), COMPANIES)]
    assert names == ["Boissons SA", "Glaces SAS"]  # FR et DE sont dans l'UE ; tri par score


def test_worldwide_keeps_all_category_matches_with_lower_score():
    result = match(_analysis(["boissons_sans_alcool"], ["MONDE"]), COMPANIES)
    assert {c.nom for c in result} == {"Boissons SA", "Glaces SAS"}
    assert all("impact mondial" in c.raison for c in result)
    assert result[0].score == 2.0  # 1 + 60/(100+20) + 0.5


def test_no_category_no_match():
    assert match(_analysis([], ["FR"]), COMPANIES) == []
    assert match(_analysis(["cafe_the"], ["FR"]), COMPANIES) == []


def test_beauty_category_uses_beauty_source():
    [company] = match(_analysis(["parfumerie_fine"], ["FR"]), COMPANIES)
    assert company.source == "https://world.openbeautyfacts.org/brand/beautesa"


def test_limit():
    assert len(match(_analysis(["boissons_sans_alcool"], ["MONDE"]), COMPANIES, limit=1)) == 1


# --- Étape 5 : validation après coup -----------------------------------------------------
def _ent(nom):
    return EntrepriseConcernee(nom=nom, raison="r", source="https://s", score=1)


def test_validate_rejects_invented_company_and_facts():
    allowed = [_ent("Boissons SA")]
    opps = [
        HypotheseOpportunite(entreprise="Boissons SA", hypothese="Reformuler", faits_utilises=[0]),
        HypotheseOpportunite(entreprise="Inventée SA", hypothese="x", faits_utilises=[0]),       # hors entrées
        HypotheseOpportunite(entreprise="Boissons SA", hypothese="y", faits_utilises=[5]),       # fait inexistant
        HypotheseOpportunite(entreprise="Boissons SA", hypothese="z", faits_utilises=[]),        # aucun fait
        HypotheseOpportunite(entreprise="Boissons SA", hypothese="Comme Glaces SAS, lancer…", faits_utilises=[0]),
    ]
    rejects = []
    kept = validate(opps, allowed, n_facts=2, reference=COMPANIES, rejects=rejects)
    assert [o.hypothese for o in kept] == ["Reformuler"]
    assert len(rejects) == 4


def test_company_cited_in_title_is_matched_first_with_article_source():
    result = match(_analysis(["boissons_sans_alcool"], ["FR"]), COMPANIES,
                   title="Glaces SAS rachète une marque de sorbets", article_url="https://news/1")
    assert result[0].nom == "Glaces SAS"
    assert result[0].source == "https://news/1" and "Citée dans l'actualité" in result[0].raison


def test_cited_company_matched_even_without_category():
    [company] = match(_analysis([], []), COMPANIES, title="Beauté SA ouvre une usine", article_url="https://n/2")
    assert company.nom == "Beauté SA"


def test_raw_material_origin_country_is_not_a_market():
    # Récolte de cacao en Côte d'Ivoire : aucune entreprise n'y est commercialisée, mais toutes sont concernées.
    analysis = _analysis(["boissons_sans_alcool"], ["CI"])
    assert match(analysis, COMPANIES) == []
    result = match(analysis, COMPANIES, famille="matieres_premieres_aromatiques")
    assert {c.nom for c in result} == {"Boissons SA", "Glaces SAS"}
    assert all("origine de la matière première" in c.raison for c in result)


def test_cited_company_always_gets_a_hypothesis(monkeypatch):
    from veille import opportunities
    from veille.llm import LLMStats
    from veille.models import Analyse, Fait, HypothesesOpportunites

    calls = []

    def fake_generate(model, messages, stats, label="", schema=None, **kw):
        calls.append((label, schema["$defs"]["HypotheseOpportunite"]["properties"]["entreprise"]["enum"],
                      schema["properties"]["opportunites"].get("minItems")))
        if label == "5-opportunites":  # 1er appel : le LLM ignore l'entreprise citée
            return HypothesesOpportunites(opportunites=[
                HypotheseOpportunite(entreprise="Boissons SA", hypothese="h1", faits_utilises=[0])])
        return HypothesesOpportunites(opportunites=[
            HypotheseOpportunite(entreprise="Glaces SAS", hypothese="h2", faits_utilises=[0])])

    monkeypatch.setattr(opportunities.llm, "generate", fake_generate)
    analysis = Analyse(faits=[Fait(enonce="f", citation="c", source_url="https://n/1")],
                       hypotheses=AnalyseImpact(resume="r"))
    rejects = []
    opps = opportunities.hypothesize("Glaces SAS rachète une marque", analysis,
                                     [_ent("Glaces SAS"), _ent("Boissons SA")], COMPANIES, LLMStats(), rejects)
    assert [o.entreprise for o in opps] == ["Glaces SAS", "Boissons SA"]
    assert calls[1] == ("5-opportunites-citees", ["Glaces SAS"], 1)
    assert rejects == []


def test_cited_company_not_retried_when_already_covered(monkeypatch):
    from veille import opportunities
    from veille.llm import LLMStats
    from veille.models import Analyse, Fait, HypothesesOpportunites

    labels = []

    def fake_generate(model, messages, stats, label="", schema=None, **kw):
        labels.append(label)
        return HypothesesOpportunites(opportunites=[
            HypotheseOpportunite(entreprise="Glaces SAS", hypothese="h", faits_utilises=[0])])

    monkeypatch.setattr(opportunities.llm, "generate", fake_generate)
    analysis = Analyse(faits=[Fait(enonce="f", citation="c", source_url="https://n/1")],
                       hypotheses=AnalyseImpact(resume="r"))
    opportunities.hypothesize("Glaces SAS rachète", analysis, [_ent("Glaces SAS")], COMPANIES, LLMStats(), [])
    assert labels == ["5-opportunites"]

