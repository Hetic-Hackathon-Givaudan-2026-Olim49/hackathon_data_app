"""Réponses factices conformes au contrat, pour que l'équipe Tech intègre sans attendre
le pipeline. Tout y est explicitement marqué comme fictif (préfixe [DUMMY], domaine
example.org, entreprise « Exemple Agro SA ») : rien ne peut être pris pour une vraie
actualité ou une vraie entreprise.
"""

from __future__ import annotations

from datetime import datetime, timezone

from veille.models import (
    Actualite,
    Analyse,
    AnalyseImpact,
    AnalyzeRequest,
    AnalyzeResponse,
    Diagnostic,
    EntrepriseConcernee,
    Fait,
    FicheOpportunite,
    HypotheseOpportunite,
    MaillonImpact,
    OpportunitiesResponse,
    ResultatFiltre,
    Signal,
    Source,
)

_DATE = datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc)
_URL = "https://example.org/dummy/reglementation-edulcorant"
_SOURCE_ENTREPRISE = "https://example.org/dummy/base-entreprises/exemple-agro"


def _actualite(titre: str | None = None, url: str | None = None) -> Actualite:
    return Actualite(
        id="dummy-0001",
        titre=titre or "[DUMMY] Un pays européen relève la taxe sur les boissons sucrées en 2027",
        url=url or _URL,
        media="example.org",
        date=_DATE,
        langue="french",
        pays_media="FR",
        source_api="dummy",
    )


def _analyse(url: str) -> Analyse:
    return Analyse(
        faits=[
            Fait(
                enonce="[DUMMY] Une taxe plus élevée s'appliquera aux boissons contenant plus de 5 g de sucre / 100 ml.",
                citation="[DUMMY] « le barème sera relevé au 1er janvier 2027 »",
                source_url=url,
            )
        ],
        hypotheses=AnalyseImpact(
            resume="[DUMMY] Les fabricants de boissons vont chercher à réduire le sucre sans perdre le goût.",
            chaine_impact=[
                MaillonImpact(
                    fait="[DUMMY] Taxe relevée au-delà de 5 g de sucre / 100 ml.",
                    consequence="Reformulation des boissons sucrées pour passer sous le seuil.",
                    impact_sensoriel="Perte de rondeur et de sucrosité à compenser (modulateurs de goût, arômes).",
                )
            ],
            categories_produits=["boissons"],
            pays=["FR"],
            ingredients_notes=["sucre", "édulcorants"],
            confiance="moyen",
        ),
    )


def _entreprise() -> EntrepriseConcernee:
    return EntrepriseConcernee(
        nom="[DUMMY] Exemple Agro SA",
        raison="Catégorie « boissons » commune ; pays FR commun.",
        source=_SOURCE_ENTREPRISE,
        categories=["boissons"],
        pays=["FR"],
        score=2.0,
    )


def _fiche(actualite: Actualite, analyse: Analyse) -> FicheOpportunite:
    entreprise = _entreprise()
    opportunite = HypotheseOpportunite(
        entreprise=entreprise.nom,
        type_opportunite="reformulation",
        division_givaudan="Taste & Wellbeing",
        hypothese="[DUMMY] Proposer des solutions de réduction de sucre pour les gammes de boissons concernées.",
        fenetre_action="avant l'entrée en vigueur (janvier 2027)",
        question_client="[DUMMY] Quelles références de votre gamme dépassent aujourd'hui 5 g de sucre / 100 ml ?",
        faits_utilises=[0],
        confiance="moyen",
    )
    return FicheOpportunite(
        id=f"{actualite.id}--exemple-agro",
        signal=Signal(titre=actualite.titre, media=actualite.media, date=actualite.date, url=actualite.url),
        famille="reglementation_sensorielle",
        score_filtre=0.71,
        ce_que_ca_implique=analyse.hypotheses.resume,
        faits=analyse.faits,
        chaine_impact=analyse.hypotheses.chaine_impact,
        entreprise=entreprise,
        opportunite=opportunite,
        question_a_poser=opportunite.question_client,
        confiance=opportunite.confiance,
        sources=[
            Source(type="article", libelle=f"{actualite.media} — {actualite.titre}", url=actualite.url),
            Source(type="base_entreprises", libelle=f"Référence {entreprise.nom}", url=entreprise.source),
        ],
    )


def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if req.actualite:
        actualite = _actualite(req.actualite.titre, str(req.actualite.url))
    else:
        actualite = _actualite(url=str(req.url))
    analyse = _analyse(actualite.url)
    return AnalyzeResponse(
        mode="dummy",
        actualite=actualite,
        filtre=ResultatFiltre(
            retenue=True,
            famille="reglementation_sensorielle",
            score=0.71,
            seuil=0.55,
            scores_par_famille={
                "reglementation_sensorielle": 0.71,
                "matieres_premieres_aromatiques": 0.48,
                "tendances_gout_odorat": 0.52,
                "mouvements_entreprises_ff": 0.39,
            },
        ),
        analyse=analyse,
        entreprises=[_entreprise()],
        fiches=[_fiche(actualite, analyse)],
        diagnostic=Diagnostic(json_valides=3, duree_s=0.0),
    )


def opportunities(requete: str | None, periode: str, langues: list[str]) -> OpportunitiesResponse:
    actualite = _actualite()
    return OpportunitiesResponse(
        mode="dummy",
        requete=requete,
        periode=periode,
        langues=langues,
        genere_le=datetime.now(timezone.utc),
        nb_collectees=42,
        nb_retenues=1,
        fiches=[_fiche(actualite, _analyse(actualite.url))],
        diagnostic=Diagnostic(json_valides=3, duree_s=0.0),
    )
