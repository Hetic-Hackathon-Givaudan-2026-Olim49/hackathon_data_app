"""Contrat de données (Pydantic) : sert à la fois à valider les sorties LLM et à
générer le contrat OpenAPI exposé par l'API.

Conventions :
- champs en français, valeurs textuelles en français ;
- tout champ qualitatif accepte la valeur "inconnu" plutôt que d'inventer ;
- les faits (`Fait`) sont toujours rattachés à une source ; les hypothèses
  (`MaillonImpact`, `HypotheseOpportunite`) sont séparées et portent une confiance.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

INCONNU = "inconnu"

Famille = Literal[
    "reglementation_sensorielle",
    "matieres_premieres_aromatiques",
    "tendances_gout_odorat",
    "mouvements_entreprises_ff",
]
FAMILLES: tuple[str, ...] = Famille.__args__  # type: ignore[attr-defined]

Confiance = Literal["élevé", "moyen", "spéculatif", "inconnu"]
TypeOpportunite = Literal[
    "reformulation", "lancement", "substitution_ingredient", "expansion_geographique", "inconnu"
]
Division = Literal["Taste & Wellbeing", "Fragrance & Beauty", "inconnu"]
Mode = Literal["live", "demo", "dummy"]


def _strip_accents(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", value) if unicodedata.category(c) != "Mn")


def normalise_confiance(value: object) -> object:
    """Tolère les variantes sans accent / en anglais renvoyées par un petit LLM."""
    if not isinstance(value, str):
        return value
    key = _strip_accents(value).strip().lower()
    return {
        "eleve": "élevé", "elevee": "élevé", "high": "élevé", "fort": "élevé",
        "moyen": "moyen", "moyenne": "moyen", "medium": "moyen",
        "speculatif": "spéculatif", "speculative": "spéculatif", "faible": "spéculatif", "low": "spéculatif",
        "inconnu": "inconnu", "unknown": "inconnu",
    }.get(key, value)


# ----------------------------------------------------------------------------------------
# Étape 1 : actualité collectée
# ----------------------------------------------------------------------------------------
class Actualite(BaseModel):
    """Une actualité telle que collectée (ou fournie à /analyze)."""

    id: str = Field(description="Identifiant stable (hash de l'URL normalisée).")
    titre: str
    url: str
    media: str = Field(INCONNU, description="Nom de domaine du média.")
    date: datetime | None = Field(None, description="Date de publication (UTC) si connue.")
    langue: str = Field(INCONNU, description="Langue de l'article (french, english…).")
    pays_media: str = Field(INCONNU, description="Pays du média selon la source.")
    description: str | None = Field(None, description="Chapeau / description si la source le fournit.")
    source_api: Literal["gdelt", "googlenews", "gnews", "manuel", "demo", "dummy"] = "manuel"


# ----------------------------------------------------------------------------------------
# Étape 2 : filtre par similarité d'embeddings (sans LLM génératif)
# ----------------------------------------------------------------------------------------
class ResultatFiltre(BaseModel):
    retenue: bool
    famille: Famille | None = Field(None, description="Famille la plus proche (None si aucune référence).")
    score: float = Field(description="Similarité cosinus maximale (0 à 1).")
    seuil: float
    scores_par_famille: dict[str, float] = Field(default_factory=dict)


# ----------------------------------------------------------------------------------------
# Étape 3 : analyse implicite (LLM) — faits d'un côté, hypothèses de l'autre
# ----------------------------------------------------------------------------------------
class Fait(BaseModel):
    """Fait vérifiable, présent dans l'article."""

    enonce: str = Field(description="Le fait, reformulé en français.")
    citation: str = Field(INCONNU, description="Extrait de l'article qui l'établit.")
    source_url: str = Field(description="URL de l'article dont provient le fait.")


class ExtractionFaits(BaseModel):
    """Sortie de l'appel 3a : uniquement des faits sourcés."""

    faits: list[Fait] = Field(default_factory=list)


class MaillonImpact(BaseModel):
    """Hypothèse : fait → conséquence → impact sensoriel."""

    fait: str = Field(description="Reprise d'un fait extrait (aucun fait nouveau).")
    consequence: str = Field(INCONNU, description="Conséquence business probable.")
    impact_sensoriel: str = Field(INCONNU, description="Effet attendu sur goût / odeur / formulation.")


class AnalyseImpact(BaseModel):
    """Sortie de l'appel 3b : hypothèses d'impact, toujours qualifiées."""

    resume: str = Field(INCONNU, description="Ce que ça implique, en une ou deux phrases.")
    chaine_impact: list[MaillonImpact] = Field(default_factory=list)
    categories_produits: list[str] = Field(default_factory=list, description="Ex. boissons, confiserie, parfumerie fine.")
    pays: list[str] = Field(default_factory=list, description="Codes ISO 3166-1 alpha-2 ou noms de pays.")
    ingredients_notes: list[str] = Field(default_factory=list, description="Ingrédients / notes touchés (vanille, menthe…).")
    confiance: Confiance = INCONNU

    @field_validator("confiance", mode="before")
    @classmethod
    def _norm_confiance(cls, v: object) -> object:
        return normalise_confiance(v)


class Analyse(BaseModel):
    """Analyse complète d'une actualité (3a + 3b)."""

    faits: list[Fait]
    hypotheses: AnalyseImpact


# ----------------------------------------------------------------------------------------
# Étape 4 : jointure déterministe avec la base d'entreprises (sans LLM)
# ----------------------------------------------------------------------------------------
class EntrepriseConcernee(BaseModel):
    nom: str
    raison: str = Field(description="Pourquoi elle est rattachée (critères de jointure qui ont matché).")
    source: str = Field(description="URL de la donnée de référence qui justifie le rattachement.")
    categories: list[str] = Field(default_factory=list)
    pays: list[str] = Field(default_factory=list)
    score: float = Field(description="Score de jointure (plus haut = plus pertinent).")


# ----------------------------------------------------------------------------------------
# Étape 5 : hypothèses d'opportunités (LLM)
# ----------------------------------------------------------------------------------------
class HypotheseOpportunite(BaseModel):
    entreprise: str = Field(description="Doit être une des entreprises fournies en entrée.")
    type_opportunite: TypeOpportunite = INCONNU
    division_givaudan: Division = INCONNU
    hypothese: str = Field(INCONNU, description="L'opportunité, formulée comme une hypothèse.")
    fenetre_action: str = Field(INCONNU, description="Ex. « 0-3 mois », « avant l'entrée en vigueur (2027) ».")
    question_client: str = Field(INCONNU, description="Question ouverte à poser au client.")
    faits_utilises: list[int] = Field(default_factory=list, description="Indices (0-based) des faits d'entrée utilisés.")
    confiance: Confiance = INCONNU

    @field_validator("confiance", mode="before")
    @classmethod
    def _norm_confiance(cls, v: object) -> object:
        return normalise_confiance(v)


class HypothesesOpportunites(BaseModel):
    """Sortie de l'étape 5 : un seul appel LLM par actualité."""

    opportunites: list[HypotheseOpportunite] = Field(default_factory=list)


# ----------------------------------------------------------------------------------------
# Sortie finale : une fiche par opportunité
# ----------------------------------------------------------------------------------------
class Signal(BaseModel):
    titre: str
    media: str
    date: datetime | None
    url: str


class Source(BaseModel):
    type: Literal["article", "base_entreprises"]
    libelle: str
    url: str


class FicheOpportunite(BaseModel):
    """Fiche commerciale prête à l'emploi. Les sources sont toujours présentes."""

    id: str
    signal: Signal
    famille: Famille | None
    score_filtre: float
    ce_que_ca_implique: str
    faits: list[Fait] = Field(description="Faits sourcés (issus de l'article).")
    chaine_impact: list[MaillonImpact] = Field(description="Hypothèses d'impact (non vérifiées).")
    entreprise: EntrepriseConcernee
    opportunite: HypotheseOpportunite
    question_a_poser: str
    confiance: Confiance
    sources: list[Source] = Field(min_length=1)

    @model_validator(mode="after")
    def _sources_obligatoires(self) -> "FicheOpportunite":
        if not any(s.type == "article" for s in self.sources):
            raise ValueError("Une fiche doit citer l'article source.")
        return self


class Diagnostic(BaseModel):
    """Traçabilité technique, utile à l'évaluation."""

    json_valides: int = 0
    json_invalides: int = 0
    rejets_validation: list[str] = Field(default_factory=list, description="Sorties LLM rejetées et pourquoi.")
    erreurs: list[str] = Field(default_factory=list)
    texte_analyse: Literal["article", "titre", "fourni"] | None = Field(
        None, description="Ce que le LLM a lu : texte de l'article, titre seul (confiance plafonnée à « moyen ») ou texte fourni.")
    duree_s: float | None = None


# ----------------------------------------------------------------------------------------
# Requêtes / réponses de l'API
# ----------------------------------------------------------------------------------------
class ActualiteEntree(BaseModel):
    """Actualité fournie directement (tous les champs sauf titre et url sont optionnels)."""

    titre: str = Field(min_length=3)
    url: HttpUrl
    media: str | None = None
    date: datetime | None = None
    langue: str | None = None
    texte: str | None = Field(None, description="Texte de l'article ; sinon il est récupéré depuis l'URL.")


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {"actualite": {
                "titre": "L'UE restreint l'usage d'un allergène parfumant dans les cosmétiques",
                "url": "https://example.org/article-ue-allergenes",
                "langue": "french",
            }},
            {"url": "https://example.org/article-cacao-prix"},
        ]
    })

    actualite: ActualiteEntree | None = None
    url: HttpUrl | None = Field(None, description="Raccourci : seule l'URL est fournie, le titre est récupéré.")
    forcer: bool = Field(False, description="Analyser même si le filtre écarte l'actualité.")

    @model_validator(mode="after")
    def _une_entree(self) -> "AnalyzeRequest":
        if (self.actualite is None) == (self.url is None):
            raise ValueError("Fournir exactement un des deux champs : `actualite` ou `url`.")
        return self


class AnalyzeResponse(BaseModel):
    mode: Mode
    actualite: Actualite
    filtre: ResultatFiltre
    analyse: Analyse | None = Field(None, description="Absente si l'actualité est écartée par le filtre.")
    entreprises: list[EntrepriseConcernee] = Field(default_factory=list)
    fiches: list[FicheOpportunite] = Field(default_factory=list)
    diagnostic: Diagnostic = Field(default_factory=Diagnostic)


class OpportunitiesResponse(BaseModel):
    mode: Mode
    requete: str | None
    periode: str
    langues: list[str]
    genere_le: datetime
    nb_collectees: int
    nb_retenues: int
    fiches: list[FicheOpportunite]
    diagnostic: Diagnostic = Field(default_factory=Diagnostic)


class HealthResponse(BaseModel):
    statut: Literal["ok", "degrade"]
    mode: Mode
    version: str
    ollama: dict
    reglages: dict
    collections: dict[str, int] = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)
