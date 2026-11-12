"""Tests du filtre (étape 2) avec des embeddings factices (aucun appel à Ollama)."""

from veille import settings
from veille.filter import HORS_PERIMETRE, FamilyFilter, reference_rows
from veille.models import Actualite

# Espace à 3 dimensions : x = réglementation, y = matières premières, z = hors périmètre
REFS = [
    ("reglementation_sensorielle", "french", [1.0, 0.0, 0.0]),
    ("matieres_premieres_aromatiques", "french", [0.0, 1.0, 0.0]),
    (HORS_PERIMETRE, "french", [0.0, 0.0, 1.0]),
]
VECTORS = {
    "interdiction additif": [0.9, 0.1, 0.1],
    "prix vanille": [0.1, 0.8, 0.0],
    "match de foot": [0.3, 0.0, 0.9],      # au-dessus du seuil en apparence, mais plus proche du hors périmètre
    "vague": [0.3, 0.3, 0.25],  # cosinus ≈ 0.62 avec chaque famille
}


def _items(*titles):
    return [Actualite(id=t, titre=t, url=f"https://x/{i}", langue="french") for i, t in enumerate(titles)]


def _filter():
    return FamilyFilter(REFS, lambda texts: [VECTORS[t] for t in texts])


def test_family_and_score(monkeypatch):
    monkeypatch.setattr(settings, "FILTER_THRESHOLD", 0.6)
    a, b = _filter().apply(_items("interdiction additif", "prix vanille"))
    assert a.retenue and a.famille == "reglementation_sensorielle" and a.score > 0.9
    assert b.retenue and b.famille == "matieres_premieres_aromatiques"
    assert set(a.scores_par_famille) == {"reglementation_sensorielle", "matieres_premieres_aromatiques", HORS_PERIMETRE}


def test_threshold_rejects_vague_news(monkeypatch):
    monkeypatch.setattr(settings, "FILTER_THRESHOLD", 0.7)
    [res] = _filter().apply(_items("vague"))
    assert not res.retenue and res.seuil == 0.7 and res.famille is not None


def test_out_of_scope_wins(monkeypatch):
    monkeypatch.setattr(settings, "FILTER_THRESHOLD", 0.2)
    [res] = _filter().apply(_items("match de foot"))
    assert not res.retenue  # le score hors périmètre est plus élevé


def test_threshold_override_by_language(monkeypatch):
    monkeypatch.setattr(settings, "FILTER_THRESHOLD", 0.99)
    monkeypatch.setitem(settings.FILTER_THRESHOLD_BY_LANG, "french", 0.5)
    [res] = _filter().apply(_items("prix vanille"))
    assert res.retenue and res.seuil == 0.5


def test_reference_rows_respect_languages():
    families = {"reglementation_sensorielle": {"descriptions": {"french": ["a", "b"], "english": ["c"]}}}
    assert [r[3] for r in reference_rows(families, ["french"])] == ["a", "b"]
    assert len(reference_rows(families, ["french", "english"])) == 3


def test_empty_input():
    assert _filter().apply([]) == []
