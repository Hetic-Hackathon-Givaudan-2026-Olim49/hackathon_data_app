"""Garde-fous : le jeu de référence n'est jamais annoté par le code ; confiance plafonnée sur titre seul."""

import importlib.util
import json
from pathlib import Path

from veille.pipeline import cap_confidence

ROOT = Path(__file__).resolve().parent.parent


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_reference_set", ROOT / "eval" / "build_reference_set.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_set_leaves_annotations_empty_and_preserves_existing(tmp_path, monkeypatch):
    builder = _load_builder()
    source, target = tmp_path / "news_to_annotate.json", tmp_path / "reference_set.json"
    monkeypatch.setattr(builder, "SOURCE", source)
    monkeypatch.setattr(builder, "TARGET", target)
    news = [{"id": "n1", "titre": "Titre 1", "date": "2026-11-10", "media": "m.fr", "url": "https://m.fr/1", "langue": "french"}]
    source.write_text(json.dumps(news), encoding="utf-8")

    builder.main()
    [entry] = json.loads(target.read_text(encoding="utf-8"))["entrees"]
    assert entry["titre"] == "Titre 1" and entry["url"] == "https://m.fr/1"
    for field, empty in builder.ANNOTATION_FIELDS.items():
        assert entry[field] == empty  # aucun champ d'annotation n'est rempli

    # L'équipe annote, puis une nouvelle actualité est ajoutée : l'annotation doit survivre.
    data = json.loads(target.read_text(encoding="utf-8"))
    data["entrees"][0].update(pertinente="oui", annotateur="AB")
    target.write_text(json.dumps(data), encoding="utf-8")
    news.append({**news[0], "id": "n2", "url": "https://m.fr/2"})
    source.write_text(json.dumps(news), encoding="utf-8")

    builder.main()
    entries = {e["id"]: e for e in json.loads(target.read_text(encoding="utf-8"))["entrees"]}
    assert entries["n1"]["pertinente"] == "oui" and entries["n1"]["annotateur"] == "AB"
    assert entries["n2"]["pertinente"] == ""


def test_cap_confidence():
    assert cap_confidence("élevé") == "moyen"
    assert cap_confidence("spéculatif") == "spéculatif"
    assert cap_confidence("inconnu") == "inconnu"


def test_select_diverse_alternates_families():
    from veille.models import Actualite, ResultatFiltre
    from veille.pipeline import select_diverse

    def k(i, fam, score):
        return (Actualite(id=str(i), titre=str(i), url=f"https://x/{i}"),
                ResultatFiltre(retenue=True, famille=fam, score=score, seuil=0.5))

    kept = [k(1, "tendances_gout_odorat", 0.9), k(2, "tendances_gout_odorat", 0.8),
            k(3, "tendances_gout_odorat", 0.7), k(4, "reglementation_sensorielle", 0.6)]
    assert [i.id for i, _ in select_diverse(kept, 3)] == ["1", "4", "2"]


def _load_eval():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_eval_rejects_unknown_family():
    import pytest

    run_eval = _load_eval()
    run_eval.check_families([{"id": "a", "famille": "tendances_gout_odorat"}, {"id": "b", "famille": ""}])
    with pytest.raises(run_eval.AnnotationError, match="n9 → 'réglementation'"):
        run_eval.check_families([{"id": "n9", "famille": "réglementation"}])


def test_reference_set_families_are_valid():
    path = ROOT / "eval" / "reference_set.json"
    if path.exists():
        _load_eval().check_families(json.loads(path.read_text(encoding="utf-8"))["entrees"])
