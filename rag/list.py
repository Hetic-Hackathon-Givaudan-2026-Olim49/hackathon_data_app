"""Liste le contenu de la base vectorielle.

Usage :
    python list.py          # aperçu de chaque morceau
    python list.py --full   # texte complet de chaque morceau
"""

import argparse

import config
import rag

PREVIEW_CHARS = 120


def main() -> None:
    parser = argparse.ArgumentParser(description="Liste les morceaux stockés dans la base.")
    parser.add_argument("--full", action="store_true", help="affiche le texte complet de chaque morceau")
    args = parser.parse_args()

    client = rag.get_chroma_client()
    if config.COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        print(f"La collection '{config.COLLECTION_NAME}' n'existe pas : lance `python ingest.py`.")
        return

    collection = client.get_collection(config.COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    rows = sorted(zip(data["ids"], data["documents"], data["metadatas"]), key=lambda r: r[0])

    print(f"Collection : {config.COLLECTION_NAME} ({config.CHROMA_DIR})")
    print(f"Modèle d'embeddings : {collection.metadata.get('embed_model', 'inconnu')}")
    print(f"Morceaux : {len(rows)}")

    current_chapter = None
    for id_, doc, meta in rows:
        if meta.get("chapter") != current_chapter:
            current_chapter = meta.get("chapter")
            print(f"\n=== {current_chapter} ===")
        if args.full:
            print(f"\n[{id_}] ({len(doc)} caractères)\n{doc}")
        else:
            preview = " ".join(doc.split())
            if len(preview) > PREVIEW_CHARS:
                preview = preview[:PREVIEW_CHARS] + "..."
            print(f"[{id_}] ({len(doc)} car.) {preview}")


if __name__ == "__main__":
    main()
