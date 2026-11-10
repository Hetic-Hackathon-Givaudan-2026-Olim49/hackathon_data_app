"""Vide la base vectorielle pour pouvoir recharger l'histoire.

Usage :
    python clean.py
"""

import config
import rag


def main() -> None:
    client = rag.get_chroma_client()
    existing = [c.name for c in client.list_collections()]
    if config.COLLECTION_NAME not in existing:
        print(f"Rien à nettoyer : la collection '{config.COLLECTION_NAME}' n'existe pas.")
        return
    count = client.get_collection(config.COLLECTION_NAME).count()
    client.delete_collection(config.COLLECTION_NAME)
    print(f"Collection '{config.COLLECTION_NAME}' supprimée ({count} morceaux).")


if __name__ == "__main__":
    main()
