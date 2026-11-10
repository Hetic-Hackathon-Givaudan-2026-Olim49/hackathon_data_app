"""Charge story.txt dans la base vectorielle.

Usage :
    python ingest.py                 # charge story.txt
    python ingest.py autre_histoire.txt
"""

import re
import sys
from pathlib import Path

import config
import rag

# Un titre de chapitre ressemble à "IV. Le Sanctuaire Brisé" ou "Chapitre 1 : Le Grand Départ"
CHAPTER_RE = re.compile(r"^(?:[IVXLC]+\.\s+\S|chapitre\s+\w+\b)", re.IGNORECASE)


def split_into_sections(text: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """Retourne (titre de l'histoire, [(titre du chapitre, [paragraphes])])."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    title = ""
    if paragraphs and not CHAPTER_RE.match(paragraphs[0]):
        title = paragraphs.pop(0)

    sections: list[tuple[str, list[str]]] = []
    current_title, current = "Introduction", []
    for p in paragraphs:
        if CHAPTER_RE.match(p) and len(p) < 120:
            if current:
                sections.append((current_title, current))
            current_title, current = p, []
        else:
            current.append(p)
    if current:
        sections.append((current_title, current))
    return title, sections


def chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    """Regroupe les paragraphes en morceaux de ~CHUNK_SIZE caractères,
    avec un chevauchement de CHUNK_OVERLAP_PARAGRAPHS paragraphe(s)."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for p in paragraphs:
        if current and size + len(p) > config.CHUNK_SIZE:
            chunks.append("\n\n".join(current))
            overlap = current[-config.CHUNK_OVERLAP_PARAGRAPHS:] if config.CHUNK_OVERLAP_PARAGRAPHS else []
            current = list(overlap)
            size = sum(len(x) for x in current)
        current.append(p)
        size += len(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.STORY_PATH
    text = path.read_text(encoding="utf-8")
    story_title, sections = split_into_sections(text)

    ids, documents, metadatas = [], [], []
    for chap_idx, (chapter, paragraphs) in enumerate(sections):
        for chunk_idx, chunk in enumerate(chunk_paragraphs(paragraphs)):
            ids.append(f"{path.stem}-{chap_idx:02d}-{chunk_idx:03d}")
            documents.append(chunk)
            metadatas.append({
                "source": path.name,
                "story_title": story_title,
                "chapter": chapter,
                "chapter_index": chap_idx,
                "chunk_index": chunk_idx,
            })

    print(f"Histoire : {story_title or path.name}")
    print(f"{len(sections)} chapitres, {len(documents)} morceaux à indexer")
    print(f"Calcul des embeddings avec '{config.EMBED_MODEL}'...")

    # Le titre du chapitre est ajouté au texte embarqué pour aider la recherche
    to_embed = [f"{m['chapter']}\n{d}" for d, m in zip(documents, metadatas)]
    embeddings = rag.embed_documents(to_embed)

    collection = rag.get_collection()
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"OK : {collection.count()} morceaux dans la collection '{config.COLLECTION_NAME}' "
          f"({config.CHROMA_DIR})")


if __name__ == "__main__":
    main()
