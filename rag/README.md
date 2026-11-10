# Story RAG API

Pose des questions sur une histoire pour enfants (`story.txt`) via une API FastAPI.
Les passages pertinents sont retrouvés dans une base vectorielle (ChromaDB), puis un
LLM Ollama rédige la réponse.

## Fichiers du projet

| Fichier            | Rôle |
|--------------------|------|
| `story.txt`        | L'histoire pour enfants, source des réponses. Le titre est sur la première ligne, les chapitres commencent par « Chapitre N : … » ou « I. … ». |
| `ingest.py`        | Charge l'histoire dans la base : découpe le texte par chapitre en morceaux d'environ 800 caractères, calcule leurs embeddings avec Ollama et les enregistre dans ChromaDB. Peut être relancé sans créer de doublons. Accepte un autre fichier en argument : `python ingest.py autre.txt`. |
| `clean.py`         | Vide la base (supprime la collection ChromaDB), pour recharger l'histoire ou changer de modèle d'embeddings. |
| `api.py`           | L'API FastAPI avec son Swagger : retrouve les passages proches de la question, puis demande au LLM d'y répondre simplement, comme à un enfant. Contient aussi le prompt système. |
| `list.py`          | Affiche le contenu de la base : modèle d'embeddings utilisé, nombre de morceaux, puis chaque morceau (identifiant, taille, aperçu du texte) groupé par chapitre. Avec `--full`, affiche le texte complet. |
| `rag.py`           | Code partagé par les scripts : connexion à ChromaDB, calcul des embeddings et appel du LLM via Ollama. |
| `config.py`        | Tous les réglages (modèles, chemins, taille des morceaux, nombre de passages), modifiables par variables d'environnement. |
| `requirements.txt` | Dépendances Python : FastAPI, Uvicorn, Ollama, ChromaDB. |
| `.gitignore`       | Exclut de git le venv, la base `chroma_db/`, les caches Python et les fichiers locaux. |
| `chroma_db/`       | Base vectorielle créée par `ingest.py` (non versionnée). |

## Installation

Prérequis : Python 3.10+ et [Ollama](https://ollama.com) lancé en local.

### 1. Créer et activer un environnement virtuel

```bash
python -m venv .venv
```

Puis l'activer selon votre terminal :

| Terminal                  | Commande                          |
|---------------------------|-----------------------------------|
| Windows PowerShell        | `.venv\Scripts\Activate.ps1`      |
| Windows cmd               | `.venv\Scripts\activate.bat`      |
| Git Bash (Windows)        | `source .venv/Scripts/activate`   |
| macOS / Linux             | `source .venv/bin/activate`       |

Si PowerShell refuse d'exécuter le script d'activation, autorisez-le pour votre
utilisateur une fois pour toutes :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Le nom `(.venv)` apparaît alors au début de l'invite. Pour sortir du venv : `deactivate`.

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger les modèles Ollama

```bash
ollama pull nomic-embed-text   # modèle d'embeddings
ollama pull qwen2.5:1.5b         # LLM (ou un autre, voir Configuration)
```

## Utilisation

```bash
python ingest.py        # charge story.txt dans la base (chroma_db/)
python list.py          # affiche le contenu de la base (--full pour le texte complet)
python clean.py         # vide la base
python api.py           # lance l'API sur http://127.0.0.1:8000
```

Swagger : http://127.0.0.1:8000/docs

Pour recharger après modification de l'histoire : `python clean.py` puis `python ingest.py`.

## Endpoints

| Méthode | Route     | Rôle                                                   |
|---------|-----------|--------------------------------------------------------|
| GET     | `/health` | Nombre de morceaux en base et modèles utilisés         |
| POST    | `/search` | Passages retrouvés seulement (sans LLM), pour déboguer |
| POST    | `/ask`    | Réponse du LLM + passages sources                      |

Corps de requête : `{"question": "Comment s'appelle le cheval de Léo ?", "top_k": 4}`

## Configuration (variables d'environnement)

| Variable      | Défaut              |
|---------------|---------------------|
| `LLM_MODEL`   | `qwen2.5:1.5b`        |
| `EMBED_MODEL` | `nomic-embed-text`  |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `CHROMA_DIR`  | `./chroma_db`       |
| `STORY_PATH`  | `./story.txt`       |
| `CHUNK_SIZE`  | `800` (caractères)  |
| `TOP_K`       | `4`                 |

Si vous changez `EMBED_MODEL`, relancez `clean.py` puis `ingest.py` : les vecteurs
de deux modèles différents ne sont pas compatibles.
