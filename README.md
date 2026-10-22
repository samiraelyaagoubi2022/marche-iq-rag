
# MarchéIQ — Assistant RAG pour l'analyse d'appels d'offres publics marocains

MarchéIQ est un assistant intelligent basé sur l'architecture RAG (Retrieval-Augmented Generation). Il permet d'interroger en langage naturel des documents d'appels d'offres publics (CPS, Règlement de Consultation, Avis) et de fournir des réponses précises, toujours accompagnées des sources exactes (document, page et passage).

L'objectif est de faire gagner un temps précieux aux entreprises et consultants lors de l'analyse de dossiers complexes.

## Fonctionnalités principales

- **Recherche hybride** : combine BM25 (mots-clés exacts) et recherche dense (sens) via RRF, avec un reranker cross-encoder multilingue.
- **Filtrage par métadonnées** : quand la question mentionne un document (ANP, BERD, CPS, FNUAP), la recherche ne s'effectue que dans ce document.
- **Génération de réponses sourcées** : chaque affirmation renvoie à un passage numéroté du document original, avec surlignage des mots-clés.
- **Interface Streamlit** : questions d'exemple, historique, export Markdown, feedback utilisateur.
- **Fiabilité** : alerte visuelle si le modèle génère une réponse sans citer de source.

## Architecture et pipeline

1. **Ingestion** : lecture et nettoyage des PDF (natif via PyMuPDF, OCR via Tesseract pour les scans), détection des QR Codes.
2. **Découpage (chunking)** : séparation des documents en passages, avec conservation du numéro de page. Chunking adaptatif : par article si le document est structuré, par paragraphe sinon.
3. **Indexation** : transformation de chaque passage en vecteur (embeddings multilingues E5, 384 dimensions) et stockage dans Qdrant.
4. **Recherche** : récupération des passages les plus proches via recherche hybride (BM25 + dense + RRF), puis reranking avec un cross-encoder multilingue.
5. **Génération** : le modèle (via Groq) rédige la réponse en se basant uniquement sur les passages récupérés et ajoute les citations.

## Stack technique

- **Langage** : Python 3.13
- **Interface web** : Streamlit
- **Modèle de langage (LLM)** : Groq API (`openai/gpt-oss-120b`)
- **Base de données vectorielle** : Qdrant (mode local)
- **Embeddings** : `intfloat/multilingual-e5-small` (384 dim)
- **Reranker** : `BAAI/bge-reranker-v2-m3`
- **Recherche par mots-clés** : `rank-bm25`
- **OCR** : Tesseract
- **Tests** : pytest (23 tests unitaires)

## Installation et lancement en local

### Prérequis

- Python 3.11+
- Tesseract OCR installé (avec les langues `fra` et `ara`)
- Une clé API Groq (gratuite) disponible sur `console.groq.com`

### Cloner le projet

```bash
git clone https://github.com/votre-nom-utilisateur/marche-iq.git
cd marche-iq
```

### Créer un environnement virtuel et installer les dépendances

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

### Configuration de la clé API

Créez un fichier `.env` à la racine du projet :

```
GROQ_API_KEY=votre_cle_api_groq_ici
```

### Lancer l'ingestion et l'indexation

```bash
python -m scripts.run_ingestion
python -m src.embedding.embedder
python -m src.vectorstore.qdrant_client
```

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`.

## Structure du projet

```
marche-iq/
├── app.py                     # Interface Streamlit
├── requirements.txt           # Dépendances Python
├── .env                       # Variables d'environnement
├── .gitignore
│
├── data/
│   ├── raw/                   # PDF originaux
│   ├── processed/             # Chunks et embeddings
│   ├── qdrant/                # Base vectorielle locale
│   └── eval/                  # Golden dataset et résultats
│
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── parser.py          # Extraction natif + OCR + QR Codes
│   │   └── chunker.py         # Chunking adaptatif
│   ├── embedding/
│   │   └── embedder.py        # Embeddings E5
│   ├── vectorstore/
│   │   └── qdrant_client.py   # Indexation Qdrant
│   ├── retrieval/
│   │   ├── search.py          # Recherche dense
│   │   ├── hybrid_search.py   # Recherche hybride BM25 + dense
│   │   └── reranker.py        # Cross-encoder
│   └── generation/
│       └── generator.py       # Génération citée via Groq
│
├── scripts/
│   ├── run_ingestion.py
│   ├── validate_ingestion.py
│   └── eval_rag.py
│
├── tests/
│   └── test_ingestion.py
│
└── logs/
    └── feedback.jsonl
```

## Évaluation

Le projet a été évalué sur un golden dataset de 15 questions (6 catégories : factuel, rectificatif, sans document, arabe, négatif).

| Métrique | Valeur |
|---|---|
| Réussite globale | 93% |
| Réponses exactes | 100% |
| Citations valides | 100% |
| Refus corrects (questions négatives) | 100% |
| MRR retrieval | 0.923 |

Les résultats détaillés sont dans `data/eval/report.json`. Les métriques déterministes (hit, MRR, citations) sont privilégiées par rapport à RAGAS, plus variables et coûteuses.

## Auteur

Samira EL YAAGOUBI
