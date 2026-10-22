
# 📑 MarchéIQ - Assistant RAG pour l'analyse d'appels d'offres

MarchéIQ est un assistant intelligent basé sur l'architecture **RAG (Retrieval-Augmented Generation)**. Il permet d'interroger en langage naturel des documents d'appels d'offres publics (CPS, Règlement de Consultation, Avis) et de fournir des réponses précises, **toujours accompagnées des sources exactes** (document, page et passage).

L'objectif est de faire gagner un temps précieux aux entreprises et consultants lors de l'analyse de dossiers complexes.

## ✨ Fonctionnalités principales

- **Recherche sémantique** : Trouve les passages pertinents même si les mots exacts ne sont pas utilisés.
- **Génération de réponses sourcées** : Chaque affirmation renvoie à un passage numéroté du document original (avec surlignage des mots-clés).
- **Interface intuitive** : Développée avec Streamlit, elle propose des questions d'exemple et un historique des conversations.
- **Exportation** : Possibilité de télécharger la réponse et ses sources au format Markdown.
- **Système de feedback** : Boutons "Utile / Pas utile" pour enregistrer les retours utilisateurs dans un fichier de logs.
- **Fiabilité** : Alerte visuelle (en rouge) si le modèle génère une réponse sans citer de source.

## 🏗️ Architecture et Pipeline

Le projet repose sur un pipeline en 5 étapes :

1. **Ingestion** : Lecture et nettoyage des PDF (CPS, RC, avis).
2. **Découpage (Chunking)** : Séparation des documents en passages (chunks) avec conservation du numéro de page.
3. **Indexation** : Transformation de chaque passage en vecteur (embeddings) et stockage dans une base vectorielle.
4. **Recherche (Retrieval)** : Récupération des passages les plus proches de la question de l'utilisateur.
5. **Génération (LLM)** : Le modèle (via Groq) rédige la réponse en se basant *uniquement* sur les passages récupérés et ajoute les citations.

## 🛠️ Stack Technique

- **Langage** : Python 3.9+
- **Interface Web** : Streamlit
- **Modèle de Langage (LLM)** : Groq API (Llama 3 ou Mixtral)
- **Base de données vectorielle** : ChromaDB (ou FAISS)
- **Gestion des variables d'environnement** : `python-dotenv`

## 🚀 Installation et Lancement en local

### 1. Prérequis
- Python installé sur votre machine.
- Une clé API Groq (gratuite) disponible sur [console.groq.com](https://console.groq.com).

### 2. Cloner le projet
```bash
git clone https://github.com/votre-nom-utilisateur/marche-iq.git
cd marche-iq
```

### 3. Créer un environnement virtuel et installer les dépendances
```bash
python -m venv venv
source venv/Scripts/activate  # Sur Windows (Git Bash)
# ou source venv/bin/activate  # Sur Mac/Linux

pip install -r requirements.txt
```
*(Si vous n'avez pas de `requirements.txt`, installez manuellement : `pip install streamlit python-dotenv groq`)*

### 4. Configuration de la clé API
Créez un fichier `.env` à la racine du projet et ajoutez votre clé :
```env
GROQ_API_KEY=votre_cle_api_groq_ici
```

### 5. Lancer l'application
```bash
streamlit run app.py
```
*(Remplacez `app.py` par le nom de votre fichier principal si nécessaire).*

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

## 📂 Structure du projet

```text
marche-iq/
│
├── app.py                  # Point d'entrée de l'interface Streamlit
├── requirements.txt        # Dépendances Python
├── .env                    # Variables d'environnement (ignoré par Git)
├── .gitignore              # Fichiers à ignorer par Git
│
├── data/                   # Dossier contenant les PDF et l'index
│   ├── raw/                # PDF originaux
│   └── eval/               # Résultats d'évaluation (ragas_results.json)
│
├── src/                    # Code source métier
│   ├── config.py           # Configuration globale
│   ├── ingestion/          # Scripts de lecture et découpage des PDF
│   │   ├── parser.py
│   │   └── chunker.py
│   ├── retrieval/          # Logique de recherche vectorielle
│   └── generation/         # Logique d'appel au LLM (Groq)
│
├── scripts/                # Scripts utilitaires
│   └── run_ingestion.py    # Script pour lancer l'ingestion des PDF
│
├── tests/                  # Tests unitaires
│   └── test_ingestion.py
│
└── logs/                   # Logs de feedback (feedback.jsonl)
```

## 🧪 Évaluation
Le projet inclut un dossier `data/eval/` avec des résultats d'évaluation générés via le framework **Ragas**, permettant de mesurer la fidélité (faithfulness) et la pertinence des réponses générées par rapport aux contextes récupérés.

## 👨‍💻 Auteur
**[Samira EL YAAGOUBI]**
