import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import QDRANT_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "marcheiq_chunks"
VECTOR_SIZE = 384  # Dimension de multilingual-e5-small

def get_client() -> QdrantClient:
    """Retourne un client Qdrant en mode local (un simple dossier)."""
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def create_collection(client: QdrantClient, recreate: bool = False):
    """Crée la collection si elle n'existe pas."""
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if exists and recreate:
        logger.info(f"Suppression de la collection existante {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
        exists = False
    
    if not exists:
        logger.info(f"Création de la collection {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        logger.info(f"Collection {COLLECTION_NAME} existe déjà")


def index_chunks(client: QdrantClient, chunks_path: Path, batch_size: int = 100):
    """Insère les chunks avec leurs embeddings dans Qdrant."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    logger.info(f"Indexation de {len(chunks)} chunks...")
    
    points = []
    for i, c in enumerate(chunks):
        points.append(PointStruct(
            id=i,
            vector=c["embedding"],
            payload={
                "chunk_id": c["chunk_id"],
                "doc": c["doc"],
                "title": c["title"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "source": c["source"],
                "text": c["text"],
            }
        ))
    
    # Insertion par batch
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info(f"Batch {i//batch_size + 1}/{(len(points)+batch_size-1)//batch_size} inséré")
    
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f" {info.points_count} points indexés dans {COLLECTION_NAME}")


if __name__ == "__main__":
    from src.config import DATA_PROCESSED
    
    input_file = DATA_PROCESSED / "chunks_with_embeddings.json"
    if not input_file.exists():
        logger.error(f"Fichier introuvable : {input_file}")
        exit(1)
    
    client = get_client()
    create_collection(client, recreate=True)
    index_chunks(client, input_file)
    
    print(f"\n Qdrant local prêt dans {QDRANT_PATH}")
    print(f" Prochaine étape : python -m src.retrieval.search")