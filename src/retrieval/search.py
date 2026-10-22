import atexit
import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

from src.vectorstore.qdrant_client import get_client, COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "intfloat/multilingual-e5-small"

_MODEL = None
_CLIENT = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        logger.info(f"Chargement du modèle {MODEL_NAME}...")
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _get_qdrant():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = get_client()
        atexit.register(_CLIENT.close)
    return _CLIENT


def search(query: str, top_k: int = 5, doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Recherche les chunks les plus pertinents.
    
    Args:
        query: la question
        top_k: nombre de résultats
        doc_filter: si fourni, ne cherche que dans les documents dont le nom contient cette chaîne
    """
    model = _get_model()
    client = _get_qdrant()
    
    query_vector = model.encode(f"query: {query}", normalize_embeddings=True).tolist()
    
    query_filter = None
    if doc_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchText
        query_filter = Filter(
            must=[FieldCondition(key="doc", match=MatchText(text=doc_filter))]
        )
        logger.info(f"Filtre appliqué : doc contient '{doc_filter}'")
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    ).points
    
    return [
        {
            "score": r.score,
            "chunk_id": r.payload["chunk_id"],
            "doc": r.payload["doc"],
            "title": r.payload["title"],
            "page_start": r.payload["page_start"],
            "text": r.payload["text"],
        }
        for r in results
    ]


if __name__ == "__main__":
    query = "Quelle est la caution provisoire du marché ANP ?"
    print(f"\n[RECHERCHE] Question : {query}\n")
    print("=" * 70)
    
    print("\n--- SANS FILTRE ---")
    for i, r in enumerate(search(query, top_k=5), 1):
        print(f"#{i} [{r['score']:.4f}] {r['doc'][:50]} | {r['title'][:40]} (p.{r['page_start']})")
    
    print("\n--- AVEC FILTRE 'ANP' ---")
    for i, r in enumerate(search(query, top_k=5, doc_filter="ANP"), 1):
        print(f"#{i} [{r['score']:.4f}] {r['doc'][:50]} | {r['title'][:40]} (p.{r['page_start']})")