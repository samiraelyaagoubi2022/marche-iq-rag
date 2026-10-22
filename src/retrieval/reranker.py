"""Reranker : re-trie les résultats avec un cross-encoder."""
import logging
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# BGE reranker multilingue (bien pour FR + AR)
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_RERANKER = None


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        logger.info(f"Chargement du reranker {MODEL_NAME}...")
        _RERANKER = CrossEncoder(MODEL_NAME)
    return _RERANKER


def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Re-trie les chunks avec un cross-encoder.
    
    Le cross-encoder prend (query, chunk) ensemble et produit un score de pertinence.
    C'est plus lent que le bi-encoder (dense), mais plus précis.
    """
    if not chunks:
        return []
    
    reranker = _get_reranker()
    
    # Paires (query, texte du chunk)
    pairs = [(query, c["text"]) for c in chunks]
    
    # Scores de pertinence
    scores = reranker.predict(pairs)
    
    # Attacher les scores et trier
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


if __name__ == "__main__":
    from src.retrieval.hybrid_search import hybrid_search
    
    query = "Quelle est la caution provisoire du marché ANP ?"
    print(f"\n[RECHERCHE] Question : {query}\n")
    print("=" * 70)
    
    # Étape 1 : hybride top-20
    candidates = hybrid_search(query, top_k=20)
    print(f"\n--- HYBRIDE (top 20 candidats) ---")
    for i, r in enumerate(candidates[:10], 1):
        print(f"#{i} [{r['rrf_score']:.4f}] {r['doc'][:40]} | {r['title'][:40]} (p.{r['page_start']})")
    
    # Étape 2 : rerank top-5
    reranked = rerank(query, candidates, top_k=5)
    print(f"\n--- HYBRIDE + RERANKER (top 5) ---")
    for i, r in enumerate(reranked, 1):
        print(f"#{i} [rerank={r['rerank_score']:.4f}] {r['doc'][:40]} | {r['title'][:40]} (p.{r['page_start']})")
    
    print("\n" + "=" * 70)