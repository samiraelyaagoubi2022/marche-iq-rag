"""Compare les 3 variantes de retrieval : Dense, Hybride, Hybride+Reranker."""
import logging

from src.retrieval.search import search as dense
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank

logging.basicConfig(level=logging.WARNING)  # Réduire les logs INFO

QUERIES = [
    "Quelle est la caution provisoire du marché ANP ?",
    "Quelle est la date limite de dépôt des offres ?",
    "Quels sont les critères d'éligibilité des concurrents ?",
]


def show(title, results, score_key):
    print(f"\n  {title}")
    if not results:
        print("    (aucun résultat)")
        return
    for i, r in enumerate(results[:5], 1):
        score = r.get(score_key, 0)
        print(f"    #{i} [{score:.4f}] {r['doc'][:35]} | {r['title'][:35]} (p.{r['page_start']})")


if __name__ == "__main__":
    for q_idx, query in enumerate(QUERIES, 1):
        print("\n" + "=" * 80)
        print(f"[RECHERCHE] QUESTION {q_idx} : {query}")
        print("=" * 80)
        
        # 1. Dense
        dense_results = dense(query, top_k=5)
        show("DENSE (top 5)", dense_results, "score")
        
        # 2. Hybride
        hybrid_results = hybrid_search(query, top_k=5)
        show("HYBRIDE (top 5)", hybrid_results, "rrf_score")
        
        # 3. Hybride + Reranker
        candidates = hybrid_search(query, top_k=20)
        reranked = rerank(query, candidates, top_k=5)
        show("HYBRIDE + RERANKER (top 5)", reranked, "rerank_score")
    
    print("\n" + "=" * 80)
    print("[OK] Comparaison terminée.")
    print("=" * 80)