"""Recherche hybride : combine BM25 (mots-clés) et Dense (sens) via RRF."""
import json
import logging
import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from src.retrieval.search import search as dense_search
from src.config import DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# chunks.json (léger) suffit pour BM25 : pas besoin des embeddings
_CHUNKS_PATH = DATA_PROCESSED / "chunks.json"
_ALL_CHUNKS = None
_BM25 = None


def _tok(s: str) -> List[str]:
    """Tokenizer FR + AR : mots alphanumériques, minuscules, sans ponctuation."""
    return re.findall(r"\w+", s.lower())


def _load_bm25():
    global _ALL_CHUNKS, _BM25
    if _BM25 is not None:
        return
    logger.info("Chargement des chunks pour BM25...")
    with open(_CHUNKS_PATH, "r", encoding="utf-8") as f:
        _ALL_CHUNKS = json.load(f)
    _BM25 = BM25Okapi([_tok(c["text"]) for c in _ALL_CHUNKS])
    logger.info(f"BM25 initialisé sur {len(_ALL_CHUNKS)} chunks")


def bm25_search(query: str, top_k: int = 10, doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Recherche BM25 pure (mots-clés exacts), avec filtre optionnel sur le nom du document."""
    _load_bm25()
    scores = _BM25.get_scores(_tok(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if doc_filter:
        f = doc_filter.lower()
        order = [i for i in order if f in _ALL_CHUNKS[i]["doc"].lower()]
    indices = order[:top_k]

    return [
        {
            "chunk_id": _ALL_CHUNKS[i]["chunk_id"],
            "doc": _ALL_CHUNKS[i]["doc"],
            "title": _ALL_CHUNKS[i]["title"],
            "page_start": _ALL_CHUNKS[i]["page_start"],
            "text": _ALL_CHUNKS[i]["text"],
            "bm25_score": float(scores[i]),
        }
        for i in indices if scores[i] > 0
    ]


def hybrid_search(query: str, top_k: int = 5, k_rrf: int = 60,
                  doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Dense + BM25 fusionnés avec RRF (score = somme de 1 / (k + rang))."""
    _load_bm25()

    dense_results = dense_search(query, top_k=top_k * 2, doc_filter=doc_filter)
    bm25_results = bm25_search(query, top_k=top_k * 2, doc_filter=doc_filter)

    rrf_scores: Dict[str, float] = {}
    for rank, r in enumerate(dense_results):
        rrf_scores[r["chunk_id"]] = rrf_scores.get(r["chunk_id"], 0) + 1 / (k_rrf + rank + 1)
    for rank, r in enumerate(bm25_results):
        rrf_scores[r["chunk_id"]] = rrf_scores.get(r["chunk_id"], 0) + 1 / (k_rrf + rank + 1)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    id_to_result = {r["chunk_id"]: r for r in dense_results + bm25_results}
    results = []
    for cid in sorted_ids:
        r = id_to_result[cid].copy()
        r["rrf_score"] = rrf_scores[cid]
        results.append(r)
    return results


if __name__ == "__main__":
    query = "Quelle est la caution provisoire du marché ANP ?"
    print(f"\n[RECHERCHE] Question : {query}\n" + "=" * 70)

    for label, flt in (("SANS FILTRE", None), ("FILTRE 'ANP'", "ANP")):
        print(f"\n##### {label} #####")
        print("\n--- BM25 SEUL ---")
        for i, r in enumerate(bm25_search(query, top_k=5, doc_filter=flt), 1):
            print(f"#{i} [{r['bm25_score']:.2f}] {r['doc'][:40]} | {r['title'][:40]} (p.{r['page_start']})")
        print("\n--- HYBRIDE (RRF) ---")
        for i, r in enumerate(hybrid_search(query, top_k=5, doc_filter=flt), 1):
            print(f"#{i} [{r['rrf_score']:.4f}] {r['doc'][:40]} | {r['title'][:40]} (p.{r['page_start']})")