"""Mesure le recall@5 des 3 variantes de retrieval sur un golden dataset."""
import json
from pathlib import Path

from src.retrieval.search import search as dense
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank

GOLDEN = Path("data/eval/golden_dataset.json")

if not GOLDEN.exists():
    print("[ERREUR] Crée data/eval/golden_dataset.json d'abord")
    print("Format : [{\"question\": ..., \"expected_chunk_ids\": [...]}, ...]")
    exit(1)

with open(GOLDEN, encoding="utf-8") as f:
    golden = json.load(f)

def recall_at_k(results, expected_ids, k=5):
    retrieved = {r["chunk_id"] for r in results[:k]}
    expected = set(expected_ids)
    return len(retrieved & expected) / len(expected) if expected else 0


variants = {
    "Dense": lambda q: dense(q, top_k=5),
    "Hybride": lambda q: hybrid_search(q, top_k=5),
    "Hybride+Reranker": lambda q: rerank(q, hybrid_search(q, top_k=20), top_k=5),
}

print(f"{'Variante':20} {'Recall@5':>10}")
print("-" * 32)
for name, fn in variants.items():
    recalls = []
    for item in golden:
        results = fn(item["question"])
        recalls.append(recall_at_k(results, item["expected_chunk_ids"]))
    avg = sum(recalls) / len(recalls)
    print(f"{name:20} {avg:.2%}")