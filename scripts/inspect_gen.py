"""Affiche réponse, citations et extraits de questions de generations.json.
Usage : python -m scripts.inspect_gen anp-caution pieces-justificatives"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
g = json.loads(Path("data/eval/generations.json").read_text(encoding="utf-8"))
for qid in sys.argv[1:]:
    r = g[qid]
    print(f"\n=== {qid} ===")
    print("RÉPONSE :", repr(r["answer"]))
    print("CITÉES  :", [c["id"] for c in r["cited"]], "| invalides :", r["invalid_citations"])
    for s in r["sources"]:
        print(f"  {s['id']} {s['doc'][:45]} | {s['title'][:60]} (p.{s['page']})")