"""Évaluation de MarchéIQ sur data/eval/golden_dataset.json.

Deux niveaux :
  1. Métriques déterministes (gratuites, reproductibles, robustes aux chiffres) :
     retrieval (hit, rang, MRR, précision), exactitude de la réponse, refus attendus, citations.
  2. RAGAS (optionnel, --ragas) : faithfulness, relevancy, context recall / precision (LLM juge).

Usage, depuis la racine du projet :
    python -m scripts.eval_rag                       # génère + note
    python -m scripts.eval_rag --reuse               # réutilise data/eval/generations.json
    python -m scripts.eval_rag --ragas               # + métriques RAGAS
    python -m scripts.eval_rag --ids anp-caution,berd-ouverture
    python -m scripts.eval_rag --category arabe
    python -m scripts.eval_rag --min-pass 0.8        # code retour 1 si moins de 80 % de réussite

Format d'une question du golden dataset :
    id, category, question, reference (réponse complète, pour RAGAS)
    answerable            false = la réponse correcte est un refus
    expected_docs         sous-chaînes du nom de document (un seul doit correspondre)
    expected_titles       sous-chaînes du titre de section (optionnel)
    context_contains      textes qui doivent figurer dans le chunk pertinent (optionnel)
    answer_contains       textes attendus dans la réponse ; "a|b" = a OU b
    answer_not_contains   textes interdits dans la réponse (pièges)
Les comparaisons ignorent la casse, les accents, les espaces et les chiffres arabes-indiens.
"""
import argparse
import json
import logging
import os
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# [ALERTE] HF_HUB_OFFLINE=1 évite les timeouts réseau quand les modèles sont déjà en cache.
#    Si tu changes de modèle (embeddings ou reranker), mets HF_HUB_OFFLINE=0 dans .env
#    ou laisse la variable vide pour laisser Hugging Face télécharger.
if os.getenv("HF_HUB_OFFLINE") is None:
    os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_rag")

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
GOLDEN_PATH = EVAL_DIR / "golden_dataset.json"
GEN_PATH = EVAL_DIR / "generations.json"
REPORT_PATH = EVAL_DIR / "report.json"
RAGAS_PATH = EVAL_DIR / "ragas_results.json"

# [ALERTE] Pour RAGAS, un modèle léger suffit et coûte moins cher. Change-le dans .env :
#    RAGAS_JUDGE_MODEL=llama-3.1-8b-instant
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


# ---------------------------------------------------------------- normalisation
def norm(text: str) -> str:
    """Minuscules, sans accents ni espaces, chiffres arabes-indiens -> 0-9."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("\u2019", "'").translate(ARABIC_DIGITS).lower()
    return "".join(s.split())


def has_spec(text_norm: str, spec: str) -> bool:
    """spec = 'a|b' : vrai si l'une des variantes est présente."""
    return any(norm(alt) in text_norm for alt in spec.split("|") if alt.strip())


# ---------------------------------------------------------------- golden
def load_golden():
    if not GOLDEN_PATH.exists():
        sys.exit(f"[ERREUR] {GOLDEN_PATH} introuvable.")
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    ids = [g["id"] for g in golden]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        sys.exit(f"[ERREUR] ids dupliqués dans le golden dataset : {sorted(dup)}")
    for g in golden:
        if "question" not in g:
            sys.exit(f"[ERREUR] question manquante pour {g['id']}")
    return golden


# ---------------------------------------------------------------- génération
def run_generation(items, top_k: int, pause: float):
    from src.generation.generator import generate

    out = {}
    for i, it in enumerate(items, 1):
        logger.info("[%d/%d] %s", i, len(items), it["id"])
        res, err = None, "inconnue"
        for attempt in range(1, 4):
            try:
                t0 = time.time()
                res = generate(it["question"], top_k=top_k)
                res["latency_s"] = round(time.time() - t0, 1)
                break
            except Exception as e:
                err = str(e)
                logger.warning("Essai %d/3 échoué : %s", attempt, err[:150])
                time.sleep(15 * attempt)
        out[it["id"]] = res if res else {"error": err}
        time.sleep(pause)
    return out


# ---------------------------------------------------------------- notation déterministe
def is_relevant(item, src) -> bool:
    docs = item.get("expected_docs") or []
    if docs and not any(norm(d) in norm(src["doc"]) for d in docs):
        return False
    titles = item.get("expected_titles") or []
    if titles and not any(norm(t) in norm(src["title"]) for t in titles):
        return False
    text = norm(src["text"])
    return all(has_spec(text, spec) for spec in item.get("context_contains") or [])


def score(item, res):
    answerable = item.get("answerable", True)
    row = {"id": item["id"], "category": item.get("category", "-"), "answerable": answerable,
           "hit": None, "rank": None, "precision": None, "answer_ok": False, "cite_ok": None,
           "latency_s": res.get("latency_s"), "reasons": [], "passed": False}

    if "error" in res:
        row["reasons"].append(f"erreur de génération : {res['error'][:100]}")
        return row

    if not answerable:
        row["answer_ok"] = res.get("answerable") is False
        if not row["answer_ok"]:
            row["reasons"].append("aurait dû refuser (information absente des documents)")
        row["passed"] = row["answer_ok"]
        return row

    rel = [is_relevant(item, s) for s in res.get("sources", [])]
    row["hit"] = any(rel)
    row["rank"] = rel.index(True) + 1 if row["hit"] else None
    row["precision"] = sum(rel) / len(rel) if rel else 0.0
    if not row["hit"]:
        row["reasons"].append("aucun extrait pertinent récupéré")

    ans = norm(res["answer"])
    if not res.get("answerable", True):
        row["reasons"].append("refus à tort")
    else:
        manque = [s for s in item.get("answer_contains") or [] if not has_spec(ans, s)]
        interdit = [s for s in item.get("answer_not_contains") or [] if has_spec(ans, s)]
        if manque:
            row["reasons"].append(f"valeur attendue absente : {manque}")
        if interdit:
            row["reasons"].append(f"valeur interdite présente : {interdit}")
        row["answer_ok"] = not manque and not interdit

    row["cite_ok"] = bool(res.get("cited")) and not res.get("invalid_citations")
    if not row["cite_ok"] and res.get("answerable", True):
        row["reasons"].append("citations absentes ou invalides")

    row["passed"] = bool(row["hit"] and row["answer_ok"] and row["cite_ok"])
    return row


# ---------------------------------------------------------------- affichage
def mark(v):
    return "  -" if v is None else (" [OK]" if v else " [ERREUR]")


def print_report(rows):
    print("\n" + "=" * 92)
    print("[STATS] ÉVALUATION MARCHÉIQ (métriques déterministes)")
    print("=" * 92)
    print(f"{'ID':30} {'Catégorie':14} {'Doc':>4} {'Rang':>5} {'Préc':>5} {'Rép':>4} {'Cit':>4}  Statut")
    print("-" * 92)
    for r in rows:
        rang = "-" if r["rank"] is None else str(r["rank"])
        prec = "-" if r["precision"] is None else f"{r['precision']:.0%}"
        print(f"{r['id'][:30]:30} {r['category'][:14]:14} {mark(r['hit'])} {rang:>5} {prec:>5} "
              f"{mark(r['answer_ok'])} {mark(r['cite_ok'])}  {'[OK]' if r['passed'] else '[ERREUR]'}")

    echecs = [r for r in rows if not r["passed"]]
    if echecs:
        print("\n--- Détail des échecs ---")
        for r in echecs:
            print(f"■ {r['id']}")
            for reason in r["reasons"] or ["(échec sans détail)"]:
                print(f"    · {reason}")

    scored = [r for r in rows if r["hit"] is not None]
    pos = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    print("\n" + "=" * 92)
    print("[SYNTHESE] SYNTHÈSE")
    print("=" * 92)
    print(f"  Réussite globale        : {sum(r['passed'] for r in rows)}/{len(rows)} "
          f"({sum(r['passed'] for r in rows) / len(rows):.0%})")
    if scored:
        mrr = sum(1 / r["rank"] for r in scored if r["rank"]) / len(scored)
        print(f"  Retrieval hit@k         : {sum(r['hit'] for r in scored)}/{len(scored)}")
        print(f"  Retrieval MRR           : {mrr:.3f}")
        print(f"  Précision des extraits  : {sum(r['precision'] for r in scored) / len(scored):.0%}")
    if pos:
        print(f"  Réponses exactes        : {sum(r['answer_ok'] for r in pos)}/{len(pos)}")
        print(f"  Citations valides       : {sum(bool(r['cite_ok']) for r in pos)}/{len(pos)}")
    if neg:
        print(f"  Refus corrects          : {sum(r['answer_ok'] for r in neg)}/{len(neg)}")
    lat = [r["latency_s"] for r in rows if r["latency_s"]]
    if lat:
        print(f"  Latence moyenne         : {sum(lat) / len(lat):.1f} s")

    par_cat = defaultdict(list)
    for r in rows:
        par_cat[r["category"]].append(r["passed"])
    print("\n  Par catégorie :")
    for cat, vals in sorted(par_cat.items()):
        print(f"    {cat:16} {sum(vals)}/{len(vals)}")
    print("=" * 92)


# ---------------------------------------------------------------- RAGAS (optionnel)
def run_ragas(items, gens):
    try:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                                   LLMContextRecall, ResponseRelevancy)
    except ImportError as e:
        logger.error("RAGAS indisponible (%s). pip install ragas langchain-groq langchain-huggingface", e)
        return

    rows, ids = [], []
    for it in items:
        g = gens.get(it["id"])
        if not it.get("answerable", True) or not g or "error" in g or not g.get("answerable", True):
            continue
        if not it.get("reference"):
            logger.warning("%s : pas de 'reference', ignoré par RAGAS", it["id"])
            continue
        rows.append({"user_input": it["question"], "response": g["answer"],
                     "retrieved_contexts": [s["text"] for s in g["sources"]],
                     "reference": it["reference"]})
        ids.append(it["id"])
    if not rows:
        logger.error("Aucune question évaluable par RAGAS.")
        return

    # Modèle léger pour le juge RAGAS : rapide, gratuit, suffisant pour une note indicative.
    llm = LangchainLLMWrapper(ChatGroq(
        model=JUDGE_MODEL, temperature=0, max_tokens=4096,
        api_key=os.getenv("GROQ_API_KEY"),
    ))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        encode_kwargs={"normalize_embeddings": True},
    ))
    metrics = [
        Faithfulness(),
        # [ALERTE] Groq n'accepte pas n>1 pour la génération de questions.
        #    strictness=1 évite les erreurs, mais peut rendre la métrique instable.
        ResponseRelevancy(strictness=1),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]
    logger.info("RAGAS sur %d questions (juge : %s)...", len(rows), JUDGE_MODEL)
    result = evaluate(
        dataset=EvaluationDataset.from_list(rows), metrics=metrics, llm=llm, embeddings=emb,
        run_config=RunConfig(max_workers=2, timeout=240, max_retries=6),
        raise_exceptions=False, show_progress=True,
    )
    df = result.to_pandas()
    df.insert(0, "id", ids)
    entree = {"user_input", "response", "retrieved_contexts", "reference", "id"}
    cols = [c for c in df.columns if c not in entree]

    print("\n" + "=" * 92)
    print("[STATS] RAGAS (note par un LLM : indicatif, à croiser avec les métriques déterministes)")
    print("=" * 92)
    print(df[["id"] + cols].round(2).to_string(index=False))
    print("\n  Moyennes :")
    for c in cols:
        print(f"    {c:40} {df[c].mean():.3f}")
    df.drop(columns=["retrieved_contexts"]).to_json(RAGAS_PATH, orient="records", force_ascii=False, indent=2)
    print(f"\n[FICHIER] {RAGAS_PATH}")


# ---------------------------------------------------------------- main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Évaluation de MarchéIQ")
    ap.add_argument("--reuse", action="store_true", help="réutilise generations.json (ne régénère que les manquantes)")
    ap.add_argument("--ragas", action="store_true", help="ajoute les métriques RAGAS")
    ap.add_argument("--ids", help="ids séparés par des virgules")
    ap.add_argument("--category", help="une seule catégorie")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--pause", type=float, default=2.0, help="secondes entre deux questions")
    ap.add_argument("--min-pass", type=float, default=None, help="taux de réussite minimal (0-1)")
    args = ap.parse_args()

    items = load_golden()
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        items = [i for i in items if i["id"] in wanted]
    if args.category:
        items = [i for i in items if i.get("category") == args.category]
    if not items:
        sys.exit("[ERREUR] Aucune question sélectionnée.")
    logger.info("%d question(s) à évaluer", len(items))

    gens = {}
    if args.reuse and GEN_PATH.exists():
        gens = json.loads(GEN_PATH.read_text(encoding="utf-8"))
    todo = [i for i in items if i["id"] not in gens or "error" in gens[i["id"]]]
    if todo:
        gens.update(run_generation(todo, args.top_k, args.pause))
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        GEN_PATH.write_text(json.dumps(gens, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [score(it, gens[it["id"]]) for it in items]
    print_report(rows)
    REPORT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[FICHIER] {REPORT_PATH}\n[FICHIER] {GEN_PATH}")

    if args.ragas:
        run_ragas(items, gens)

    if args.min_pass is not None:
        taux = sum(r["passed"] for r in rows) / len(rows)
        if taux < args.min_pass:
            sys.exit(1)


if __name__ == "__main__":
    main()