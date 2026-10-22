"""Validation complète de data/processed/chunks.json avant les embeddings.

Usage (depuis la racine du projet) :
    python -m scripts.validate_ingestion
    python -m scripts.validate_ingestion --strict            # les alertes font échouer (code 1)
    python -m scripts.validate_ingestion --exemples 5        # plus d'exemples par problème
    python -m scripts.validate_ingestion --doc Decret        # un seul document
    python -m scripts.validate_ingestion --json rapport.json # export machine

Niveaux : ERREUR (bloquant) · ALERTE (à corriger) · INFO (à connaître).
Code retour : 0 = OK, 1 = erreurs (ou alertes avec --strict).
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from src.config import DATA_PROCESSED, DATA_RAW

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    from src.ingestion import chunker
    CHUNK_SIZE = chunker.CHUNK_SIZE
    MIN_CHARS = chunker.MIN_CHUNK_CHARS
    HEADING_RE = chunker.HEADING_RE
    PAGE_ATOMIQUE_MAX = getattr(chunker, "PAGE_ATOMIQUE_MAX", 2400)
    AVIS_MAX_PAGES = getattr(chunker, "AVIS_MAX_PAGES", 3)
    _EST_SOMMAIRE = getattr(chunker, "_est_sommaire", None)
except Exception:
    CHUNK_SIZE, MIN_CHARS = 1200, 30
    PAGE_ATOMIQUE_MAX, AVIS_MAX_PAGES, _EST_SOMMAIRE = 2400, 3, None
    HEADING_RE = re.compile(r"^\s*(art(icle)?\.?|chapitre|titre|section|المادة)\b", re.I)

CTX_RE = re.compile(r"\A\[Document : [^\n]*\]\n")

def txt(c):
    return CTX_RE.sub("", str(c.get("text", "")), count=1)

CHUNKS_FILE = DATA_PROCESSED / "chunks.json"

CODES = {
    "SCHEMA":        ("ERREUR", "Champs manquants ou invalides", "Vérifier le dict retourné par chunk_document()."),
    "ID_DUP":        ("ERREUR", "chunk_id dupliqué", "Les ids doivent être uniques : doc::numéro."),
    "VIDE":          ("ERREUR", "Chunk vide", "Filtrer les chunks sans texte dans chunk_document()."),
    "ENCODAGE":      ("ERREUR", "Caractères corrompus / mojibake", "Mauvais décodage : vérifier l'extraction ou la langue OCR."),
    "TROP_PETIT":    ("ERREUR", "Chunk sous le minimum du chunker", "Le filtre MIN_CHUNK_CHARS n'est pas appliqué."),
    "TROP_GRAND":    ("ERREUR", "Chunk démesuré (>3000 car.)", "split_long() n'a pas été appliqué à ce bloc."),
    "PAGES_PERDUES": ("ERREUR", "Pages avec contenu sans aucun chunk", "Contenu perdu : page ignorée par le parser ou le chunker."),
    "COUVERTURE_BASSE": ("ERREUR", "Moins de 60 % du texte du PDF retrouvé", "Une grande partie du document a été perdue."),
    "GRAND":         ("ALERTE", "Chunk plus grand que CHUNK_SIZE", "Vérifier split_long()."),
    "MICRO":         ("ALERTE", "Micro-chunk (<50 car.)", "Fusionner les petits blocs avec le suivant/précédent."),
    "TITRE_SEUL":    ("ALERTE", "Chunk sans corps (titre seul / titre répété)", "Ne pas créer de chunk pour un titre sans contenu : le rattacher au texte qui suit."),
    "SOMMAIRE":      ("ALERTE", "Chunk issu d'une table des matières", "Détecter et ignorer le sommaire (points de suite, n° de page en fin de ligne), sinon il pollue la recherche."),
    "BRUIT":         ("ALERTE", "Peu de lettres (bruit, symboles)", "Nettoyer les séparateurs / lignes de points / tableaux dégénérés."),
    "OCR_DOUTEUX":   ("ALERTE", "OCR de mauvaise qualité", "Augmenter la résolution (dpi), vérifier les langues Tesseract (ara+fra)."),
    "DOUBLON":       ("ALERTE", "Chunks identiques dans le même document", "Dédupliquer avant l'indexation."),
    "ENTETE_REPETE": ("ALERTE", "Ligne répétée dans de nombreux chunks (en-tête/pied de page)", "Retirer les en-têtes/pieds de page récurrents au nettoyage."),
    "NUM_PAGE":      ("ALERTE", "Numéros de page isolés dans le texte", "Supprimer les lignes composées uniquement d'un numéro."),
    "ORDRE_PAGES":   ("ALERTE", "Ordre des pages incohérent", "Les chunks doivent suivre l'ordre du document."),
    "ARTICLES_MANQUANTS": ("ALERTE", "Trous dans la numérotation des articles", "Article non détecté, fusionné ou perdu : vérifier HEADING_RE sur ces numéros."),
    "SANS_TITRES":   ("ALERTE", "Document long sans structure détectée", "Les titres 'Article' ne sont pas reconnus dans ce document."),
    "FRAGMENTATION": ("ALERTE", "Sur-fragmentation", "Trop de petits chunks : relever MIN_CHUNK_CHARS ou fusionner."),
    "COUVERTURE_HAUTE": ("ALERTE", "Texte dupliqué (>140 % du PDF)", "Titres répétés dans le corps ou chevauchement excessif."),
    "COUVERTURE_MOYENNE": ("ALERTE", "Texte partiellement perdu (<85 % du PDF)", "Du texte natif n'apparaît dans aucun chunk."),
    "CESURE":        ("INFO", "Mots coupés par un tiret en fin de ligne", "Recoller 'mot-\\nsuite' au nettoyage (optionnel)."),
    "TABLEAU_NUM":   ("INFO", "Chunks majoritairement numériques (tableaux/prix)", "Normal pour un bordereau de prix, mais peu utile en recherche sémantique."),
    "DOUBLON_INTER": ("INFO", "Texte identique dans plusieurs documents", "Normal pour des avis rectificatifs."),
    "PAGES_VIDES":   ("INFO", "Pages réellement vides (blanches)", "Aucune action."),
    "PDF_ABSENT":    ("INFO", "PDF source introuvable : contrôles de couverture sautés", ""),
}
ICONE = {"ERREUR": "[ERREUR]", "ALERTE": "[ALERTE] ", "INFO": "[INFO] "}
ORDRE = {"ERREUR": 0, "ALERTE": 1, "INFO": 2}

REQUIS = {"chunk_id": str, "doc": str, "title": str, "page_start": int,
          "page_end": int, "source": str, "text": str}
MOJIBAKE = re.compile(r"\ufffd|Ã[\x80-\xbf]|â€|Â[\xa0-\xbf]")
CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
POINTS_SUITE = re.compile(r"(\.{4,}|…{2,}|(\. ){4,})")
FIN_NUMERO = re.compile(r"\s\d{1,3}\s*$")
LIGNE_NUMERO = re.compile(r"^\s*(page\s*)?\d{1,3}(\s*(/|sur)\s*\d{1,3})?\s*$", re.I)
CESURE = re.compile(r"\w-\n\w")
PONCT_OK = set(".,;:'’\"()[]-–—/%°«»?!&+=*€$#_…·•")
ART_NUM = re.compile(r"^\s*(?:art(?:icle)?\.?|المادة)\s*(1er|premi[eè]re?|الأولى|\d+)", re.I)
TITRE_KEYWORDS = re.compile(r"(?i)(article|art\.|chapitre|section|page|annexe|lot|prix|المادة|الفصل|الباب)")

class Rapport:
    def __init__(self):
        self.d = {}

    def ajouter(self, doc, code, ex=None, niveau=None, detail=""):
        e = self.d.setdefault((doc, code), {"niveau": niveau or CODES[code][0],
                                            "n": 0, "ex": [], "detail": ""})
        e["n"] += 1
        if ex:
            e["ex"].append(ex)
        if detail:
            e["detail"] = detail

    def niveau_doc(self, doc):
        niv = [e["niveau"] for (d, _), e in self.d.items() if d == doc]
        return min(niv, key=ORDRE.get) if niv else "OK"

def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t)).strip().lower()

def ex(c, n=70):
    t = re.sub(r"\s+", " ", txt(c)).strip()
    return f"{c.get('chunk_id', '?')} (p.{c.get('page_start', '?')}) « {t[:n]}{'…' if len(t) > n else ''} »"

def plages(pages):
    pages, out = sorted(pages), []
    i = 0
    while i < len(pages):
        j = i
        while j + 1 < len(pages) and pages[j + 1] == pages[j] + 1:
            j += 1
        out.append(str(pages[i]) if i == j else f"{pages[i]}-{pages[j]}")
        i = j + 1
    return ", ".join(out)

def corps(c):
    titre = c["title"].strip()
    lignes = [l.strip() for l in txt(c).splitlines() if l.strip()]
    return " ".join(l for l in lignes if l != titre)

def est_sommaire(c, texte_corps):
    if _EST_SOMMAIRE:
        return _EST_SOMMAIRE(txt(c))
    t = txt(c)
    if POINTS_SUITE.search(t):
        return True
    lignes = [l for l in t.splitlines() if l.strip()]
    nb_titres = sum(bool(HEADING_RE.match(l)) for l in lignes)
    if nb_titres >= 3 and len(t) / nb_titres < 120:
        return True
    return len(texte_corps) < 150 and bool(FIN_NUMERO.search(texte_corps)) and not texte_corps.rstrip().endswith(".")

def numero_article(titre):
    m = ART_NUM.match(titre)
    if not m:
        return None
    g = m.group(1).lower()
    return 1 if not g.isdigit() else int(g)

def controles_chunks(doc, chunks, rapport):
    valides = []
    for c in chunks:
        pb = [k for k, t in REQUIS.items() if not isinstance(c.get(k), t)]
        if not pb and (c["page_start"] < 1 or c["page_end"] < c["page_start"]
                       or c["source"] not in ("native", "ocr")):
            pb = ["pages/source"]
        if pb:
            rapport.ajouter(doc, "SCHEMA", ex(c) + f"  -> {pb}")
            continue
        valides.append(c)

    ids = Counter(c["chunk_id"] for c in valides)
    for i, n in ids.items():
        if n > 1:
            rapport.ajouter(doc, "ID_DUP", f"{i} ×{n}")

    pages_doc = max((c["page_end"] for c in valides), default=0)
    limite = PAGE_ATOMIQUE_MAX * 1.05 if pages_doc <= AVIS_MAX_PAGES else CHUNK_SIZE * 1.25
    for c in valides:
        t = txt(c)
        L = len(t.strip())
        if L == 0:
            rapport.ajouter(doc, "VIDE", ex(c))
            continue
        if MOJIBAKE.search(t) or CONTROLE.search(t):
            rapport.ajouter(doc, "ENCODAGE", ex(c))
        if L < MIN_CHARS:
            if TITRE_KEYWORDS.search(t):
                rapport.ajouter(doc, "TITRE_SEUL", ex(c))
            else:
                rapport.ajouter(doc, "TROP_PETIT", ex(c))
        if L > 3000:
            rapport.ajouter(doc, "TROP_GRAND", ex(c))
        elif L > limite:
            rapport.ajouter(doc, "GRAND", ex(c))
        if CESURE.search(t):
            rapport.ajouter(doc, "CESURE")

        b = corps(c)
        nb_lettres = sum(ch.isalpha() for ch in t)
        nb_chiffres = sum(ch.isdigit() for ch in t)
        ratio_lettres = nb_lettres / L
        ratio_chiffres = nb_chiffres / L

        if len(b) < 20:
            rapport.ajouter(doc, "TITRE_SEUL", ex(c))
        elif est_sommaire(c, b):
            rapport.ajouter(doc, "SOMMAIRE", ex(c))
        elif c["source"] == "ocr" and ocr_douteux(t):
            rapport.ajouter(doc, "OCR_DOUTEUX", ex(c))
        elif ratio_lettres < 0.5:
            rapport.ajouter(doc, "TABLEAU_NUM" if ratio_chiffres > 0.35 else "BRUIT", ex(c))
        elif L < 50 and not TITRE_KEYWORDS.search(t):
            rapport.ajouter(doc, "MICRO", ex(c))
    return valides

def ocr_douteux(t):
    mots = t.split()
    if not mots:
        return True
    isoles = sum(len(w) == 1 and w.isalpha() and w.lower() not in "àayو" for w in mots)
    bizarres = sum(not (ch.isalnum() or ch.isspace() or ch in PONCT_OK) for ch in t)
    return isoles / len(mots) > 0.3 or bizarres / max(len(t), 1) > 0.15

def controles_document(doc, chunks, rapport):
    n = len(chunks)
    if n == 0:
        return

    vus = defaultdict(list)
    for c in chunks:
        if len(corps(c)) >= 20:
            vus[hashlib.md5(norm(txt(c)).encode()).hexdigest()].append(c)
    for groupe in vus.values():
        if len(groupe) > 1:
            for c in groupe[1:]:
                rapport.ajouter(doc, "DOUBLON", ex(c) + f"  = {groupe[0]['chunk_id']}")

    regressions = [(a, b) for a, b in zip(chunks, chunks[1:]) if b["page_start"] < a["page_start"]]
    for a, b in regressions:
        rapport.ajouter(doc, "ORDRE_PAGES", f"{a['chunk_id']} (p.{a['page_start']}) -> {b['chunk_id']} (p.{b['page_start']})")

    if n >= 12:
        titres = {norm(c["title"]) for c in chunks}
        cnt = Counter()
        for c in chunks:
            lignes = {re.sub(r"\d+", "#", norm(l)) for l in txt(c).splitlines() if len(l.strip()) >= 6}
            cnt.update(l for l in lignes if l not in titres and not HEADING_RE.match(l))
        for ligne, k in cnt.most_common(3):
            if k >= max(5, 0.25 * n):
                rapport.ajouter(doc, "ENTETE_REPETE", f"« {ligne[:80]} » dans {k}/{n} chunks")
        def num_aux_bords(c):
            ls = [l for l in txt(c).splitlines() if l.strip()]
            return bool(ls) and bool(LIGNE_NUMERO.match(ls[0]) or LIGNE_NUMERO.match(ls[-1]))
        avec_num = [c for c in chunks if num_aux_bords(c)]
        if len(avec_num) > 0.1 * n:
            rapport.ajouter(doc, "NUM_PAGE", ex(avec_num[0]),
                            detail=f" ({len(avec_num)}/{n} chunks)")

    utiles = [c for c in chunks if len(corps(c)) >= 20 and not est_sommaire(c, corps(c))]
    nums = {numero_article(c["title"]) for c in utiles} - {None}
    if len(nums) >= 10:
        manquants = sorted(set(range(min(nums), max(nums) + 1)) - nums)
        if manquants:
            rapport.ajouter(doc, "ARTICLES_MANQUANTS",
                            detail=f" — articles {min(nums)}→{max(nums)}, absents : {plages(manquants[:60])}"
                                   f"{' …' if len(manquants) > 60 else ''}")
    pages_doc = max(c["page_end"] for c in chunks)
    nb_titres = sum(bool(HEADING_RE.match(c["title"])) for c in chunks)
    if pages_doc >= 5 and nb_titres < 3:
        rapport.ajouter(doc, "SANS_TITRES", detail=f" — {nb_titres} titre(s) reconnu(s) sur {pages_doc} pages")

    moyenne = sum(len(txt(c)) for c in chunks) / n
    if n >= 8 and (moyenne < 250 or n / max(pages_doc, 1) > 12):
        rapport.ajouter(doc, "FRAGMENTATION", detail=f" — {n} chunks, {moyenne:.0f} car. en moyenne")

def controles_pdf(doc, chunks, rapport):
    chemin = DATA_RAW / f"{doc}.pdf"
    if pymupdf is None or not chemin.exists():
        rapport.ajouter(doc, "PDF_ABSENT")
        return None
    with pymupdf.open(chemin) as pdf:
        total = len(pdf)
        couvertes = set()
        for c in chunks:
            couvertes.update(range(c["page_start"], c["page_end"] + 1))
        manquantes = [p for p in range(1, total + 1) if p not in couvertes]
        reelles, vides = [], []
        for p in manquantes:
            page = pdf[p - 1]
            (reelles if len(page.get_text().strip()) >= 30 or page.get_images() else vides).append(p)
        if reelles:
            rapport.ajouter(doc, "PAGES_PERDUES", detail=f" — pages {plages(reelles)} sur {total}")
        if vides:
            rapport.ajouter(doc, "PAGES_VIDES", detail=f" — pages {plages(vides)}")

        if not any(c["source"] == "ocr" for c in chunks):
            natif = sum(len(re.sub(r"\s", "", pdf[p].get_text())) for p in range(total))
            if natif >= 200:
                dans_chunks = sum(len(re.sub(r"\s", "", txt(c))) for c in chunks)
                r = dans_chunks / natif
                if r < 0.60:
                    rapport.ajouter(doc, "COUVERTURE_BASSE", detail=f" — {r:.0%} du texte du PDF")
                elif r < 0.85:
                    rapport.ajouter(doc, "COUVERTURE_MOYENNE", detail=f" — {r:.0%} du texte du PDF")
                elif r > 1.40:
                    rapport.ajouter(doc, "COUVERTURE_HAUTE", detail=f" — {r:.0%} du texte du PDF")
        return total

def doublons_inter_documents(par_doc, rapport):
    index = defaultdict(set)
    for doc, chunks in par_doc.items():
        for c in chunks:
            t = norm(txt(c))
            if len(t) >= 100:
                index[hashlib.md5(t.encode()).hexdigest()].add(doc)
    for docs in index.values():
        if len(docs) > 1:
            for d in docs:
                rapport.ajouter(d, "DOUBLON_INTER", detail=f" — avec : {', '.join(sorted(x[:25] for x in docs if x != d))}")

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="les alertes font échouer (code 1)")
    ap.add_argument("--exemples", type=int, default=3, help="exemples affichés par problème")
    ap.add_argument("--doc", help="ne valider que les documents contenant ce texte")
    ap.add_argument("--json", help="écrire le rapport dans ce fichier JSON")
    args = ap.parse_args()

    if not CHUNKS_FILE.exists():
        sys.exit("[ERREUR] chunks.json introuvable. Lance d'abord : python -m scripts.run_ingestion")
    try:
        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"[ERREUR] chunks.json illisible : {e}")
    if not isinstance(chunks, list) or not chunks:
        sys.exit("[ERREUR] Aucun chunk. data/raw/ contient-il des PDF lisibles ?")

    par_doc = defaultdict(list)
    for c in chunks:
        par_doc[c.get("doc", "?") if isinstance(c, dict) else "?"].append(c)
    if args.doc:
        par_doc = {d: v for d, v in par_doc.items() if args.doc.lower() in d.lower()}
        if not par_doc:
            sys.exit(f"Aucun document ne contient « {args.doc} ».")

    rapport, lignes = Rapport(), []
    for doc in sorted(par_doc):
        valides = controles_chunks(doc, par_doc[doc], rapport)
        controles_document(doc, valides, rapport)
        pages_pdf = controles_pdf(doc, valides, rapport) if valides else None
        tailles = [len(txt(c)) for c in valides] or [0]
        lignes.append({
            "doc": doc, "chunks": len(valides),
            "pages": f"{len({p for c in valides for p in range(c['page_start'], c['page_end'] + 1)})}/{pages_pdf or '?'}",
            "ocr": sum(c["source"] == "ocr" for c in valides),
            "min": min(tailles), "moy": sum(tailles) // len(tailles), "max": max(tailles),
        })
    doublons_inter_documents(par_doc, rapport)

    print(f"{'Document':46} {'Chunks':>6} {'Pages':>8} {'OCR':>4} {'Min':>5} {'Moy':>5} {'Max':>5}  Statut")
    print("-" * 94)
    for l in lignes:
        statut = {"ERREUR": "[ERREUR]", "ALERTE": "[ALERTE]", "INFO": "[OK]", "OK": "[OK]"}[rapport.niveau_doc(l["doc"])]
        print(f"{l['doc'][:46]:46} {l['chunks']:>6} {l['pages']:>8} {l['ocr']:>4} "
              f"{l['min']:>5} {l['moy']:>5} {l['max']:>5}  {statut}")
    print("(Pages = pages couvertes par des chunks / pages du PDF)")

    comptes = Counter()
    codes_vus = set()
    for doc in sorted(par_doc):
        entrees = sorted(((c, e) for (d, c), e in rapport.d.items() if d == doc),
                         key=lambda x: (ORDRE[x[1]["niveau"]], x[0]))
        if not entrees:
            continue
        print(f"\n■ {doc}")
        for code, e in entrees:
            comptes[e["niveau"]] += 1
            codes_vus.add(code)
            print(f"  {ICONE[e['niveau']]} [{code}] {CODES[code][1]} — {e['n']} cas{e['detail']}")
            for x in e["ex"][:args.exemples]:
                print(f"       · {x}")
            if len(e["ex"]) > args.exemples:
                print(f"       · … +{len(e['ex']) - args.exemples} autres")

    pistes = [(c, CODES[c][2]) for c in sorted(codes_vus, key=lambda c: ORDRE[CODES[c][0]]) if CODES[c][2]]
    if pistes:
        print("\n=== PISTES DE CORRECTION ===")
        for c, p in pistes:
            print(f"  [{c}] {p}")

    print("\n=== VERDICT ===")
    print(f"{len(chunks)} chunks · {len(par_doc)} documents · "
          f"{comptes['ERREUR']} type(s) d'erreur · {comptes['ALERTE']} d'alerte · {comptes['INFO']} d'info")
    echec = comptes["ERREUR"] > 0 or (args.strict and comptes["ALERTE"] > 0)
    if comptes["ERREUR"]:
        print("[ERREUR] Ingestion NON validée : corrige les erreurs avant les embeddings.")
    elif comptes["ALERTE"]:
        print("[ALERTE]  Ingestion utilisable mais dégradée : corrige les alertes pour une meilleure recherche."
              + ("  (--strict : échec)" if args.strict else ""))
    else:
        print("[OK] Ingestion validée. Prêt pour les embeddings.")

    if args.json:
        sortie = [{"doc": d, "code": c, "niveau": e["niveau"], "n": e["n"], "detail": e["detail"].strip(" —"),
                   "exemples": e["ex"][:20]} for (d, c), e in sorted(rapport.d.items())]
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(sortie, f, ensure_ascii=False, indent=2)
        print(f"Rapport JSON -> {args.json}")
    sys.exit(1 if echec else 0)

if __name__ == "__main__":
    main()