"""Découpage des pages en chunks prêts pour les embeddings.

Stratégie :
  1. nettoyage des en-têtes / pieds de page répétés et des numéros de page ;
  2. document structuré (>= 2 titres « Article / Chapitre ») -> un chunk par article ;
     sinon (avis, courriers) -> un chunk par page (ou par paragraphes si la page est longue) ;
  3. filtres qualité (sommaire, titre seul, bruit) puis fusion des micro-chunks ;
  4. injection du contexte [Document | Section | page] et des liens QR dans le texte.
"""
import re
from collections import Counter
from typing import Any, Dict, List

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

# Seuls les vrais titres structurels découpent le document.
# (Les lignes EN MAJUSCULES ne sont PLUS des titres : « QUATRE CENT TRENTE MILLE DIRHAMS »
#  ou « BULLETIN OFFICIEL » coupaient les chunks et séparaient les montants de leur phrase.)
HEADING_RE = re.compile(
    r"^\s*(?:(?i:article|art\.?|chapitre)\s+|(?:المادة|الفصل)\s*)"     # arabe : espace facultatif (« المادة5 »)
    r"(?:\d+|[٠-٩]+|(?i:premi[eè]re?)|1er|الأولى|[IVXLC]+)\b.*$"
)

MIN_CHUNK_CHARS = 30       # en dessous : jamais gardé
MIN_ALPHA_RATIO = 0.40     # sauf si le chunk contient un mot-clé utile (montant, délai…)
MERGE_MIN_CHARS = 120      # en dessous : fusionné avec le chunk voisin de la même section
AVIS_MAX_PAGES = 3         # document de <= 3 pages sans articles = avis : 1 chunk par page
PAGE_ATOMIQUE_MAX = 2400   # taille max d'une page gardée entière (limite e5 ~512 tokens)
ENTETE_ZONE = 3            # nb de lignes en haut / en bas de page examinées
INJECT_CONTEXT = True      # préfixe « [Document : … | … | p.N] » dans le texte du chunk

POINTS_SUITE = re.compile(r"(\.{4,}|…{2,}|(\. ){4,})")
FIN_NUMERO = re.compile(r"\s\d{1,3}\s*$")
NUM_SEUL = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:/|sur)\s*\d{1,4})?$", re.I)
MOTS_CLES = re.compile(
    r"cautionnement|caution|estimation|montant|dirham|\bdhs?\b|prix|délai|ouverture des plis|ضمان|درهم",
    re.I,
)


# ---------------------------------------------------------------- filtres
def _est_sommaire(texte: str) -> bool:
    """Table des matières : lignes à points de suite + n° de page, ou titres + n° de page."""
    lignes = [l for l in texte.splitlines() if l.strip()]
    if not lignes:
        return False
    # points de suite SUIVIS d'un numéro de page (un formulaire à trous « Nom : ..... » n'en est pas un)
    n_toc = sum(1 for l in lignes if POINTS_SUITE.search(l) and FIN_NUMERO.search(l))
    if n_toc >= 2 or (n_toc >= 1 and len(lignes) <= 2):
        return True
    if len(lignes) >= 3:
        nb_titres = sum(bool(HEADING_RE.match(l)) for l in lignes)
        if nb_titres >= 3 and len(texte) / nb_titres < 120:
            return True
    # un titre (éventuellement sur 2-3 lignes) qui finit par un numéro de page
    return (
        len(lignes) <= 3 and len(texte) < 250
        and bool(HEADING_RE.match(lignes[0]))
        and bool(FIN_NUMERO.search(lignes[-1]))
        and not texte.rstrip().endswith(".")
    )


def _est_titre_seul(texte: str, titre: str) -> bool:
    lignes = [l.strip() for l in texte.splitlines() if l.strip()]
    corps = " ".join(l for l in lignes if l != titre.strip())
    return len(corps) < 20


def _is_valid_chunk(text: str) -> bool:
    """Assez de lettres, sauf si le chunk parle de montants / délais (chiffres utiles)."""
    if len(text) < MIN_CHUNK_CHARS:
        return False
    if MOTS_CLES.search(text):
        return True
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text) >= MIN_ALPHA_RATIO


# ---------------------------------------------------------------- nettoyage des pages
def _cle(ligne: str) -> str:
    return re.sub(r"\d+", "#", ligne.strip().lower())


def _candidate_entete(ligne: str) -> bool:
    """Un en-tête est court, ne finit pas comme une phrase et n'est pas un titre d'article."""
    l = ligne.strip()
    return bool(l) and len(l) <= 100 and not l.endswith(".") and not HEADING_RE.match(l)


def _retirer_entetes(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Retire les en-têtes/pieds de page répétés (« BULLETIN OFFICIEL », « CPS A.O N°… »)
    et les numéros de page isolés, uniquement en haut/bas de page."""
    n = len(pages)
    if n < 6:
        return pages
    compteur: Counter = Counter()
    zones = []
    for p in pages:
        lignes = p["text"].splitlines()
        idx = [i for i, l in enumerate(lignes) if l.strip()]
        zone = set(idx[:ENTETE_ZONE] + idx[-ENTETE_ZONE:])
        zones.append((lignes, zone))
        compteur.update({_cle(lignes[i]) for i in zone if _candidate_entete(lignes[i])})
    seuil = max(4, int(0.3 * n))
    repetes = {k for k, v in compteur.items() if v >= seuil and k not in ("", "#")}
    out = []
    for p, (lignes, zone) in zip(pages, zones):
        gardees = [
            l for i, l in enumerate(lignes)
            if not (i in zone and ((_candidate_entete(l) and _cle(l) in repetes) or NUM_SEUL.match(l.strip())))
        ]
        out.append({**p, "text": "\n".join(gardees).strip()})
    return out


# ---------------------------------------------------------------- découpage long
def _cut_point(text: str, start: int, end: int, size: int) -> int:
    """Point de coupe naturel ; ne coupe jamais juste après « : » (valeur séparée de son libellé)."""
    lo = start + size // 2
    for sep in (". ", "\n", " "):
        pos = end
        while True:
            cut = text.rfind(sep, lo, pos)
            if cut == -1:
                break
            if not text[start:cut].rstrip().endswith(":"):
                return cut + len(sep.rstrip(" ")) if sep == ". " else cut
            pos = cut
    return end


def split_long(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= size:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            end = _cut_point(text, start, end, size)
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start, end)
        if nxt != -1:
            start = nxt + 1
    return [p for p in parts if p]


# ---------------------------------------------------------------- sections
def _titre_page(texte: str, page: int) -> str:
    """Titre lisible pour un document sans articles (au lieu de « Préambule »)."""
    lignes = [l.strip() for l in texte.splitlines() if l.strip()]
    for l in lignes:
        if l.isupper() and re.search(r"AVIS|APPEL D.OFFRES|CONSULTATION|CAHIER|MARCH", l):
            return f"Page {page} — {l[:90]}"
    for l in lignes:
        if l.isupper() and len(l) >= 15:
            return f"Page {page} — {l[:90]}"
    for l in lignes:
        if len(l) >= 12 and sum(c.isalpha() for c in l) >= 8:
            return f"Page {page} — {l[:90]}"
    return f"Page {page}"


def _sections_by_article(pages: List[Dict[str, Any]]):
    sections: List[Dict[str, Any]] = [{"title": None, "pars": []}]
    for p in pages:
        buf: List[str] = []
        qr_codes = p.get("qr_codes", [])

        def flush_par():
            if buf:
                text = "\n".join(buf).strip()
                if text:
                    sections[-1]["pars"].append(
                        {"text": text, "page": p["page"], "source": p["source"], "qr_codes": qr_codes})
                buf.clear()

        for line in p["text"].splitlines():
            if HEADING_RE.match(line):
                flush_par()
                sections.append({"title": line.strip(), "pars": []})
                buf.append(line)
            elif not line.strip():
                flush_par()
            else:
                buf.append(line)
        flush_par()

    sections = [s for s in sections if s["pars"]]
    for s in sections:                      # texte avant le 1er article : titre = 1re ligne
        if s["title"] is None:
            premiere = s["pars"][0]["text"].splitlines()[0].strip()
            s["title"] = premiere[:100] or "Préambule"
    return sections


def _sections_by_paragraph(pages: List[Dict[str, Any]], atomique: bool):
    sections = []
    for p in pages:
        texte = p["text"].strip()
        if not texte:
            continue
        entiere = atomique and len(texte) <= PAGE_ATOMIQUE_MAX
        paras = [texte] if entiere else [x.strip() for x in re.split(r"\n\s*\n", texte) if x.strip()]
        sec = {"title": _titre_page(texte, p["page"]), "pars": []}
        for para in paras:
            sec["pars"].append({"text": para, "page": p["page"], "source": p["source"],
                                "qr_codes": p.get("qr_codes", []), "atomic": entiere})
        sections.append(sec)
    return sections


# ---------------------------------------------------------------- fusion des petits morceaux
def _joindre(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {"text": a["text"] + "\n\n" + b["text"], "pages": a["pages"] + b["pages"],
            "ocr": a["ocr"] or b["ocr"], "qr": a["qr"] + b["qr"]}


def _fusionner_petits(morceaux: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    limite = CHUNK_SIZE * 1.25
    out: List[Dict[str, Any]] = []
    for m in morceaux:
        if out and len(m["text"]) < MERGE_MIN_CHARS and len(out[-1]["text"]) + len(m["text"]) + 2 <= limite:
            out[-1] = _joindre(out[-1], m)
        else:
            out.append(m)
    if len(out) >= 2 and len(out[0]["text"]) < MERGE_MIN_CHARS \
            and len(out[0]["text"]) + len(out[1]["text"]) + 2 <= limite:
        out[:2] = [_joindre(out[0], out[1])]
    return out


# ---------------------------------------------------------------- point d'entrée
def chunk_document(doc_name: str, pages: List[Dict[str, Any]],
                   with_context: bool = INJECT_CONTEXT) -> List[Dict[str, Any]]:
    pages = _retirer_entetes(pages)
    nb_titres = sum(bool(HEADING_RE.match(l)) for p in pages for l in p["text"].splitlines())
    if nb_titres >= 2:
        sections = _sections_by_article(pages)
    else:
        sections = _sections_by_paragraph(pages, atomique=len(pages) <= AVIS_MAX_PAGES)

    chunks: List[Dict[str, Any]] = []
    for sec in sections:
        buf: List[Dict[str, Any]] = []
        morceaux: List[Dict[str, Any]] = []
        size = 0

        def flush():
            nonlocal size
            if not buf:
                return
            text = "\n\n".join(b["text"] for b in buf)
            entiere = all(b.get("atomic") for b in buf)
            if (entiere or not _est_sommaire(text)) \
                    and not _est_titre_seul(text, sec["title"]) \
                    and _is_valid_chunk(text):
                morceaux.append({
                    "text": text,
                    "pages": [b["page"] for b in buf],
                    "ocr": any(b["source"] == "ocr" for b in buf),
                    "qr": [q for b in buf for q in b.get("qr_codes", [])],
                })
            buf.clear()
            size = 0

        for par in sec["pars"]:
            if par.get("atomic"):
                pieces = [par["text"]]
            else:
                pieces = split_long(par["text"]) if len(par["text"]) > CHUNK_SIZE else [par["text"]]
            for piece in pieces:
                if buf and size + len(piece) + 2 > CHUNK_SIZE:
                    flush()
                buf.append({"text": piece, "page": par["page"], "source": par["source"],
                            "qr_codes": par.get("qr_codes", []), "atomic": par.get("atomic", False)})
                size += len(piece) + 2
        flush()

        for m in _fusionner_petits(morceaux):
            p0, p1 = min(m["pages"]), max(m["pages"])
            text = m["text"]
            qr = list(dict.fromkeys(m["qr"]))
            if qr:
                text += "\n\n[Liens QR Codes détectés : " + ", ".join(qr) + "]"
            if with_context:
                lieu = f"p.{p0}" if p0 == p1 else f"p.{p0}-{p1}"
                text = f"[Document : {doc_name} | {sec['title']} | {lieu}]\n{text}"
            chunks.append({
                "chunk_id": f"{doc_name}::{len(chunks):04d}",
                "doc": doc_name,
                "title": sec["title"],
                "page_start": p0,
                "page_end": p1,
                "source": "ocr" if m["ocr"] else "native",
                "text": text,
            })
    return chunks