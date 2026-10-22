"""Lancer avec :  python -m pytest -v   (depuis la racine du projet)."""
import io

import pymupdf
import pytest

from src.ingestion import chunker, parser


# ---------- Titres ----------
@pytest.mark.parametrize("ligne", [
    "Article 1 : Objet du marché", "ARTICLE 12 - Délai d'exécution",
    "Art. 3 Caution provisoire", "Article premier", "Chapitre II", "المادة 4",
])
def test_titres_detectes(ligne):
    assert chunker.HEADING_RE.match(ligne)


@pytest.mark.parametrize("ligne", [
    "Le présent article précise les pièces à fournir.",
    "Voir l'article 5 ci-dessus", "Montant : 20 000 dirhams", "",
])
def test_phrases_non_titres(ligne):
    assert not chunker.HEADING_RE.match(ligne)


# ---------- Nettoyage ----------
def test_clean_text_garde_les_retours_a_la_ligne():
    brut = "Article 1 : Objet\r\n\r\n\r\n\r\nTexte   avec    espaces  \nArticle 2 : Prix"
    propre = parser.clean_text(brut)
    assert "\n" in propre
    assert "  " not in propre
    assert "\n\n\n" not in propre
    assert any(chunker.HEADING_RE.match(l) for l in propre.splitlines())


# ---------- Découpage long ----------
def test_texte_long_respecte_la_taille():
    morceaux = chunker.split_long("Phrase de test numéro un. " * 300)
    assert len(morceaux) > 1
    assert all(len(m) <= chunker.CHUNK_SIZE for m in morceaux)


def test_debut_et_fin_conserves():
    morceaux = chunker.split_long("Debut" + "x" * 4000 + " fin.")
    assert morceaux[0].startswith("Debut") and morceaux[-1].endswith("fin.")


# ---------- Chunking ----------
def pages_exemple():
    return [
        {"page": 1, "source": "native", "qr_codes": [], "text": "Cahier des prescriptions spéciales\nMarché de travaux de voirie."},
        {"page": 2, "source": "native", "qr_codes": [], "text": "Article 1 : Objet du marché\nConstruction d'une route communale en zone rurale."},
        {"page": 3, "source": "ocr", "qr_codes": [], "text": "Article 2 : Caution provisoire\nLe montant est de 20 000 dirhams, à joindre au dossier."},
    ]


def test_un_chunk_par_article_avec_pages_et_source():
    chunks = {c["title"]: c for c in chunker.chunk_document("doc", pages_exemple())}
    assert chunks["Article 1 : Objet du marché"]["page_start"] == 2
    assert chunks["Article 1 : Objet du marché"]["source"] == "native"
    assert chunks["Article 2 : Caution provisoire"]["page_start"] == 3
    assert chunks["Article 2 : Caution provisoire"]["source"] == "ocr"


def test_ids_uniques_champs_et_pas_de_chunk_vide():
    chunks = chunker.chunk_document("doc", pages_exemple())
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        assert set(c) >= {"chunk_id", "doc", "title", "page_start", "page_end", "source", "text"}
        assert len(c["text"]) >= chunker.MIN_CHUNK_CHARS


# ---------- Lecture PDF ----------
@pytest.fixture
def pdf_texte_et_vide(tmp_path):
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((50, 72), "Article 1 : Objet du marche\nConstruction d'une route communale selon les plans.")
    doc.new_page()  # page blanche
    chemin = tmp_path / "t.pdf"
    doc.save(chemin)
    return chemin


def test_page_blanche_sans_image_nest_pas_un_scan(pdf_texte_et_vide):
    pages, st = parser.extract_pdf(pdf_texte_et_vide, use_ocr=True)
    assert st["native"] == 1 and st["vides"] == 1 and st["ocr"] == 0
    assert pages[0]["source"] == "native"


def test_lignes_preservees_dans_pdf_natif(pdf_texte_et_vide):
    pages, _ = parser.extract_pdf(pdf_texte_et_vide, use_ocr=False)
    assert "\n" in pages[0]["text"]


# ---------- OCR (ignoré si Tesseract absent) ----------
@pytest.mark.skipif(not parser.ocr_available(), reason="Tesseract indisponible")
def test_ocr_lit_une_page_scannee(tmp_path, monkeypatch):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1240, 400), "white")
    d = ImageDraw.Draw(img)
    try:
        police = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        police = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    d.text((60, 60), "Article 2 : Caution provisoire", font=police, fill="black")
    d.text((60, 140), "Le montant de la caution est de 20 000 dirhams.", font=police, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=buf.getvalue())
    chemin = tmp_path / "scan.pdf"
    doc.save(chemin)

    monkeypatch.setattr(parser, "OCR_CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    pages, st = parser.extract_pdf(chemin, use_ocr=True)
    assert st["ocr"] == 1
    assert "caution" in pages[0]["text"].lower()
    assert "\n" in pages[0]["text"]


# ---------- Documents sans titres (avis) et précision des pages ----------
def test_document_sans_titres_nest_pas_un_preambule():
    pages = [{"page": 1, "source": "native", "qr_codes": [],
              "text": "AVIS D'APPEL D'OFFRES OUVERT\n\nLe jeudi 12 novembre 2026 il sera procédé à l'ouverture des plis."}]
    chunks = chunker.chunk_document("avis", pages)
    assert chunks
    titre = chunks[0]["title"]
    assert "Page 1" in titre or "AVIS" in titre


def test_chaque_chunk_a_ses_propres_pages():
    pages = [
        {"page": 1, "source": "ocr", "qr_codes": [], "text": "Première page, texte arabe ou autre de plus de trente caractères."},
        {"page": 2, "source": "native", "qr_codes": [], "text": "Deuxième page, texte français de plus de trente caractères."},
    ]
    ancien = chunker.CHUNK_SIZE
    chunker.CHUNK_SIZE = 70
    try:
        chunks = chunker.chunk_document("d", pages)
    finally:
        chunker.CHUNK_SIZE = ancien
    assert len(chunks) == 2
    assert chunks[0]["page_start"] == chunks[0]["page_end"] == 1 and chunks[0]["source"] == "ocr"
    assert chunks[1]["page_start"] == chunks[1]["page_end"] == 2 and chunks[1]["source"] == "native"


def test_aucun_mot_coupe_au_milieu():
    mots = " ".join(f"mot{i:03d}" for i in range(600))
    for morceau in chunker.split_long(mots):
        assert all(m.startswith("mot") and len(m) == 6 for m in morceau.split())


def test_avis_sans_titres_ne_melange_pas_deux_pages():
    pages = [
        {"page": 1, "source": "ocr", "qr_codes": [], "text": "Page arabe : texte assez long pour dépasser le seuil minimum."},
        {"page": 2, "source": "ocr", "qr_codes": [], "text": "Page française : texte assez long pour dépasser le seuil minimum."},
    ]
    chunks = chunker.chunk_document("avis", pages)
    assert [(c["page_start"], c["page_end"]) for c in chunks] == [(1, 1), (2, 2)]


# ---------- NOUVEAU : QR Codes ----------
def test_qr_codes_injectes_dans_le_texte():
    """Vérifie que les liens QR sont ajoutés au texte du chunk."""
    pages = [
        {"page": 1, "source": "native", "qr_codes": ["https://example.com/qr1"],
         "text": "Article 1 : Objet du marché\nConstruction d'une route communale."},
    ]
    chunks = chunker.chunk_document("doc_qr", pages)
    assert len(chunks) == 1
    assert "[Liens QR Codes détectés :" in chunks[0]["text"]
    assert "https://example.com/qr1" in chunks[0]["text"]