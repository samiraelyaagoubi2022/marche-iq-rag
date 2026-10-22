import pymupdf, pytest
from src.ingestion import chunker, parser

@pytest.fixture
def pdf_texte_et_vide(tmp_path):
    doc = pymupdf.open(); p = doc.new_page()
    p.insert_text((50, 72), "Article 1 : Objet du marche\nConstruction d'une route communale selon les plans.")
    doc.new_page(); chemin = tmp_path / "t.pdf"; doc.save(chemin); return chemin

def test_clean_text_garde_les_retours_a_la_ligne():
    brut = "Article 1 : Objet\r\n\r\n\r\n\r\nTexte   avec    espaces  \nArticle 2 : Prix"
    propre = parser.clean_text(brut)
    assert "\n" in propre and "  " not in propre and "\n\n\n" not in propre
    assert any(chunker.HEADING_RE.match(l) for l in propre.splitlines())

def test_page_blanche_sans_image_nest_pas_un_scan(pdf_texte_et_vide):
    pages, st = parser.extract_pdf(pdf_texte_et_vide, use_ocr=True)
    assert st["native"] == 1 and st["vides"] == 1 and st["ocr"] == 0 and pages[0]["source"] == "native"

def test_lignes_preservees_dans_pdf_natif(pdf_texte_et_vide):
    pages, _ = parser.extract_pdf(pdf_texte_et_vide, use_ocr=False)
    assert "\n" in pages[0]["text"]

def test_lecture_triee_pour_avis(pdf_texte_et_vide, monkeypatch):
    monkeypatch.setattr(parser, "SORT_SHORT_DOCS", True)
    pages, st = parser.extract_pdf(pdf_texte_et_vide, use_ocr=False)
    assert st["native"] == 1 and "\n" in pages[0]["text"]

def test_formes_de_presentation_arabes_normalisees():
    brut = "\ufe8d\ufedf\ufee4\ufe8e\ufeaa\ufe94 5\u200f"
    propre = parser.clean_text(brut)
    assert propre == "المادة 5" and chunker.HEADING_RE.match(propre)

def test_ligatures_latines_normalisees():
    assert parser.clean_text("\ufb01n de la proc\u00e9dure") == "fin de la procédure"

def test_cesure_recollee_mais_pas_les_puces():
    assert "fourniture" in parser.clean_text("la fourni-\nture de mobilier")
    assert parser.clean_text("sécurité.\n-\nFeux vifs") == "sécurité.\n-\nFeux vifs"
    assert parser.clean_text("Prix : 20-\n30") == "Prix : 20-\n30"

def test_cache_ocr_depend_de_la_version(tmp_path, monkeypatch, pdf_texte_et_vide):
    monkeypatch.setattr(parser, "OCR_CACHE_DIR", tmp_path)
    a = parser._cache_path(pdf_texte_et_vide, 1)
    monkeypatch.setattr(parser, "CACHE_VERSION", "v3")
    assert a != parser._cache_path(pdf_texte_et_vide, 1)

def test_qr_pas_de_crash_sans_image(pdf_texte_et_vide):
    with pymupdf.open(pdf_texte_et_vide) as d:
        assert parser.extract_qr_codes(d[0]) == []