"""Lecture des PDF, page par page : texte natif (PyMuPDF) ou OCR (Tesseract) + QR Codes.

Test rapide, depuis la racine du projet :
    python -m src.ingestion.parser
"""
import hashlib
import io
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pymupdf
import pytesseract
from PIL import Image

from src.config import DATA_PROCESSED, OCR_LANG, TESSERACT_PATH

# Import optionnel de pyzbar (si non installé, la détection QR est désactivée)
try:
    from pyzbar.pyzbar import decode as decode_qr
    PYZBAR_OK = True
except Exception:  # ImportError, ou DLL zbar introuvable sous Windows
    PYZBAR_OK = False
    decode_qr = None

logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

OCR_CACHE_DIR = DATA_PROCESSED / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_CHARS = 20
OCR_DPI = 300
OCR_CONFIG = "--oem 3 --psm 3"
CACHE_VERSION = "v2"            # à incrémenter si clean_text() change : invalide le cache OCR

DEHYPHENATE = True              # recolle « fourni-\nture » -> « fourniture »
SORT_SHORT_DOCS = False         # True : ordre de lecture visuel pour les avis (<= 3 pages)
SHORT_DOC_PAGES = 3
QR_MIN_SIDE = 60                # px : en dessous, ce n'est pas un QR lisible

# Formes de présentation arabes (اﻟﺠﺮﻳﺪة) et ligatures latines (ﬁ, ﬂ) -> lettres normales (NFKC)
_PRESENTATION = re.compile(r"[\uFB00-\uFB06\uFB50-\uFDFF\uFE70-\uFEFF]+")
# Marques de direction invisibles, qui cassent la détection des titres (« المادة5 »)
_BIDI = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_CESURE = re.compile(r"(?<=[a-zàâäçéèêëîïôöûùüÿœ])-\n(?=[a-zàâäçéèêëîïôöûùüÿœ])")


def ocr_available() -> bool:
    """Vérifie que Tesseract et les langues demandées sont installés."""
    try:
        langs = pytesseract.get_languages()
    except Exception as e:
        logger.warning("Tesseract introuvable (%s). OCR désactivé.", e)
        return False
    missing = [lang for lang in OCR_LANG.split("+") if lang not in langs]
    if missing:
        logger.warning("Langues Tesseract manquantes : %s. OCR désactivé.", missing)
        return False
    return True


def clean_text(text: str) -> str:
    """Nettoie le texte SANS supprimer les retours à la ligne."""
    text = _PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    text = _BIDI.sub("", text)
    text = text.replace("\r", "")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    if DEHYPHENATE:
        text = _CESURE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _cache_path(pdf_path: Path, page_num: int) -> Path:
    """Chemin de cache unique pour l'OCR (change si le PDF, la langue, le DPI ou le nettoyage changent)."""
    key = (f"{pdf_path.name}|{pdf_path.stat().st_size}|{page_num}|{OCR_LANG}|{OCR_DPI}"
           f"|{OCR_CONFIG}|{CACHE_VERSION}")
    return OCR_CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.txt"


def is_page_scanned(page: "pymupdf.Page", text: str) -> bool:
    """Combine plusieurs signaux pour décider si une page est un scan."""
    images = page.get_images(full=True)
    n_images = len(images)

    if len(text) < MIN_TEXT_CHARS:
        return n_images > 0

    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return False

    image_area = 0.0
    for img in images:
        try:
            bbox = page.get_image_bbox(img)
            image_area += bbox.width * bbox.height
        except Exception as e:
            logger.debug("bbox image illisible : %s", e)
    image_coverage = image_area / page_area

    text_area = 0.0
    try:
        for block in page.get_text("blocks"):
            if block[6] == 0:
                text_area += (block[2] - block[0]) * (block[3] - block[1])
    except Exception:
        pass
    text_coverage = text_area / page_area

    if image_coverage > 0.5 and text_coverage < 0.05:
        return True
    if len(text) < 50 and image_coverage > 0.2:
        return True
    return False


def _ocr_page(page: "pymupdf.Page", pdf_path: Path, page_num: int) -> str:
    """OCR d'une page avec cache disque."""
    cache = _cache_path(pdf_path, page_num)
    if cache.exists():
        logger.debug("Page %d : OCR récupéré du cache.", page_num)
        return cache.read_text(encoding="utf-8")
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = clean_text(pytesseract.image_to_string(img, lang=OCR_LANG, config=OCR_CONFIG))
    cache.write_text(text, encoding="utf-8")
    return text


def extract_qr_codes(page: "pymupdf.Page") -> List[str]:
    """Décode les QR codes présents dans les images d'une page.

    Seules les images à peu près carrées et assez grandes sont testées : cela évite de décoder
    inutilement les scans pleine page (lent) et les logos/filets."""
    if not PYZBAR_OK:
        return []
    qr_data: List[str] = []
    for img_info in page.get_images(full=True):
        try:
            base_image = page.parent.extract_image(img_info[0])
            w, h = base_image.get("width", 0), base_image.get("height", 0)
            if min(w, h) < QR_MIN_SIDE or not (0.8 <= w / h <= 1.25):
                continue
            image = Image.open(io.BytesIO(base_image["image"]))
            for d in decode_qr(image):
                data = d.data.decode("utf-8", errors="ignore")
                if data and data not in qr_data:
                    qr_data.append(data)
        except Exception as e:
            logger.debug("QR illisible : %s", e)
    return qr_data


def extract_pdf(pdf_path: Path, use_ocr: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Retourne (pages, stats)."""
    stats = {"native": 0, "ocr": 0, "vides": 0, "erreurs_ocr": 0, "qr_codes": 0}
    pages: List[Dict[str, Any]] = []

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        logger.error("Impossible d'ouvrir %s : %s", pdf_path.name, e)
        return pages, stats

    with doc:
        if doc.needs_pass:
            logger.error("%s est protégé par mot de passe : ignoré.", pdf_path.name)
            return pages, stats
        trier = SORT_SHORT_DOCS and doc.page_count <= SHORT_DOC_PAGES

        for page_num, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text", sort=True) if trier else page.get_text())
            source = "native"

            if is_page_scanned(page, text):
                if not use_ocr:
                    stats["vides"] += 1
                    continue
                try:
                    text = _ocr_page(page, pdf_path, page_num)
                    source = "ocr"
                except Exception as e:
                    logger.error("Erreur OCR %s p.%d : %s", pdf_path.name, page_num, e)
                    stats["erreurs_ocr"] += 1
                    continue

            if len(text) < MIN_TEXT_CHARS:
                logger.warning("%s p.%d : page vide ou illisible.", pdf_path.name, page_num)
                stats["vides"] += 1
                continue

            qr_codes = extract_qr_codes(page)
            stats["qr_codes"] += len(qr_codes)

            pages.append({"page": page_num, "text": text, "source": source, "qr_codes": qr_codes})
            stats[source] += 1

    return pages, stats


if __name__ == "__main__":
    from src.config import DATA_RAW

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    pdfs = sorted(DATA_RAW.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("Aucun PDF dans data/raw/")
    result, st = extract_pdf(pdfs[0], use_ocr=ocr_available())
    print(pdfs[0].name, st)
    if result:
        print("-" * 50)
        print(result[0]["text"][:500])
        if result[0]["qr_codes"]:
            print("QR Codes trouvés :", result[0]["qr_codes"])
        print("-" * 50)