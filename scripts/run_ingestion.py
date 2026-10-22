"""Lit tous les PDF de data/raw/ et écrit data/processed/chunks.json.

Usage (depuis la racine du projet) :
    python -m scripts.run_ingestion
    python -m scripts.run_ingestion --no-ocr
"""
import json
import logging
import sys

from src.config import DATA_PROCESSED, DATA_RAW
from src.ingestion.chunker import chunk_document
from src.ingestion.parser import extract_pdf, ocr_available

OUT_FILE = DATA_PROCESSED / "chunks.json"


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
    use_ocr = "--no-ocr" not in sys.argv and ocr_available()
    print(f"OCR : {'activé' if use_ocr else 'désactivé'}\n")

    pdfs = sorted(DATA_RAW.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Aucun PDF dans {DATA_RAW}")

    all_chunks = []
    print(f"{'Document':42} {'natif':>6} {'ocr':>5} {'vide':>5} {'err':>4} {'chunks':>7}")
    for pdf in pdfs:
        pages, st = extract_pdf(pdf, use_ocr=use_ocr)
        chunks = chunk_document(pdf.stem, pages)
        all_chunks.extend(chunks)
        print(f"{pdf.name[:42]:42} {st['native']:>6} {st['ocr']:>5} {st['vides']:>5} "
              f"{st['erreurs_ocr']:>4} {len(chunks):>7}")

    OUT_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ocr = sum(c["source"] == "ocr" for c in all_chunks)
    print(f"\n{len(all_chunks)} chunks ({n_ocr} issus de l'OCR) -> {OUT_FILE}")


if __name__ == "__main__":
    main()
