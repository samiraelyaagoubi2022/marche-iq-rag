from pathlib import Path
import os
from dotenv import load_dotenv

# Charge les variables du fichier .env
load_dotenv()

# ==========================================
# CHEMINS DU PROJET
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
QDRANT_PATH = BASE_DIR / "data" / "qdrant"

# Créer les dossiers s'ils n'existent pas
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
QDRANT_PATH.mkdir(parents=True, exist_ok=True)

# ==========================================
# PARAMÈTRES DE CHUNKING
# ==========================================
# Un article de CPS dépasse souvent 500 caractères : on garde la clause entière
# autant que possible, et on ne coupe qu'au-delà de CHUNK_SIZE.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# ==========================================
# PARAMÈTRES OCR (TESSERACT)
# ==========================================
OCR_LANG = os.getenv("OCR_LANG", "fra")

# Chemin de Tesseract : modifiable dans .env (TESSERACT_PATH=...)
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
