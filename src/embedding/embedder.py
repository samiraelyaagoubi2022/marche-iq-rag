import json
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "intfloat/multilingual-e5-small"

def load_model():
    logger.info(f"Chargement du modèle {MODEL_NAME}...")
    return SentenceTransformer(MODEL_NAME)

def embed_chunks(chunks: List[Dict[str, Any]], model: SentenceTransformer) -> List[Dict[str, Any]]:
    logger.info(f"Calcul des embeddings pour {len(chunks)} chunks...")
    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks

if __name__ == "__main__":
    from src.config import DATA_PROCESSED
    
    input_file = DATA_PROCESSED / "chunks.json"
    output_file = DATA_PROCESSED / "chunks_with_embeddings.json"
    
    if not input_file.exists():
        logger.error(f"Fichier introuvable : {input_file}")
        exit(1)
        
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    model = load_model()
    chunks_with_emb = embed_chunks(chunks, model)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_with_emb, f, ensure_ascii=False)
        
    logger.info(f"[OK] {len(chunks_with_emb)} chunks avec embeddings sauvegardés")
    logger.info(f"Dimension des vecteurs : {len(chunks_with_emb[0]['embedding'])}")