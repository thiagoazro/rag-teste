
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    df = pd.read_csv("docs.csv")
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    df["doc_id"] = df["doc_id"].astype(str)

    # Cada linha do CSV é tratada como um "documento" (sem chunking) para manter a aula simples.
    texts = (df["title"] + "\n\n" + df["text"]).tolist()

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product + vetores normalizados ≈ cosine similarity
    index.add(emb)

    faiss.write_index(index, str(DATA_DIR / "faiss.index"))
    df.to_parquet(DATA_DIR / "meta.parquet", index=False)

    print("OK: índice salvo em data/faiss.index e metadata em data/meta.parquet")

if __name__ == "__main__":
    main()
