from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# -----------------------------
# Load embedding model ONCE
# -----------------------------
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using embedding device:", DEVICE)

embed_model = SentenceTransformer(
    EMBED_MODEL_NAME,
    device=DEVICE
)

EMBED_DIM = embed_model.get_sentence_embedding_dimension()


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunk_text = " ".join(chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks



# -----------------------------
# Embedding
# -----------------------------
def embed_texts(texts: list[str]):
    texts = [t for t in texts if t.strip()]  # prevent empty items

    embeddings = embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings)

    # fix shape
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    return embeddings


