from sentence_transformers import SentenceTransformer
import torch

# -----------------------------
# Load embedding model ONCE
# -----------------------------
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for embedding model")

embed_model = SentenceTransformer(
    EMBED_MODEL_NAME,
    device="cuda"
)

EMBED_DIM = embed_model.get_sentence_embedding_dimension()


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[str]:
    """
    Split text into overlapping chunks.
    Chunk size is in words (safe for LLMs).
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks


# -----------------------------
# Embedding
# -----------------------------
def embed_texts(texts: list[str]):
    """
    Convert text chunks into normalized embeddings.
    """
    return embed_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )
