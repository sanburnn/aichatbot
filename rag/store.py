import faiss
import numpy as np


class VectorStore:
    """
    Simple FAISS-based vector store.
    Uses cosine similarity via inner product.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts: list[str] = []

    def add(self, embeddings, texts: list[str]):
        """
        Add embeddings + corresponding texts.
        """
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts length mismatch")

        vectors = np.asarray(embeddings, dtype="float32")
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding, top_k: int = 5) -> list[str]:
        """
        Search for most relevant chunks.
        """
        query = np.asarray([query_embedding], dtype="float32")
        scores, indices = self.index.search(query, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results

    def size(self) -> int:
        return self.index.ntotal
