import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        """
        embeddings: numpy array (N x dim)
        texts: list of strings
        """
        embeddings = np.array(embeddings).astype("float32")

        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have same length")

        # Add to FAISS
        self.index.add(embeddings)

        # Add text chunks
        self.texts.extend(texts)

    def search(self, query_embedding, top_k=3):
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx in indices[0]:
            if idx == -1:
                continue

            #  FIX: avoid error if FAISS returns an index beyond stored texts
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx]
                })

        return results

    def size(self):
        return len(self.texts)
