# rag/rag_answer.py

from rag.embedder import embed_texts
from chat import generate_text


def answer_with_rag(tokenizer, model, query, vector_store, max_new_tokens=128):
    # user query
    query_embedding = embed_texts([query])[0]

    # top-K chunks
    results = vector_store.search(query_embedding, top_k=3)

    if len(results) == 0:
        context = "No relevant information found in the uploaded file."
    else:
        context = "\n".join([r["text"] for r in results])

    prompt = f"""
You are an AI assistant. Answer the user question strictly based on the document context.

Document Context:
{context}

User question:
{query}

Answer:
"""

    return generate_text(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens
    )

