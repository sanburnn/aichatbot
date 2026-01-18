from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from model import load_model
from chat import generate_text

from rag.loader import load_file
from rag.embedder import chunk_text, embed_texts, EMBED_DIM, embed_model
from rag.store import VectorStore

vector_store = VectorStore(dim=EMBED_DIM)


# -----------------------------
# App setup
# -----------------------------
app = FastAPI(
    title="Qwen Chatbot API",
    description="Local Qwen AI Chatbot (CPU)",
    version="1.0.0"
)

# -----------------------------
# Load model ONCE (important)
# -----------------------------
tokenizer, model = load_model("qwen-1.5b")
# tokenizer, model = load_model("llama-1b") # waiting for approval Llama

# -----------------------------
# Request / Response schemas
# -----------------------------
class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

class ChatResponse(BaseModel):
    response: str

class AskRequest(BaseModel):
    query: str
    max_new_tokens: int = 128

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(data: ChatRequest):
    answer = generate_text(
        tokenizer=tokenizer,
        model=model,
        prompt=data.prompt,
        max_new_tokens=data.max_new_tokens
    )
    return {"response": answer}



@app.post("/upload")
async def upload_file_endpoint(file: UploadFile = File(...)):
    # save file temporarily
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Load text
    text = load_file(file_path)

    # Chunk text
    chunks = chunk_text(text)

    # Embed chunks
    embeddings = embed_texts(chunks)

    # Insert into FAISS
    vector_store.add(embeddings, chunks)

    return {
        "status": "success",
        "chunks_added": len(chunks),
        "store_size": vector_store.size()
    }

# class RAGRequest(BaseModel):
#     query: str
#     max_new_tokens: int = 128

# @app.post("/chat-rag")
# def rag_chat_endpoint(data: RAGRequest):
#     answer = answer_with_rag(
#         tokenizer=tokenizer,
#         model=model,
#         query=data.query,
#         vector_store=vector_store,
#         max_new_tokens=data.max_new_tokens
#     )
#     return {"response": answer}


# -----------------------------
# RAG ANSWERING FUNCTION
# -----------------------------
def answer_with_rag(tokenizer, model, query, vector_store, max_new_tokens=128):
    # 1️⃣ Embed question
    query_embedding = embed_texts([query])[0]

    # 2️⃣ Search vector store
    results = vector_store.search(query_embedding, top_k=3)

    if not results:
        context = "No relevant content found in uploaded documents."
    else:
        context = "\n\n".join([r["text"] for r in results])

    # 3️⃣ Build prompt
    prompt = f"""
Use the context below to answer the question. 
If context is not relevant, say you cannot find the answer.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    # 4️⃣ Generate text
    response = generate_text(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens
    )

    return response.strip()


# -----------------------------
# RAG /ask ENDPOINT
# -----------------------------
@app.post("/ask")
def ask_endpoint(data: AskRequest):
    answer = answer_with_rag(
        tokenizer=tokenizer,
        model=model,
        query=data.query,
        vector_store=vector_store,
        max_new_tokens=data.max_new_tokens
    )
    return {"response": answer}

@app.post("/ask-file")
async def ask_file_endpoint(
    file: UploadFile = File(...),
    query: str = Form(...),
    max_new_tokens: int = Form(128)
):
    # Save file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Load + process document
    text = load_file(file_path)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    local_store = VectorStore(dim=EMBED_DIM)
    local_store.add(embeddings, chunks)

    # RAG answer
    answer = answer_with_rag(
        tokenizer=tokenizer,
        model=model,
        query=query,
        vector_store=local_store,
        max_new_tokens=max_new_tokens
    )

    return { "response": answer }