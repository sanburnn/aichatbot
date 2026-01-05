from fastapi import FastAPI
from pydantic import BaseModel

from model import load_model
from chat import generate_text

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

# -----------------------------
# Request / Response schemas
# -----------------------------
class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

class ChatResponse(BaseModel):
    response: str

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
