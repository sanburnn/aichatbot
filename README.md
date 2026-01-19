# 🧠 Hybrid AI Chatbot (Text + Vision + RAG)

A **GPU-accelerated AI chatbot** built with **Python**, **FastAPI**, **Hugging Face Transformers**, and **RAG (Retrieval-Augmented Generation)**.

Supports:

* 💬 **Text chat** (Qwen / Llama models)
* 📎 **File attachments** (PDF, TXT, Images)
* 🖼️ **Vision understanding** (images via Vision-Language model)
* 📚 **RAG pipeline** for document-based Q&A
* ⚡ **Full GPU inference (CUDA)**

---

## 📁 Project Structure

```
project-root/
├── app.py                
├── api.py                
├── model.py          
├── chat.py               
├── requirements.txt
├── .env
│
├── rag/
│   ├── loader.py          # Load documents (PDF/TXT)
│   ├── embedder.py        # Text embedding model
│   └── store.py           # Vector store (FAISS)
│
├── uploads/               # Uploaded files
└── README.md
```

---

## 🖥️ System Requirements

| Requirement | Minimum                  |
| ----------- | ------------------------ |
| OS          | Windows 10 / 11 (64-bit) |
| Python      | 3.10 – 3.12              |
| GPU         | NVIDIA (CUDA supported)  |
| VRAM        | 8 GB (12 GB recommended) |
| RAM         | 16 GB                    |

---

## 🧪 Tested Models

| Type      | Model                                    |
| --------- | ---------------------------------------- |
| Text      | `Qwen/Qwen2.5-1.5B-Instruct`             |
| Text      | `meta-llama/Llama-3.2-1B-Instruct`       |
| Vision    | `Qwen/Qwen2-VL-2B-Instruct`              |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |

---

## 🚀 Setup Guide (Windows)

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd project-root
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

✅ You should see `(venv)` in your terminal

---

### 3️⃣ Install PyTorch (GPU)

⚠️ **DO NOT install torch from pip directly**

Check your CUDA version:

```bash
nvidia-smi
```

Install PyTorch (example CUDA 12.1):

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate sentencepiece
pip install fastapi uvicorn pydantic
pip install duckduckgo-search llama-index
pip install sentence-transformers faiss-cpu pypdf python-docx
pip install python-multipart

```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```
True
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5️⃣ Hugging Face Login (Required for LLaMA)

```bash
huggingface-cli login
```

or set environment variable:

```bash
set HF_TOKEN=your_token_here
```

---

## ▶️ Running the App

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open browser:

```
http://localhost:8000/docs
```

---

## 🧪 Testing with Postman

### 🔹 Text Chat

**POST** `/chat`

```json
{
  "prompt": "Explain REST API",
  "max_new_tokens":100
}
```

---

### 🔹 Chat with File (RAG)

**POST** `/chat/file`

Form-Data:

* `message`: "Summarize this document"
* `file`: upload PDF/TXT

---

### 🔹 Image Understanding

**POST** `/chat/vision`

Form-Data:

* `message`: "What is in this image?"
* `file`: upload image (PNG/JPG)

---

## 📚 RAG Flow Explained

```
User Question
   ↓
Embed Query
   ↓
Vector Search (FAISS)
   ↓
Relevant Chunks
   ↓
Injected into Prompt
   ↓
LLM Answer
```

---

## ⚙️ Switching Models

In `model_router.py`:

```python
load_model("llama-1b")
# or
load_model("qwen-1.5b")
```

---

## ❗ Common Issues

### ❌ `No matching distribution found for torch`

✔ Install PyTorch from **official CUDA index**

---

### ❌ Model runs on CPU

✔ Check `torch.cuda.is_available()`
✔ Check `device_map="auto"`

---

### ❌ LLaMA outputs garbage

✔ Must use `apply_chat_template()`

---

## 📌 Roadmap

* [ ] Streaming responses
* [ ] Conversation memory
* [ ] Web UI (React / Electron)
* [ ] Multi-file RAG
* [ ] 4-bit quantization

---

## 📜 License

This project follows:

* Model licenses from Hugging Face (Qwen / Meta)
* Code: MIT (recommended)

---

## 🤝 Credits

* Hugging Face 🤗
* Meta AI
* Alibaba Qwen Team

