import torch
from search import web_search
from utils import needs_web_search
from rag.embedder import embed_texts

SYSTEM_PROMPT = "Short answer but straight to the points!"

def call_llm(tokenizer, model, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 128) -> str:

    full_prompt = f"{SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"
    initial_output = call_llm(tokenizer, model, full_prompt, max_new_tokens)

    initial_answer = initial_output.split("Assistant:")[-1].strip()

    if not needs_web_search(initial_answer):
        return initial_answer

    results = web_search(prompt)

    if len(results) == 0:
        return initial_answer  

    context = ""
    for r in results:
        context += f"- {r.get('title')}\n{r.get('body')}\n{r.get('href')}\n\n"


    enhanced_prompt = f"""
Answer the question using ONLY the information below. 
Respond in ONE short sentence. English only.

Information:
{context}

Question: {prompt}

Answer:
"""

    final_output = call_llm(tokenizer, model, enhanced_prompt, max_new_tokens)
    # final_answer = final_output.strip()

    # return final_answer
    # Extract text AFTER the last "Answer:"
    if "Answer:" in final_output:
        final_answer = final_output.split("Answer:")[-1].strip()
    else:
        final_answer = final_output.strip()

    return final_answer



def answer_with_rag(tokenizer, model, query, vector_store, max_new_tokens=128):
    q_embed = embed_texts([query])[0]

    retrieved = vector_store.search(q_embed, top_k=5)

    context = "\n\n".join(retrieved)

    prompt = f"""
Use ONLY the information below to answer.
Be short, clear, and accurate.

Context:
{context}

Question:
{query}

Answer:
"""

    return call_llm(tokenizer, model, prompt, max_new_tokens)


def extract_final(s: str, marker:str, sep:str | None=None, default:str="")->str:
    idx = s.find(marker)
    if idx ==-1:
        return default
    after = s[idx + len(marker):]
    return after.split(sep, 1)[0] if sep else after 