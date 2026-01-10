import torch
from search import web_search
from utils import needs_web_search

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
    # ---------------------------
    # 1. NORMAL LOCAL LLM ANSWER
    # ---------------------------
    full_prompt = f"{SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"
    initial_output = call_llm(tokenizer, model, full_prompt, max_new_tokens)

    # Extract plain text
    initial_answer = initial_output.split("Assistant:")[-1].strip()

    # If answer is good → return
    if not needs_web_search(initial_answer):
        return initial_answer

    # ---------------------------
    # 2. FALLBACK: WEB SEARCH
    # ---------------------------
    results = web_search(prompt)

    if len(results) == 0:
        return initial_answer  # No search results → fallback to original

    context = ""
    for r in results:
        context += f"- {r.get('title')}\n{r.get('body')}\n{r.get('href')}\n\n"

    # ---------------------------
    # 3. ASK LLM AGAIN WITH CONTEXT
    # ---------------------------
    enhanced_prompt = f"""
Use the following web search results to answer the user's question accurately.

Search results:
{context}

User question: {prompt}

Final answer:
"""

    final_output = call_llm(tokenizer, model, enhanced_prompt, max_new_tokens)
    final_answer = final_output.strip()

    return final_answer
