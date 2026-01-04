import torch

SYSTEM_PROMPT = "Short answer but straight to the points!"

def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 128
) -> str:
    full_prompt = f"{SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("Assistant:")[-1].strip()
