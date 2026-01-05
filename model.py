import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "device": "cuda",
        "dtype": torch.float16
    },
    # Future bigger models
    # "qwen-7b": {
    #     "name": "Qwen/Qwen2.5-7B-Instruct",
    #     "device": "cuda",
    #     "dtype": torch.float16
    # },
}
# 
def load_model(model_key: str = "qwen-1.5b"):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    cfg = MODELS[model_key]

    print(f"Loading tokenizer: {cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])

    print(f"Loading model on GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        device_map="auto",        # auto-place on GPU
        torch_dtype=cfg["dtype"]
    )

    model.eval()
    return tokenizer, model
