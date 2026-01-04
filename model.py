import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Available models
# -----------------------------
MODELS = {
    "qwen-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "dtype": torch.float32,
        "device": "cpu"
    },
    # Future models (examples)
    # "qwen-7b": {
    #     "name": "Qwen/Qwen2.5-7B-Instruct",
    #     "dtype": torch.float16,
    #     "device": "cpu"
    # },
    # "llama-3": {
    #     "name": "meta-llama/Meta-Llama-3-8B-Instruct",
    #     "dtype": torch.float16,
    #     "device": "cpu"
    # },
}

# -----------------------------
# Model loader
# -----------------------------
def load_model(model_key: str = "qwen-1.5b"):
    if model_key not in MODELS:
        raise ValueError(
            f"Model '{model_key}' not found. Available: {list(MODELS.keys())}"
        )

    cfg = MODELS[model_key]

    print(f"Loading tokenizer: {cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])

    print(f"Loading model: {cfg['name']} (CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        device_map=cfg["device"],
        torch_dtype=cfg["dtype"]
    )

    model.eval()
    return tokenizer, model
