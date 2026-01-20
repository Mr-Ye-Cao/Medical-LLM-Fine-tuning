import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Model configurations
MODEL_CONFIGS = {
    "llama3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B",
        "use_quantization": True,
        "requires_hf_token": True,
        "add_special_tokens": ["<|eot_id|>"],
    },
    "olmo2-1b": {
        "name": "../OLMo-2-0425-1B",  # Parent directory (relative to Medical-LLM-Fine-tuning)
        "use_quantization": False,  # Small enough to run in bfloat16
        "requires_hf_token": False,
        "add_special_tokens": [],
    },
    "olmo3-7b-instruct": {
        "name": "../OLMo-3-7B-Instruct",  # Parent directory (relative to Medical-LLM-Fine-tuning)
        "use_quantization": False,  # GH200 has 100GB VRAM - no quantization needed for full fine-tuning
        "requires_hf_token": False,
        "add_special_tokens": [],  # ChatML tokens already in tokenizer
    },
}


def load_model_and_tokenizer(model_type="olmo2-1b", model_path=None):
    """
    Load model and tokenizer.

    Args:
        model_type: One of "llama3-8b" or "olmo2-1b"
        model_path: Override default model path (e.g., checkpoint path for loading fine-tuned weights)

    Returns:
        model, tokenizer
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_type]

    # If model_path is provided, use it; otherwise use default
    if model_path and os.path.isdir(model_path):
        model_name = model_path
    else:
        model_name = config["name"]

    # HuggingFace login if required
    if config["requires_hf_token"]:
        from huggingface_hub import login
        hf_token = os.getenv("HF_TOKEN")
        assert hf_token, "HF_TOKEN environment variable required for this model"
        login(token=hf_token)

    print(f"Loading {model_type} from {model_name}...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # Add special tokens if needed
    if config["add_special_tokens"]:
        tokenizer.add_special_tokens({
            "additional_special_tokens": config["add_special_tokens"]
        })

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading
    if config["use_quantization"]:
        # 4-bit quantization for large models
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
    else:
        # bfloat16 for smaller models (no quantization needed)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    # Device placement
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = model.to(f"cuda:{local_rank}")
    model.config.use_cache = False

    # Resize embeddings if new tokens were added
    if config["add_special_tokens"]:
        model.resize_token_embeddings(len(tokenizer))

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params/1e9:.2f}B params")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    return model, tokenizer
